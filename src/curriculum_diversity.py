"""
Curriculum Diversity — Diversity-driven educational content generation.

Provides tools for generating diverse exercises, explanations, examples,
assessments, and adaptive learning materials. Uses template-based generation
with combinatorial variation, cosine-similarity deduplication, entropy-based
diversity measurement, and graph traversal for topic-dependency analysis.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
import string
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations, product
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
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Exercise:
    """A single curriculum exercise."""

    topic: str
    difficulty: str
    text: str
    solution: str
    hints: List[str] = field(default_factory=list)
    exercise_type: str = "computation"


@dataclass
class Example:
    """A concept example grounded in a specific domain."""

    concept: str
    domain: str
    text: str
    complexity: float = 0.5


@dataclass
class Question:
    """An assessment question."""

    topic: str
    text: str
    question_type: str
    answer: str
    difficulty: float = 0.5
    rubric: str = ""


@dataclass
class CoverageReport:
    """Report on topic coverage and gaps."""

    covered_topics: List[str]
    uncovered_topics: List[str]
    coverage_ratio: float
    gap_details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class StudentProfile:
    """Profile of a student for adaptive content."""

    grade_level: int
    learning_style: str  # visual, auditory, reading, kinesthetic
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    prior_topics: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_STOP_WORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "of", "in", "to", "for", "with", "on", "at", "from", "by",
    "about", "as", "into", "through", "during", "before", "after",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "our", "their", "what", "which", "who", "whom",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenisation with punctuation stripping."""
    return [
        w
        for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in _STOP_WORDS
    ]


def _bow_vector(tokens: List[str], vocab: Dict[str, int]) -> np.ndarray:
    """Build a bag-of-words vector for *tokens* over *vocab*."""
    vec = np.zeros(len(vocab), dtype=np.float64)
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is not None:
            vec[idx] += 1.0
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _build_vocab(texts: Sequence[str]) -> Dict[str, int]:
    """Build vocabulary index from a collection of texts."""
    vocab: Dict[str, int] = {}
    for text in texts:
        for tok in _tokenize(text):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def _entropy(counts: Sequence[int]) -> float:
    """Shannon entropy in bits from a count distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def _deduplicate_by_similarity(
    texts: List[str], threshold: float = 0.85
) -> List[int]:
    """Return indices of texts to keep after cosine-similarity dedup."""
    if not texts:
        return []
    vocab = _build_vocab(texts)
    if not vocab:
        return list(range(len(texts)))
    vectors = [_bow_vector(_tokenize(t), vocab) for t in texts]
    keep: List[int] = [0]
    for i in range(1, len(texts)):
        duplicate = False
        for j in keep:
            if _cosine_similarity(vectors[i], vectors[j]) >= threshold:
                duplicate = True
                break
        if not duplicate:
            keep.append(i)
    return keep


# ---------------------------------------------------------------------------
# Difficulty helpers
# ---------------------------------------------------------------------------

_DIFFICULTY_SCALE: Dict[str, float] = {
    "easy": 0.2,
    "medium": 0.5,
    "hard": 0.8,
    "advanced": 0.95,
}


def _difficulty_value(label: str) -> float:
    return _DIFFICULTY_SCALE.get(label.lower(), 0.5)


# ---------------------------------------------------------------------------
# Exercise template engine
# ---------------------------------------------------------------------------


class ExerciseTemplateEngine:
    """Template-based exercise generation with slot filling.

    Maintains a registry of templates for each exercise type.  Each template
    is a format-string with named slots that get filled from a topic-specific
    content bank.
    """

    _TEMPLATES: Dict[str, List[str]] = {
        "multiple_choice": [
            "Which of the following best describes {concept}?\n"
            "A) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}",
            "Select the correct statement about {concept}.\n"
            "A) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}",
            "If {premise}, which of the following is true?\n"
            "A) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}",
        ],
        "fill_blank": [
            "Complete the sentence: {sentence_start} ______ {sentence_end}.",
            "Fill in the blank: The {concept} of {context} is ______.",
            "In the context of {concept}, ______ is defined as {definition}.",
        ],
        "word_problem": [
            "{character} needs to {action}. Given that {given}, {question}",
            "A {context} involves {concept}. If {given}, {question}",
            "Consider a situation where {premise}. Using {concept}, {question}",
        ],
        "proof": [
            "Prove that {statement} using {method}.",
            "Show that if {premise}, then {conclusion}.",
            "Demonstrate by {method} that {statement}.",
        ],
        "computation": [
            "Calculate {quantity} given that {given}.",
            "Find the value of {variable} if {equation}.",
            "Evaluate {expression} when {conditions}.",
        ],
    }

    _CHARACTERS: List[str] = [
        "Alice", "Bob", "Carlos", "Diana", "Eve",
        "Frank", "Grace", "Hector", "Ines", "Jamal",
    ]

    _CONTEXTS: List[str] = [
        "a school project", "a science experiment", "a business scenario",
        "a sports event", "a cooking recipe", "a travel plan",
        "a construction project", "a music lesson", "a garden design",
        "a financial investment",
    ]

    _METHODS: List[str] = [
        "induction", "contradiction", "direct proof",
        "contrapositive", "construction", "exhaustion",
    ]

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    # -- content banks keyed by topic -----------------------------------------

    @staticmethod
    def _sub_topics(topic: str) -> List[str]:
        """Return plausible sub-topics for a given topic."""
        base = topic.lower().replace(" ", "_")
        suffixes = [
            "fundamentals", "applications", "properties",
            "theorems", "operations", "representations",
            "transformations", "connections", "extensions",
        ]
        return [f"{base}_{s}" for s in suffixes]

    def _fill_slots(
        self,
        template: str,
        topic: str,
        difficulty: str,
    ) -> Tuple[str, str, List[str]]:
        """Fill template slots and return (text, solution, hints)."""
        sub_topics = self._sub_topics(topic)
        concept = self._rng.choice(sub_topics)
        character = self._rng.choice(self._CHARACTERS)
        context = self._rng.choice(self._CONTEXTS)
        method = self._rng.choice(self._METHODS)
        diff_val = _difficulty_value(difficulty)

        # Generate plausible options for multiple choice
        options = [
            f"{concept} is always positive",
            f"{concept} depends on initial conditions",
            f"{concept} is invariant under transformation",
            f"{concept} requires iterative computation",
        ]
        self._rng.shuffle(options)
        correct_idx = self._rng.randint(0, 3)

        # Build numeric components scaled by difficulty
        magnitude = int(10 + 90 * diff_val)
        variable_name = self._rng.choice(["x", "y", "z", "n", "k", "t"])

        slots: Dict[str, str] = {
            "concept": concept.replace("_", " "),
            "character": character,
            "context": context,
            "method": method,
            "option_a": options[0],
            "option_b": options[1],
            "option_c": options[2],
            "option_d": options[3],
            "premise": f"{concept.replace('_', ' ')} holds for n={magnitude}",
            "conclusion": f"the result follows for n={magnitude + 1}",
            "statement": f"{concept.replace('_', ' ')} satisfies the bound",
            "sentence_start": f"When studying {topic},",
            "sentence_end": f"is a key property of {concept.replace('_', ' ')}",
            "definition": f"the core invariant of {topic}",
            "given": f"{variable_name} = {magnitude}",
            "question": f"find the resulting value of {concept.replace('_', ' ')}",
            "action": f"compute the {concept.replace('_', ' ')}",
            "quantity": f"the {concept.replace('_', ' ')} measure",
            "variable": variable_name,
            "equation": f"{variable_name}^2 + {magnitude} = {magnitude**2}",
            "expression": f"{variable_name}^2 - {magnitude}",
            "conditions": f"{variable_name} = {magnitude}",
        }

        text = template.format_map(defaultdict(lambda: "___", slots))

        # Solution
        answer_letter = chr(ord("A") + correct_idx)
        solution = (
            f"The correct answer involves {concept.replace('_', ' ')}. "
            f"Option {answer_letter} is correct because it captures the "
            f"{difficulty} level relationship."
        )

        # Hints
        hints = [
            f"Consider the definition of {concept.replace('_', ' ')}.",
            f"Think about how {topic} relates to {context}.",
            f"Try working with a smaller value first, e.g. {variable_name}={magnitude // 2}.",
        ]
        return text, solution, hints

    def generate(
        self,
        exercise_type: str,
        topic: str,
        difficulty: str,
    ) -> Exercise:
        """Generate a single exercise from templates."""
        templates = self._TEMPLATES.get(
            exercise_type, self._TEMPLATES["computation"]
        )
        template = self._rng.choice(templates)
        text, solution, hints = self._fill_slots(template, topic, difficulty)
        return Exercise(
            topic=topic,
            difficulty=difficulty,
            text=text,
            solution=solution,
            hints=hints,
            exercise_type=exercise_type,
        )


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------

_EXERCISE_TYPES: List[str] = [
    "multiple_choice",
    "fill_blank",
    "word_problem",
    "proof",
    "computation",
]


def generate_diverse_exercises(
    topic: str,
    difficulty: str,
    n: int,
    *,
    seed: Optional[int] = None,
    similarity_threshold: float = 0.85,
) -> List[Exercise]:
    """Generate *n* diverse exercises on *topic* at *difficulty*.

    Variation axes:
    * exercise_type (cycles through all five types)
    * sub-topic focus (different facets of the topic)
    * complexity modulation around the requested difficulty
    * template selection within each type

    After generation a cosine-similarity deduplication pass removes near
    duplicates, and additional exercises are generated to fill gaps.
    """
    engine = ExerciseTemplateEngine(seed=seed)
    difficulty_variants = {
        "easy": ["easy", "easy", "medium"],
        "medium": ["easy", "medium", "hard"],
        "hard": ["medium", "hard", "advanced"],
        "advanced": ["hard", "advanced", "advanced"],
    }
    diff_options = difficulty_variants.get(difficulty.lower(), [difficulty])

    # Over-generate to allow dedup headroom
    pool: List[Exercise] = []
    attempts = 0
    max_attempts = n * 5
    while len(pool) < n * 2 and attempts < max_attempts:
        etype = _EXERCISE_TYPES[attempts % len(_EXERCISE_TYPES)]
        diff = diff_options[attempts % len(diff_options)]
        ex = engine.generate(etype, topic, diff)
        pool.append(ex)
        attempts += 1

    # Deduplicate
    texts = [ex.text for ex in pool]
    keep_indices = _deduplicate_by_similarity(texts, similarity_threshold)
    unique_pool = [pool[i] for i in keep_indices]

    # Ensure type diversity: pick at most ceil(n / len(types)) per type
    max_per_type = max(1, math.ceil(n / len(_EXERCISE_TYPES)))
    type_buckets: Dict[str, List[Exercise]] = defaultdict(list)
    for ex in unique_pool:
        type_buckets[ex.exercise_type].append(ex)

    result: List[Exercise] = []
    rng = random.Random(seed)
    cycle_types = list(_EXERCISE_TYPES)
    rng.shuffle(cycle_types)
    idx = 0
    while len(result) < n and idx < len(cycle_types) * max_per_type:
        etype = cycle_types[idx % len(cycle_types)]
        bucket = type_buckets.get(etype, [])
        pick_idx = idx // len(cycle_types)
        if pick_idx < len(bucket):
            result.append(bucket[pick_idx])
        idx += 1

    # Fill remaining from pool if needed
    for ex in unique_pool:
        if len(result) >= n:
            break
        if ex not in result:
            result.append(ex)

    return result[:n]


# ---------------------------------------------------------------------------
# Explanation generation
# ---------------------------------------------------------------------------

_ANALOGY_DOMAINS: List[str] = [
    "cooking", "sports", "music", "travel", "building",
    "gardening", "games", "storytelling", "nature", "machines",
]

_FORMALITY_LEVELS: List[str] = ["casual", "standard", "academic", "technical"]

_DETAIL_DEPTHS: List[str] = ["overview", "moderate", "in-depth"]


def _explanation_template(
    concept: str,
    audience: str,
    analogy_domain: str,
    formality: str,
    depth: str,
) -> str:
    """Build an explanation string from combinatorial parameters."""
    # Audience calibration
    audience_intros: Dict[str, str] = {
        "beginner": f"Let's start with the basics of {concept}.",
        "intermediate": f"Building on what you know, {concept} extends to deeper ideas.",
        "advanced": f"{concept} can be analysed through a rigorous lens.",
        "expert": f"At a research level, {concept} intersects several open problems.",
    }
    intro = audience_intros.get(
        audience.lower(),
        f"Here is an explanation of {concept} for a {audience} audience.",
    )

    # Analogy
    analogy = (
        f"Think of {concept} like {analogy_domain}: just as "
        f"{'a chef combines ingredients' if analogy_domain == 'cooking' else 'a player follows strategy'} "
        f"to achieve a goal, {concept} combines its components to produce a result."
    )

    # Formality adjustment
    if formality == "casual":
        tone = f"In plain language, {concept} is basically about patterns and structure."
    elif formality == "academic":
        tone = (
            f"Formally, {concept} is characterised by a set of axioms "
            f"and derived properties that govern its behaviour."
        )
    elif formality == "technical":
        tone = (
            f"From an implementation perspective, {concept} requires "
            f"careful handling of edge cases and computational complexity."
        )
    else:
        tone = f"{concept} involves understanding how parts relate to form a whole."

    # Depth
    if depth == "overview":
        detail = f"At a high level, {concept} matters because it underpins many related ideas."
    elif depth == "in-depth":
        detail = (
            f"Diving deeper, {concept} has several sub-components: "
            f"the foundational definitions, the key theorems, "
            f"and the practical algorithms that follow from them."
        )
    else:
        detail = (
            f"{concept} can be understood through its core properties "
            f"and a few worked examples."
        )

    return f"{intro}\n\n{analogy}\n\n{tone}\n\n{detail}"


def generate_diverse_explanations(
    concept: str,
    audience_levels: List[str],
    n: int,
    *,
    seed: Optional[int] = None,
    similarity_threshold: float = 0.80,
) -> List[str]:
    """Generate *n* diverse explanations of *concept*.

    Varies along four axes — analogy domain, formality, detail depth,
    and audience level — using the Cartesian product to maximise
    combinatorial diversity.  Near-duplicate explanations are removed
    via cosine-similarity filtering.
    """
    rng = random.Random(seed)

    if not audience_levels:
        audience_levels = ["beginner", "intermediate", "advanced"]

    combos = list(
        product(audience_levels, _ANALOGY_DOMAINS, _FORMALITY_LEVELS, _DETAIL_DEPTHS)
    )
    rng.shuffle(combos)

    pool: List[str] = []
    for audience, analogy, formality, depth in combos:
        if len(pool) >= n * 3:
            break
        explanation = _explanation_template(
            concept, audience, analogy, formality, depth
        )
        pool.append(explanation)

    keep = _deduplicate_by_similarity(pool, similarity_threshold)
    unique = [pool[i] for i in keep]
    return unique[:n]


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

_DOMAIN_MAPPINGS: Dict[str, Dict[str, str]] = {
    "physics": {
        "instance": "a ball rolling down an inclined plane",
        "context": "Newtonian mechanics",
        "application": "predicting trajectory and velocity",
    },
    "economics": {
        "instance": "supply and demand equilibrium",
        "context": "microeconomic markets",
        "application": "determining market price",
    },
    "biology": {
        "instance": "enzyme-substrate interaction",
        "context": "molecular biology",
        "application": "understanding reaction kinetics",
    },
    "computer_science": {
        "instance": "sorting an array of integers",
        "context": "algorithm design",
        "application": "efficient data processing",
    },
    "art": {
        "instance": "colour mixing on a palette",
        "context": "visual composition",
        "application": "creating harmonious colour schemes",
    },
    "music": {
        "instance": "chord progression in a song",
        "context": "music theory",
        "application": "composing melodies",
    },
    "cooking": {
        "instance": "balancing flavours in a recipe",
        "context": "culinary arts",
        "application": "creating a balanced dish",
    },
    "sports": {
        "instance": "optimising training schedules",
        "context": "athletic performance",
        "application": "peak performance planning",
    },
    "engineering": {
        "instance": "load distribution in a bridge",
        "context": "structural engineering",
        "application": "safe design under stress",
    },
    "medicine": {
        "instance": "drug dosage optimisation",
        "context": "pharmacology",
        "application": "maximising efficacy while minimising side-effects",
    },
}


def _complexity_for_domain(domain: str, concept: str) -> float:
    """Heuristic complexity score in [0, 1]."""
    technical = {"physics", "engineering", "computer_science", "medicine"}
    h = int(hashlib.md5(f"{domain}:{concept}".encode()).hexdigest()[:4], 16)
    base = 0.6 if domain.lower() in technical else 0.35
    jitter = (h % 100) / 500.0  # small deterministic noise
    return min(1.0, base + jitter)


def generate_diverse_examples(
    concept: str,
    domains: List[str],
    n: int,
    *,
    seed: Optional[int] = None,
) -> List[Example]:
    """Generate *n* examples of *concept* across *domains*.

    Maps the abstract concept to domain-specific instantiations using
    a content bank.  When more examples than domains are requested the
    function cycles through domains with varied phrasing.
    """
    rng = random.Random(seed)
    if not domains:
        domains = list(_DOMAIN_MAPPINGS.keys())

    phrasings = [
        "Consider {instance} in the context of {context}. "
        "This illustrates {concept} because {application}.",
        "An everyday example of {concept} comes from {context}: "
        "{instance}. Its practical use is {application}.",
        "In {context}, {concept} manifests as {instance}, "
        "enabling {application}.",
    ]

    results: List[Example] = []
    for i in range(n * 3):
        if len(results) >= n:
            break
        domain = domains[i % len(domains)]
        mapping = _DOMAIN_MAPPINGS.get(
            domain.lower(),
            {
                "instance": f"a scenario involving {concept}",
                "context": domain,
                "application": f"demonstrating {concept} in {domain}",
            },
        )
        phrasing = phrasings[i % len(phrasings)]
        text = phrasing.format(concept=concept, **mapping)
        complexity = _complexity_for_domain(domain, concept)
        results.append(
            Example(concept=concept, domain=domain, text=text, complexity=complexity)
        )

    # Deduplicate
    texts = [e.text for e in results]
    keep = _deduplicate_by_similarity(texts, 0.85)
    results = [results[i] for i in keep]
    return results[:n]


# ---------------------------------------------------------------------------
# Adaptive diversity
# ---------------------------------------------------------------------------

_LEARNING_STYLE_TEMPLATES: Dict[str, List[str]] = {
    "visual": [
        "Visualise {topic} as a diagram where each component connects to the next.",
        "Imagine a chart showing how {topic} changes across different conditions.",
        "Picture a colour-coded map of {topic} relationships.",
    ],
    "auditory": [
        "Listen to the rhythm of {topic}: each step builds on the previous one.",
        "Narrate the process of {topic} aloud, noting each transition.",
        "Explain {topic} to a friend as if telling a story.",
    ],
    "reading": [
        "Read through the formal definition of {topic} and annotate key terms.",
        "Study the textbook passage on {topic}, highlighting core theorems.",
        "Write a summary paragraph about {topic} in your own words.",
    ],
    "kinesthetic": [
        "Work through a hands-on exercise involving {topic}.",
        "Build a physical model that demonstrates {topic}.",
        "Manipulate objects to explore how {topic} behaves under changes.",
    ],
}


def adaptive_diversity(
    student_profile: StudentProfile,
    topic: str,
    n: int,
    *,
    seed: Optional[int] = None,
) -> List[str]:
    """Generate *n* diverse learning materials adapted to *student_profile*.

    Considers learning style, strengths/weaknesses, prior knowledge, and
    grade level to produce tailored materials.
    """
    rng = random.Random(seed)

    # Determine appropriate difficulty from grade level
    if student_profile.grade_level <= 5:
        difficulty_label = "easy"
    elif student_profile.grade_level <= 8:
        difficulty_label = "medium"
    elif student_profile.grade_level <= 11:
        difficulty_label = "hard"
    else:
        difficulty_label = "advanced"

    diff_val = _difficulty_value(difficulty_label)

    # Primary style templates
    style = student_profile.learning_style.lower()
    primary_templates = _LEARNING_STYLE_TEMPLATES.get(
        style, _LEARNING_STYLE_TEMPLATES["reading"]
    )
    # Secondary style for variety
    all_styles = list(_LEARNING_STYLE_TEMPLATES.keys())
    secondary_style = rng.choice([s for s in all_styles if s != style] or all_styles)
    secondary_templates = _LEARNING_STYLE_TEMPLATES[secondary_style]

    # Build materials
    materials: List[str] = []

    # Relate to prior knowledge
    prior_link = ""
    if student_profile.prior_topics:
        prior = rng.choice(student_profile.prior_topics)
        prior_link = f" Building on your knowledge of {prior},"

    # Address weaknesses explicitly
    weakness_note = ""
    if student_profile.weaknesses:
        weak = rng.choice(student_profile.weaknesses)
        weakness_note = (
            f" Pay special attention to how {topic} relates to {weak}, "
            f"an area where extra practice will help."
        )

    # Leverage strengths
    strength_note = ""
    if student_profile.strengths:
        strong = rng.choice(student_profile.strengths)
        strength_note = (
            f" Use your strength in {strong} to anchor your understanding."
        )

    for i in range(n * 2):
        if len(materials) >= n:
            break
        # Alternate primary / secondary style
        if i % 3 < 2:
            base = rng.choice(primary_templates)
        else:
            base = rng.choice(secondary_templates)

        body = base.format(topic=topic)

        # Compose with adaptive notes
        grade_note = (
            f"[Grade {student_profile.grade_level} — "
            f"{difficulty_label} level]"
        )

        parts = [grade_note, prior_link.strip(), body]
        if i % 2 == 0 and weakness_note:
            parts.append(weakness_note.strip())
        if i % 3 == 0 and strength_note:
            parts.append(strength_note.strip())

        materials.append(" ".join(p for p in parts if p))

    # Deduplicate
    keep = _deduplicate_by_similarity(materials, 0.80)
    materials = [materials[i] for i in keep]
    return materials[:n]


# ---------------------------------------------------------------------------
# Knowledge gap coverage (graph-based)
# ---------------------------------------------------------------------------


def _topological_sort(graph: Dict[str, List[str]]) -> List[str]:
    """Kahn's algorithm for topological ordering of a DAG."""
    in_degree: Dict[str, int] = defaultdict(int)
    all_nodes: Set[str] = set(graph.keys())
    for deps in graph.values():
        for d in deps:
            in_degree[d] += 0  # ensure node exists
            all_nodes.add(d)
    for node in all_nodes:
        in_degree.setdefault(node, 0)
    for node, deps in graph.items():
        for d in deps:
            in_degree[node] += 1  # node depends on d

    queue: deque[str] = deque(
        node for node, deg in in_degree.items() if deg == 0
    )
    order: List[str] = []
    visited: Set[str] = set()
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        # Find nodes that depend on `node`
        for candidate, deps in graph.items():
            if node in deps and candidate not in visited:
                in_degree[candidate] -= 1
                if in_degree[candidate] <= 0:
                    queue.append(candidate)

    # Append any remaining unvisited nodes (cycles)
    for node in all_nodes - visited:
        order.append(node)
    return order


def _bfs_reachable(
    graph: Dict[str, List[str]], start: str
) -> Set[str]:
    """Return all nodes reachable from *start* following dependency edges."""
    visited: Set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for dep in graph.get(node, []):
            if dep not in visited:
                queue.append(dep)
    return visited


def knowledge_gap_coverage(
    student_answers: Dict[str, float],
    topic_graph: Dict[str, List[str]],
    *,
    mastery_threshold: float = 0.7,
) -> CoverageReport:
    """Analyse topic coverage from student answers and a dependency graph.

    Parameters
    ----------
    student_answers:
        Mapping from topic name to a score in [0, 1].
    topic_graph:
        Adjacency list where ``topic_graph[t]`` lists prerequisites of *t*.
    mastery_threshold:
        Minimum score to consider a topic "covered".

    Returns
    -------
    CoverageReport with covered/uncovered topics, coverage ratio,
    gap details, and remediation recommendations.
    """
    all_topics: Set[str] = set(topic_graph.keys())
    for deps in topic_graph.values():
        all_topics.update(deps)

    covered: List[str] = []
    uncovered: List[str] = []
    weak_topics: List[Tuple[str, float]] = []

    for topic in sorted(all_topics):
        score = student_answers.get(topic, 0.0)
        if score >= mastery_threshold:
            covered.append(topic)
        else:
            uncovered.append(topic)
            weak_topics.append((topic, score))

    coverage_ratio = len(covered) / max(len(all_topics), 1)

    # --- gap analysis --------------------------------------------------
    # For each uncovered topic, find downstream topics that are blocked
    gap_details: Dict[str, Any] = {}
    for topic, score in weak_topics:
        downstream_blocked: List[str] = []
        for candidate, deps in topic_graph.items():
            if topic in deps and candidate not in covered:
                downstream_blocked.append(candidate)
        gap_details[topic] = {
            "score": score,
            "prerequisites": topic_graph.get(topic, []),
            "blocks": downstream_blocked,
            "severity": 1.0 - score,
        }

    # --- recommendations -----------------------------------------------
    # Prioritise uncovered topics whose prerequisites are all covered
    # (i.e. topics the student is ready to learn).
    recommendations: List[str] = []
    topo = _topological_sort(topic_graph)
    topo_index = {t: i for i, t in enumerate(topo)}

    ready_to_learn: List[str] = []
    needs_prereqs: List[str] = []
    for topic in uncovered:
        prereqs = topic_graph.get(topic, [])
        if all(p in covered for p in prereqs):
            ready_to_learn.append(topic)
        else:
            missing = [p for p in prereqs if p not in covered]
            needs_prereqs.append(topic)
            gap_details[topic]["missing_prerequisites"] = missing

    # Sort ready-to-learn by topological order (foundational first)
    ready_to_learn.sort(key=lambda t: topo_index.get(t, len(topo)))

    for topic in ready_to_learn:
        score = student_answers.get(topic, 0.0)
        recommendations.append(
            f"Study '{topic}' next (prerequisites met, current score: {score:.0%})."
        )

    if needs_prereqs:
        prereq_topics = set()
        for topic in needs_prereqs:
            for p in topic_graph.get(topic, []):
                if p not in covered:
                    prereq_topics.add(p)
        if prereq_topics:
            recommendations.append(
                f"First address prerequisite gaps: {', '.join(sorted(prereq_topics))}."
            )

    if coverage_ratio >= 0.9:
        recommendations.append("Strong overall coverage — focus on weak spots.")
    elif coverage_ratio >= 0.6:
        recommendations.append("Moderate coverage — targeted practice recommended.")
    else:
        recommendations.append("Significant gaps exist — consider a structured review.")

    return CoverageReport(
        covered_topics=covered,
        uncovered_topics=uncovered,
        coverage_ratio=coverage_ratio,
        gap_details=gap_details,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Diverse assessment
# ---------------------------------------------------------------------------

_BLOOMS_LEVELS: List[str] = [
    "remember", "understand", "apply", "analyse", "evaluate", "create",
]

_QUESTION_STEMS: Dict[str, Dict[str, List[str]]] = {
    "multiple_choice": {
        "remember": ["Which of the following defines {sub_topic}?"],
        "understand": ["Which statement best explains why {sub_topic} matters?"],
        "apply": ["Given a scenario involving {sub_topic}, which approach works?"],
        "analyse": ["Compare the two aspects of {sub_topic} shown below."],
        "evaluate": ["Assess the validity of the claim about {sub_topic}."],
        "create": ["Design a new application of {sub_topic}. Which plan is best?"],
    },
    "short_answer": {
        "remember": ["Define {sub_topic} in one sentence."],
        "understand": ["Explain in your own words how {sub_topic} works."],
        "apply": ["Describe a real-world use of {sub_topic}."],
        "analyse": ["What are the strengths and weaknesses of {sub_topic}?"],
        "evaluate": ["Critique the following claim about {sub_topic}."],
        "create": ["Propose a novel approach to {sub_topic}."],
    },
    "true_false": {
        "remember": ["{sub_topic} is defined as the inverse operation. True or False?"],
        "understand": ["{sub_topic} always yields a unique result. True or False?"],
        "apply": ["{sub_topic} can be applied to all real numbers. True or False?"],
        "analyse": ["{sub_topic} is more efficient than its alternative. True or False?"],
        "evaluate": ["The claim '{sub_topic} is obsolete' is valid. True or False?"],
        "create": ["A combination of {sub_topic} and its dual is always possible. True or False?"],
    },
    "essay": {
        "remember": ["Summarise the key definitions related to {sub_topic}."],
        "understand": ["Explain the significance of {sub_topic} in the broader field."],
        "apply": ["Describe how you would use {sub_topic} to solve a practical problem."],
        "analyse": ["Analyse the relationship between {sub_topic} and its prerequisites."],
        "evaluate": ["Evaluate competing theories about {sub_topic}."],
        "create": ["Develop a research proposal investigating {sub_topic}."],
    },
    "computation": {
        "remember": ["State the formula for {sub_topic}."],
        "understand": ["Derive the expression for {sub_topic} from first principles."],
        "apply": ["Calculate the result of {sub_topic} given the following data."],
        "analyse": ["Determine the error bounds for {sub_topic}."],
        "evaluate": ["Verify whether the given solution to {sub_topic} is correct."],
        "create": ["Construct a new problem that tests understanding of {sub_topic}."],
    },
}

_RUBRIC_TEMPLATES: Dict[str, str] = {
    "remember": "Full marks for accurate recall of definitions and facts.",
    "understand": "Full marks for clear explanation demonstrating comprehension.",
    "apply": "Full marks for correct application with justified steps.",
    "analyse": "Full marks for thorough breakdown with supporting evidence.",
    "evaluate": "Full marks for well-reasoned judgement with criteria.",
    "create": "Full marks for originality, feasibility, and coherence.",
}


def diverse_assessment(
    topic: str,
    question_types: List[str],
    n: int,
    *,
    seed: Optional[int] = None,
) -> List[Question]:
    """Generate *n* diverse assessment questions spanning *question_types*.

    Ensures diversity across Bloom's taxonomy levels and sub-topics.
    """
    rng = random.Random(seed)
    sub_topics = ExerciseTemplateEngine._sub_topics(topic)

    if not question_types:
        question_types = list(_QUESTION_STEMS.keys())

    # Build a pool of (type, bloom, sub_topic) combinations
    combos = list(product(question_types, _BLOOMS_LEVELS, sub_topics))
    rng.shuffle(combos)

    pool: List[Question] = []
    for qtype, bloom, sub_topic in combos:
        if len(pool) >= n * 3:
            break
        stems = _QUESTION_STEMS.get(qtype, _QUESTION_STEMS["short_answer"])
        bloom_stems = stems.get(bloom, stems["remember"])
        stem = rng.choice(bloom_stems)
        text = stem.format(sub_topic=sub_topic.replace("_", " "))

        # Difficulty based on Bloom level
        bloom_idx = _BLOOMS_LEVELS.index(bloom)
        difficulty = (bloom_idx + 1) / len(_BLOOMS_LEVELS)

        answer = (
            f"A complete answer addresses {sub_topic.replace('_', ' ')} "
            f"at the '{bloom}' level of Bloom's taxonomy."
        )
        rubric = _RUBRIC_TEMPLATES.get(bloom, "See instructor guidelines.")

        pool.append(
            Question(
                topic=topic,
                text=text,
                question_type=qtype,
                answer=answer,
                difficulty=difficulty,
                rubric=rubric,
            )
        )

    # Deduplicate
    texts = [q.text for q in pool]
    keep = _deduplicate_by_similarity(texts, 0.85)
    pool = [pool[i] for i in keep]

    # Ensure even distribution across question types and bloom levels
    type_counts: Dict[str, int] = Counter()
    bloom_counts: Dict[str, int] = Counter()
    result: List[Question] = []

    max_per_type = max(1, math.ceil(n / max(len(question_types), 1)))
    max_per_bloom = max(1, math.ceil(n / len(_BLOOMS_LEVELS)))

    for q in pool:
        if len(result) >= n:
            break
        if (
            type_counts[q.question_type] < max_per_type
            and bloom_counts[q.rubric] < max_per_bloom
        ):
            result.append(q)
            type_counts[q.question_type] += 1
            bloom_counts[q.rubric] += 1

    # Fill remaining
    for q in pool:
        if len(result) >= n:
            break
        if q not in result:
            result.append(q)

    return result[:n]


# ---------------------------------------------------------------------------
# Curriculum diversity analyser
# ---------------------------------------------------------------------------


class CurriculumDiversityAnalyzer:
    """Analyses the diversity of a curriculum or set of educational materials.

    Methods compute coverage metrics across topics, Bloom's taxonomy levels,
    difficulty bands, and learning-style modalities.
    """

    def analyze_topic_coverage(
        self, materials: Sequence[str]
    ) -> Dict[str, Any]:
        """Measure how evenly topics are represented in *materials*.

        Returns a dictionary with:
        * ``topic_counts`` — Counter of detected topic tokens
        * ``entropy`` — Shannon entropy of topic distribution (higher = more diverse)
        * ``dominant_topic`` — most frequent topic token
        * ``coverage_score`` — normalised diversity score in [0, 1]
        """
        all_tokens: List[str] = []
        for text in materials:
            all_tokens.extend(_tokenize(text))

        topic_counts = Counter(all_tokens)
        counts = list(topic_counts.values())
        ent = _entropy(counts)
        max_ent = math.log2(max(len(counts), 1)) if counts else 0.0
        coverage_score = ent / max_ent if max_ent > 0 else 0.0
        dominant = topic_counts.most_common(1)[0][0] if topic_counts else ""

        return {
            "topic_counts": dict(topic_counts.most_common(30)),
            "entropy": round(ent, 4),
            "max_entropy": round(max_ent, 4),
            "dominant_topic": dominant,
            "coverage_score": round(coverage_score, 4),
            "num_unique_tokens": len(topic_counts),
        }

    def bloom_level_distribution(
        self, questions: Sequence[Question]
    ) -> Dict[str, Any]:
        """Compute distribution of Bloom's taxonomy levels across *questions*.

        Uses the rubric field and difficulty to infer Bloom level.
        """
        bloom_counter: Counter[str] = Counter()
        for q in questions:
            inferred_bloom = _BLOOMS_LEVELS[
                min(
                    int(q.difficulty * len(_BLOOMS_LEVELS)),
                    len(_BLOOMS_LEVELS) - 1,
                )
            ]
            bloom_counter[inferred_bloom] += 1

        total = sum(bloom_counter.values())
        distribution = {
            level: bloom_counter.get(level, 0) for level in _BLOOMS_LEVELS
        }
        counts = [distribution[l] for l in _BLOOMS_LEVELS]
        ent = _entropy(counts)
        max_ent = math.log2(len(_BLOOMS_LEVELS))
        balance_score = ent / max_ent if max_ent > 0 else 0.0

        return {
            "distribution": distribution,
            "total_questions": total,
            "entropy": round(ent, 4),
            "balance_score": round(balance_score, 4),
            "missing_levels": [l for l in _BLOOMS_LEVELS if distribution[l] == 0],
        }

    def difficulty_distribution(
        self, exercises: Sequence[Exercise]
    ) -> Dict[str, Any]:
        """Analyse the spread of difficulty levels across *exercises*."""
        diff_counter: Counter[str] = Counter()
        for ex in exercises:
            diff_counter[ex.difficulty.lower()] += 1

        all_levels = list(_DIFFICULTY_SCALE.keys())
        distribution = {level: diff_counter.get(level, 0) for level in all_levels}
        counts = [distribution[l] for l in all_levels]
        ent = _entropy(counts)
        max_ent = math.log2(len(all_levels))
        balance_score = ent / max_ent if max_ent > 0 else 0.0

        # Skew: mean difficulty
        diff_values = [_difficulty_value(ex.difficulty) for ex in exercises]
        mean_diff = float(np.mean(diff_values)) if diff_values else 0.0
        std_diff = float(np.std(diff_values)) if diff_values else 0.0

        return {
            "distribution": distribution,
            "total_exercises": len(exercises),
            "entropy": round(ent, 4),
            "balance_score": round(balance_score, 4),
            "mean_difficulty": round(mean_diff, 4),
            "std_difficulty": round(std_diff, 4),
            "missing_levels": [l for l in all_levels if distribution[l] == 0],
        }

    def learning_style_coverage(
        self, materials: Sequence[str]
    ) -> Dict[str, Any]:
        """Estimate which learning styles are addressed by *materials*.

        Uses keyword heuristics to classify each material as targeting
        visual, auditory, reading, or kinesthetic learners.
        """
        style_keywords: Dict[str, List[str]] = {
            "visual": [
                "diagram", "chart", "graph", "picture", "visualise",
                "visualize", "image", "colour", "color", "map", "draw",
                "illustrate", "see", "look", "show",
            ],
            "auditory": [
                "listen", "hear", "say", "speak", "narrate", "discuss",
                "explain aloud", "rhythm", "sound", "tell", "story",
                "conversation", "dialogue",
            ],
            "reading": [
                "read", "write", "text", "book", "article", "note",
                "summary", "paragraph", "definition", "annotate",
                "highlight", "textbook", "study",
            ],
            "kinesthetic": [
                "hands-on", "build", "manipulate", "touch", "move",
                "exercise", "practice", "experiment", "construct",
                "model", "physical", "activity",
            ],
        }

        style_counter: Counter[str] = Counter()
        for text in materials:
            lower = text.lower()
            detected: Set[str] = set()
            for style, keywords in style_keywords.items():
                for kw in keywords:
                    if kw in lower:
                        detected.add(style)
                        break
            if not detected:
                detected.add("reading")  # default
            for s in detected:
                style_counter[s] += 1

        all_styles = list(style_keywords.keys())
        distribution = {s: style_counter.get(s, 0) for s in all_styles}
        counts = [distribution[s] for s in all_styles]
        ent = _entropy(counts)
        max_ent = math.log2(len(all_styles))
        balance_score = ent / max_ent if max_ent > 0 else 0.0

        return {
            "distribution": distribution,
            "entropy": round(ent, 4),
            "balance_score": round(balance_score, 4),
            "dominant_style": (
                style_counter.most_common(1)[0][0] if style_counter else "none"
            ),
            "missing_styles": [s for s in all_styles if distribution[s] == 0],
            "total_materials": len(materials),
        }
