"""
Creative AI — Creativity tools powered by diversity.

Provides brainstorming, divergent thinking, creative writing, idea evolution,
analogy generation, and contrarian viewpoint generation. All tools are
LLM-agnostic: they accept a generation callable and orchestrate diverse
creative outputs using combinatorial and evolutionary strategies.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight text helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"([" + re.escape(string.punctuation) + r"])")
_WS_RE = re.compile(r"\s+")
_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "was", "are", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "so", "if",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
}


def _tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(r" \1 ", text)
    return [t for t in _WS_RE.split(text) if t and t not in _STOPWORDS and len(t) > 1]


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _avg_pairwise_distance(texts: List[str]) -> float:
    if len(texts) < 2:
        return 0.0
    total = sum(1.0 - _jaccard(a, b) for a, b in combinations(texts, 2))
    return total / (len(texts) * (len(texts) - 1) / 2)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BrainstormResult:
    """Output of a brainstorming session."""
    topic: str
    ideas: List[str]
    categories: Dict[str, List[str]]
    diversity_score: float
    n_unique: int

    def top(self, n: int = 10) -> List[str]:
        return self.ideas[:n]

    def by_category(self, cat: str) -> List[str]:
        return self.categories.get(cat, [])

    def __len__(self) -> int:
        return len(self.ideas)

    def __iter__(self) -> Iterator[str]:
        return iter(self.ideas)


@dataclass
class Approach:
    """A single approach to solving a problem."""
    name: str
    description: str
    category: str
    novelty_score: float
    feasibility_score: float

    def __repr__(self) -> str:
        return f"Approach({self.name!r}, novelty={self.novelty_score:.2f})"


@dataclass
class Analogy:
    """A generated analogy mapping a concept to another domain."""
    source_concept: str
    target_domain: str
    mapping: str
    explanation: str
    strength: float

    def __repr__(self) -> str:
        return f"Analogy({self.source_concept!r} → {self.target_domain!r})"


# ---------------------------------------------------------------------------
# Perspective & style templates (used for combinatorial generation)
# ---------------------------------------------------------------------------

_THINKING_HATS = [
    ("analytical", "Break this down logically and examine the data."),
    ("emotional", "Consider feelings, intuitions, and human impact."),
    ("cautious", "Identify risks, problems, and what could go wrong."),
    ("optimistic", "Find benefits, opportunities, and best-case outcomes."),
    ("creative", "Generate wild, unconventional, lateral ideas."),
    ("process", "Think about how to organize, sequence, and manage this."),
]

_CREATIVITY_TECHNIQUES = [
    "reverse_brainstorm",   # what would make this worse?
    "scamper",              # substitute, combine, adapt, modify, put-to-other-use, eliminate, reverse
    "random_input",         # inject random stimulus
    "analogy_transfer",     # borrow from another domain
    "constraint_removal",   # what if X constraint didn't exist?
    "worst_idea",           # deliberately bad ideas, then invert
    "mind_map",             # radial association
    "six_hats",             # De Bono's six thinking hats
]

_WRITING_STYLES = [
    "formal_academic",
    "conversational",
    "poetic",
    "journalistic",
    "humorous",
    "noir",
    "minimalist",
    "epic",
    "technical",
    "stream_of_consciousness",
]

_ANALOGY_DOMAINS = [
    "biology", "physics", "cooking", "sports", "music",
    "architecture", "warfare", "gardening", "theater", "economics",
    "mythology", "computing", "medicine", "navigation", "geology",
]

# ---------------------------------------------------------------------------
# Core: brainstorm
# ---------------------------------------------------------------------------


def brainstorm(
    topic: str,
    n_ideas: int = 50,
    creativity: float = 0.8,
    *,
    gen_fn: Optional[Callable[[str], str]] = None,
    seed: int = 42,
) -> BrainstormResult:
    """
    Generate *n_ideas* diverse ideas on *topic*.

    If *gen_fn* is provided, it is called with prompts to produce ideas via
    an LLM. Otherwise, a combinatorial template strategy is used to produce
    idea *prompts* (useful as prompts for downstream LLM calls).

    Parameters
    ----------
    topic : str
        The brainstorming topic or problem statement.
    n_ideas : int
        Target number of ideas.
    creativity : float in [0, 1]
        Higher values use more unconventional techniques.
    gen_fn : callable(prompt) -> str, optional
        LLM generation function.
    seed : int
        Random seed.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Build diverse prompts using different techniques
    prompts: List[str] = []
    categories: Dict[str, List[int]] = defaultdict(list)

    # Phase 1: direct brainstorm
    base_prompt = f"Generate a unique idea about: {topic}"
    n_direct = max(1, int(n_ideas * 0.3))
    for i in range(n_direct):
        prompts.append(f"{base_prompt} (idea #{i + 1}, be original)")
        categories["direct"].append(len(prompts) - 1)

    # Phase 2: thinking hats
    n_hats = max(1, int(n_ideas * 0.2))
    for i in range(n_hats):
        hat_name, hat_desc = _THINKING_HATS[i % len(_THINKING_HATS)]
        p = f"Think about '{topic}' from a {hat_name} perspective. {hat_desc}"
        prompts.append(p)
        categories[f"hat_{hat_name}"].append(len(prompts) - 1)

    # Phase 3: creativity techniques
    n_tech = max(1, int(n_ideas * 0.25))
    techniques = _CREATIVITY_TECHNIQUES.copy()
    rng.shuffle(techniques)
    for i in range(n_tech):
        tech = techniques[i % len(techniques)]
        if tech == "reverse_brainstorm":
            p = f"What would make '{topic}' fail completely? Then invert."
        elif tech == "scamper":
            ops = ["substitute", "combine", "adapt", "modify", "eliminate", "reverse"]
            op = rng.choice(ops)
            p = f"Apply SCAMPER ({op}) to '{topic}': how would you {op} it?"
        elif tech == "random_input":
            stimuli = ["ocean", "clock", "forest", "mirror", "bridge", "storm"]
            stim = rng.choice(stimuli)
            p = f"Connect '{topic}' with '{stim}' — what new idea emerges?"
        elif tech == "analogy_transfer":
            domain = rng.choice(_ANALOGY_DOMAINS)
            p = f"How would '{topic}' work if it were like something from {domain}?"
        elif tech == "constraint_removal":
            p = f"If there were no constraints at all, what would '{topic}' look like?"
        elif tech == "worst_idea":
            p = f"What's the worst possible approach to '{topic}'? Now flip it."
        elif tech == "mind_map":
            p = f"Free-associate from '{topic}': what adjacent concepts connect?"
        else:
            p = f"Apply {tech} thinking to '{topic}'."
        prompts.append(p)
        categories[f"technique_{tech}"].append(len(prompts) - 1)

    # Phase 4: cross-pollination
    n_cross = n_ideas - len(prompts)
    for i in range(max(0, n_cross)):
        hat_name, _ = rng.choice(_THINKING_HATS)
        tech = rng.choice(_CREATIVITY_TECHNIQUES)
        p = f"Combine {hat_name} thinking with {tech} on '{topic}'."
        prompts.append(p)
        categories["cross_pollination"].append(len(prompts) - 1)

    prompts = prompts[:n_ideas]

    # Generate ideas
    if gen_fn is not None:
        ideas = [gen_fn(p) for p in prompts]
    else:
        ideas = prompts  # prompts themselves serve as idea seeds

    # Deduplicate
    seen: Set[str] = set()
    unique_ideas: List[str] = []
    for idea in ideas:
        key = idea.strip().lower()
        if key not in seen:
            seen.add(key)
            unique_ideas.append(idea.strip())

    # Build category mapping for unique ideas
    cat_map: Dict[str, List[str]] = defaultdict(list)
    for cat, indices in categories.items():
        for idx in indices:
            if idx < len(ideas) and ideas[idx].strip() in seen:
                cat_map[cat].append(ideas[idx].strip())

    diversity = _avg_pairwise_distance(unique_ideas)

    return BrainstormResult(
        topic=topic,
        ideas=unique_ideas,
        categories=dict(cat_map),
        diversity_score=diversity,
        n_unique=len(unique_ideas),
    )


# ---------------------------------------------------------------------------
# Divergent thinking
# ---------------------------------------------------------------------------


def divergent_thinking(
    problem: str,
    n_approaches: int = 10,
    *,
    gen_fn: Optional[Callable[[str], str]] = None,
    seed: int = 42,
) -> List[Approach]:
    """
    Generate *n_approaches* diverse approaches to *problem*.

    Each approach is generated from a different perspective or methodology
    to maximize cognitive diversity.
    """
    rng = random.Random(seed)

    methodology_families = [
        ("analytical", "systematic", 0.7),
        ("empirical", "experimental", 0.6),
        ("creative", "lateral_thinking", 0.9),
        ("collaborative", "crowdsourced", 0.5),
        ("computational", "algorithmic", 0.6),
        ("minimalist", "reductive", 0.8),
        ("analogical", "cross_domain", 0.85),
        ("adversarial", "red_team", 0.75),
        ("evolutionary", "iterative", 0.7),
        ("philosophical", "first_principles", 0.8),
        ("pragmatic", "mvp", 0.4),
        ("artistic", "intuitive", 0.95),
    ]

    rng.shuffle(methodology_families)
    approaches: List[Approach] = []

    for i in range(n_approaches):
        name, category, novelty_base = methodology_families[i % len(methodology_families)]

        prompt = (
            f"Approach '{problem}' using a {name} methodology. "
            f"Describe a concrete {category} approach."
        )

        if gen_fn is not None:
            description = gen_fn(prompt)
        else:
            description = prompt

        # vary novelty/feasibility based on methodology
        novelty = min(1.0, novelty_base + rng.uniform(-0.1, 0.1))
        feasibility = max(0.0, min(1.0, 1.0 - novelty * 0.5 + rng.uniform(-0.1, 0.1)))

        approaches.append(Approach(
            name=f"{name}_approach_{i}",
            description=description,
            category=category,
            novelty_score=novelty,
            feasibility_score=feasibility,
        ))

    return approaches


# ---------------------------------------------------------------------------
# Creative writing
# ---------------------------------------------------------------------------


def creative_writing(
    prompt: str,
    styles: int = 5,
    blend: bool = True,
    *,
    gen_fn: Optional[Callable[[str], str]] = None,
    seed: int = 42,
) -> List[str]:
    """
    Generate multiple creative writing outputs for *prompt* in diverse styles.

    Parameters
    ----------
    prompt : str
        The writing prompt.
    styles : int
        Number of distinct style variations.
    blend : bool
        If True, also include blended styles (pairs combined).
    gen_fn : callable(prompt) -> str, optional
        LLM generation function.
    """
    rng = random.Random(seed)
    selected_styles = rng.sample(_WRITING_STYLES, min(styles, len(_WRITING_STYLES)))

    results: List[str] = []

    # Generate one piece per style
    for style in selected_styles:
        style_prompt = (
            f"Write in a {style.replace('_', ' ')} style about: {prompt}"
        )
        if gen_fn is not None:
            results.append(gen_fn(style_prompt))
        else:
            results.append(style_prompt)

    # Generate blended styles
    if blend and len(selected_styles) >= 2:
        pairs = list(combinations(selected_styles, 2))
        rng.shuffle(pairs)
        n_blends = max(1, styles // 3)
        for s1, s2 in pairs[:n_blends]:
            blend_prompt = (
                f"Write blending {s1.replace('_', ' ')} and "
                f"{s2.replace('_', ' ')} styles about: {prompt}"
            )
            if gen_fn is not None:
                results.append(gen_fn(blend_prompt))
            else:
                results.append(blend_prompt)

    return results


# ---------------------------------------------------------------------------
# Idea evolution (genetic algorithm over ideas)
# ---------------------------------------------------------------------------


def idea_evolution(
    seed_ideas: List[str],
    generations: int = 5,
    population: int = 20,
    *,
    gen_fn: Optional[Callable[[str], str]] = None,
    mutation_rate: float = 0.3,
    crossover_rate: float = 0.5,
    fitness_fn: Optional[Callable[[str], float]] = None,
    seed: int = 42,
) -> List[str]:
    """
    Evolve ideas through a genetic-algorithm-inspired process.

    Each generation: select fit ideas, cross them over, mutate, and
    evaluate. Returns the final population sorted by fitness.

    Parameters
    ----------
    seed_ideas : list of str
        Initial idea population.
    generations : int
        Number of evolutionary generations.
    population : int
        Target population size each generation.
    gen_fn : callable(prompt) -> str, optional
        LLM function used for mutation and crossover.
    mutation_rate : float
        Probability of mutating an idea.
    crossover_rate : float
        Probability of crossing over two ideas.
    fitness_fn : callable(idea) -> float, optional
        Fitness function. Defaults to idea length diversity heuristic.
    """
    rng = random.Random(seed)

    if fitness_fn is None:
        def fitness_fn(idea: str) -> float:
            tokens = _tokenize(idea)
            return len(set(tokens)) / max(len(tokens), 1)

    # Initialize population
    pop: List[str] = list(seed_ideas)
    while len(pop) < population:
        pop.append(rng.choice(seed_ideas))

    for gen in range(generations):
        # Evaluate fitness
        scores = [(idea, fitness_fn(idea)) for idea in pop]
        scores.sort(key=lambda x: -x[1])

        # Selection: top half survives
        survivors = [idea for idea, _ in scores[: population // 2]]

        offspring: List[str] = list(survivors)

        # Crossover
        while len(offspring) < population:
            if rng.random() < crossover_rate and len(survivors) >= 2:
                p1, p2 = rng.sample(survivors, 2)
                # combine by taking first half of p1 and second half of p2
                tokens1 = _tokenize(p1)
                tokens2 = _tokenize(p2)
                mid1 = len(tokens1) // 2
                mid2 = len(tokens2) // 2
                child_tokens = tokens1[:mid1] + tokens2[mid2:]
                child = " ".join(child_tokens) if child_tokens else p1

                if gen_fn is not None:
                    prompt = (
                        f"Combine these two ideas into a new one:\n"
                        f"1. {p1}\n2. {p2}\nNew idea:"
                    )
                    child = gen_fn(prompt)

                offspring.append(child)
            else:
                offspring.append(rng.choice(survivors))

        # Mutation
        mutated: List[str] = []
        for idea in offspring:
            if rng.random() < mutation_rate:
                if gen_fn is not None:
                    prompt = f"Take this idea and modify it significantly: {idea}"
                    idea = gen_fn(prompt)
                else:
                    # simple mutation: shuffle words
                    tokens = _tokenize(idea)
                    if len(tokens) > 2:
                        i, j = rng.sample(range(len(tokens)), 2)
                        tokens[i], tokens[j] = tokens[j], tokens[i]
                    idea = " ".join(tokens)
            mutated.append(idea)

        pop = mutated[:population]

        logger.debug(
            "Generation %d: best fitness=%.3f, diversity=%.3f",
            gen,
            max(fitness_fn(p) for p in pop),
            _avg_pairwise_distance(pop),
        )

    # Final sort by fitness
    final = sorted(pop, key=lambda x: -fitness_fn(x))
    return final


# ---------------------------------------------------------------------------
# Analogy generator
# ---------------------------------------------------------------------------


def analogy_generator(
    concept: str,
    domains: Optional[List[str]] = None,
    *,
    gen_fn: Optional[Callable[[str], str]] = None,
    seed: int = 42,
) -> List[Analogy]:
    """
    Generate analogies mapping *concept* to diverse domains.

    Parameters
    ----------
    concept : str
        The source concept to generate analogies for.
    domains : list of str, optional
        Target domains. Defaults to a built-in diverse set.
    gen_fn : callable(prompt) -> str, optional
        LLM generation function.
    """
    rng = random.Random(seed)

    if domains is None:
        domains = list(_ANALOGY_DOMAINS)
    rng.shuffle(domains)

    analogies: List[Analogy] = []

    for domain in domains:
        prompt = (
            f"Create an analogy: '{concept}' is like something in {domain}. "
            f"Explain the mapping and why it works."
        )

        if gen_fn is not None:
            explanation = gen_fn(prompt)
            mapping = f"{concept} → {domain}"
        else:
            explanation = prompt
            mapping = f"{concept} → {domain}"

        # strength heuristic: how distant is the domain from the concept?
        concept_tokens = set(_tokenize(concept))
        domain_tokens = set(_tokenize(domain))
        overlap = len(concept_tokens & domain_tokens)
        distance = 1.0 - overlap / max(len(concept_tokens | domain_tokens), 1)
        strength = 0.3 + 0.7 * distance  # distant domains make stronger analogies

        analogies.append(Analogy(
            source_concept=concept,
            target_domain=domain,
            mapping=mapping,
            explanation=explanation,
            strength=strength,
        ))

    # Sort by strength (strongest analogies first)
    analogies.sort(key=lambda a: -a.strength)
    return analogies


# ---------------------------------------------------------------------------
# Contrarian viewpoints
# ---------------------------------------------------------------------------


def contrarian_viewpoints(
    statement: str,
    n: int = 5,
    *,
    gen_fn: Optional[Callable[[str], str]] = None,
    seed: int = 42,
) -> List[str]:
    """
    Generate *n* contrarian viewpoints challenging *statement*.

    Uses diverse argumentation strategies to ensure viewpoints are
    meaningfully different from each other, not just rephrased disagreements.
    """
    rng = random.Random(seed)

    strategies = [
        ("empirical", f"Challenge '{statement}' with data or evidence-based counter-examples."),
        ("logical", f"Find logical flaws or hidden assumptions in '{statement}'."),
        ("historical", f"Use historical precedents that contradict '{statement}'."),
        ("cultural", f"Show how '{statement}' breaks down in different cultural contexts."),
        ("economic", f"Present economic arguments against '{statement}'."),
        ("ethical", f"Raise ethical concerns about '{statement}'."),
        ("technological", f"Argue how technology undermines '{statement}'."),
        ("philosophical", f"Use philosophical frameworks to challenge '{statement}'."),
        ("psychological", f"Explain cognitive biases that lead people to believe '{statement}'."),
        ("systemic", f"Show how systemic factors complicate '{statement}'."),
        ("devil_advocate", f"Steel-man the opposition to '{statement}'."),
        ("reductio", f"Take '{statement}' to its logical extreme to show problems."),
    ]

    rng.shuffle(strategies)
    viewpoints: List[str] = []

    for i in range(min(n, len(strategies))):
        strategy_name, prompt = strategies[i]

        if gen_fn is not None:
            viewpoint = gen_fn(prompt)
        else:
            viewpoint = f"[{strategy_name}] {prompt}"

        viewpoints.append(viewpoint)

    return viewpoints


# ---------------------------------------------------------------------------
# Creativity metrics
# ---------------------------------------------------------------------------


def creativity_score(ideas: List[str]) -> Dict[str, float]:
    """
    Compute creativity metrics for a set of ideas.

    Returns
    -------
    dict with keys:
        fluency : number of ideas
        flexibility : number of distinct categories (heuristic)
        originality : average pairwise distance (higher = more original)
        elaboration : average idea length in tokens
    """
    if not ideas:
        return {"fluency": 0, "flexibility": 0, "originality": 0, "elaboration": 0}

    # fluency
    fluency = float(len(ideas))

    # flexibility: cluster ideas by leading bigram
    categories: Set[Tuple[str, ...]] = set()
    for idea in ideas:
        tokens = _tokenize(idea)
        if len(tokens) >= 2:
            categories.add(tuple(tokens[:2]))
        elif tokens:
            categories.add(tuple(tokens[:1]))
    flexibility = float(len(categories))

    # originality: pairwise distance
    originality = _avg_pairwise_distance(ideas)

    # elaboration: average token count
    elaboration = float(np.mean([len(_tokenize(idea)) for idea in ideas]))

    return {
        "fluency": fluency,
        "flexibility": flexibility,
        "originality": originality,
        "elaboration": elaboration,
    }


# ---------------------------------------------------------------------------
# Perspective taking
# ---------------------------------------------------------------------------


def perspective_shift(
    topic: str,
    perspectives: Optional[List[str]] = None,
    *,
    gen_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, str]:
    """
    Generate viewpoints on *topic* from multiple diverse perspectives.

    Returns a dict mapping perspective name to the generated viewpoint.
    """
    if perspectives is None:
        perspectives = [
            "scientist", "artist", "child", "elder", "skeptic",
            "optimist", "historian", "futurist", "engineer", "philosopher",
        ]

    results: Dict[str, str] = {}
    for p in perspectives:
        prompt = f"From the perspective of a {p}, what do you think about: {topic}"
        if gen_fn is not None:
            results[p] = gen_fn(prompt)
        else:
            results[p] = prompt

    return results
