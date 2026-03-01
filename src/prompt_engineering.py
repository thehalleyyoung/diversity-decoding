"""
Diversity-driven prompt engineering utilities.

Generate diverse prompt rephrasings, chain-of-thought reasoning paths,
expert personas, and ensemble diverse prompt outputs — all using only
numpy/scipy and standard library (no external NLP packages).
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import math
import random
import re
import string
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"([" + re.escape(string.punctuation) + r"])")
_WS_RE = re.compile(r"\s+")


def _tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(r" \\1 ", text)
    return [t for t in _WS_RE.split(text) if t]


def _jaccard_tokens(a: str, b: str) -> float:
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 1.0


def _diversity_score(texts: List[str]) -> float:
    """Average pairwise Jaccard distance."""
    if len(texts) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            total += 1.0 - _jaccard_tokens(texts[i], texts[j])
            count += 1
    return total / count


# ---------------------------------------------------------------------------
# Prompt transformation templates
# ---------------------------------------------------------------------------

_REPHRASE_TEMPLATES = [
    "Rephrase: {prompt}",
    "In other words: {prompt}",
    "Can you explain this differently? {prompt}",
    "Restate the following: {prompt}",
    "Put this another way: {prompt}",
    "Simplify: {prompt}",
    "Elaborate on: {prompt}",
    "From a different angle: {prompt}",
    "Summarize concisely: {prompt}",
    "As a question: {prompt}",
    "As an instruction: {prompt}",
    "Using formal language: {prompt}",
    "Using casual language: {prompt}",
    "As a step-by-step request: {prompt}",
    "With more context: {prompt}",
]

_COT_PREFIXES = [
    "Let's think step by step.",
    "Let's break this down systematically.",
    "Let me reason through this carefully.",
    "Working through this logically:",
    "Let's analyze this from first principles.",
    "Consider the problem from multiple angles:",
    "Let's approach this methodically:",
    "Thinking about this differently:",
    "Here's one way to reason about it:",
    "Let me walk through the reasoning:",
]

_PERSONA_TEMPLATES = [
    "You are an expert {domain} researcher.",
    "You are a practical {domain} engineer with 20 years of experience.",
    "You are a {domain} teacher explaining to a bright student.",
    "You are a skeptical {domain} critic looking for weaknesses.",
    "You are a creative {domain} innovator exploring unconventional ideas.",
    "You are a {domain} historian tracing the evolution of ideas.",
    "You are a cross-disciplinary thinker connecting {domain} to other fields.",
    "You are a {domain} ethicist considering societal implications.",
    "You are a {domain} entrepreneur focused on real-world applications.",
    "You are a {domain} philosopher exploring foundational questions.",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CoTPath:
    """A single chain-of-thought reasoning path."""
    prefix: str
    prompt: str
    full_prompt: str
    path_id: int
    diversity_from_others: float = 0.0


@dataclass
class PersonaSpec:
    """An expert persona specification."""
    description: str
    domain: str
    style: str
    prompt_prefix: str


@dataclass
class EnsembleResult:
    """Result from ensembling diverse prompt outputs."""
    final_answer: str
    individual_outputs: List[str]
    prompts_used: List[str]
    diversity_score: float
    agreement_score: float


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def diversify_prompt(
    prompt: str,
    n_variants: int = 10,
    seed: Optional[int] = None,
) -> List[str]:
    """Generate *n_variants* diverse rephrasings of *prompt*.

    Uses template-based transformations and selects the most diverse subset
    via greedy farthest-point selection.
    """
    rng = random.Random(seed)

    # Generate candidate rephrasings
    candidates: List[str] = [prompt]  # original always included
    templates = list(_REPHRASE_TEMPLATES)
    rng.shuffle(templates)

    for tmpl in templates:
        candidates.append(tmpl.format(prompt=prompt))

    # Word-level perturbations
    words = prompt.split()
    if len(words) > 3:
        for _ in range(min(10, n_variants)):
            w = list(words)
            # Swap two random adjacent words
            idx = rng.randint(0, len(w) - 2)
            w[idx], w[idx + 1] = w[idx + 1], w[idx]
            candidates.append(" ".join(w))

        for _ in range(min(5, n_variants)):
            # Drop a random non-essential word
            w = list(words)
            idx = rng.randint(0, len(w) - 1)
            dropped = w.pop(idx)
            if len(w) >= 2:
                candidates.append(" ".join(w))

    # Add context variations
    prefixes = [
        "Please ", "Could you ", "I need you to ", "Help me ",
        "I'd like you to ", "",
    ]
    for pfx in prefixes:
        candidates.append(f"{pfx}{prompt}")

    suffixes = [
        " Be thorough.", " Be concise.", " Think carefully.",
        " Consider edge cases.", " Explain your reasoning.", "",
    ]
    for sfx in suffixes:
        candidates.append(f"{prompt}{sfx}")

    # Deduplicate
    seen: Set[str] = set()
    unique: List[str] = []
    for c in candidates:
        normalized = " ".join(c.lower().split())
        if normalized not in seen:
            seen.add(normalized)
            unique.append(c)

    # Greedy farthest-point selection for diversity
    if len(unique) <= n_variants:
        return unique

    selected = [unique[0]]
    remaining = set(range(1, len(unique)))

    while len(selected) < n_variants and remaining:
        best_idx = -1
        best_dist = -1.0
        for idx in remaining:
            min_d = min(
                1.0 - _jaccard_tokens(unique[idx], s) for s in selected
            )
            if min_d > best_dist:
                best_dist = min_d
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(unique[best_idx])
        remaining.discard(best_idx)

    return selected


def diverse_chain_of_thought(
    prompt: str,
    n_paths: int = 5,
    seed: Optional[int] = None,
) -> List[CoTPath]:
    """Generate *n_paths* diverse chain-of-thought prompts.

    Each path uses a different reasoning prefix to encourage the LLM to
    explore different reasoning strategies.
    """
    rng = random.Random(seed)
    prefixes = list(_COT_PREFIXES)
    rng.shuffle(prefixes)
    prefixes = prefixes[:n_paths]

    paths: List[CoTPath] = []
    for i, prefix in enumerate(prefixes):
        full = f"{prefix}\n\n{prompt}"
        paths.append(CoTPath(
            prefix=prefix,
            prompt=prompt,
            full_prompt=full,
            path_id=i,
        ))

    # Compute diversity from others
    for i, p in enumerate(paths):
        if len(paths) < 2:
            p.diversity_from_others = 0.0
            continue
        dists = [
            1.0 - _jaccard_tokens(p.full_prompt, paths[j].full_prompt)
            for j in range(len(paths)) if j != i
        ]
        p.diversity_from_others = float(np.mean(dists))

    return paths


def diverse_personas(
    topic: str,
    n: int = 5,
    domains: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> List[PersonaSpec]:
    """Generate *n* diverse expert personas for a given topic.

    Parameters
    ----------
    topic : str
        The subject the personas should relate to.
    domains : list of str, optional
        Custom domain labels. If not provided, domains are inferred from
        the topic via simple keyword expansion.
    """
    rng = random.Random(seed)

    if domains is None:
        # Generate domain variants from the topic
        base_words = topic.lower().split()
        domain_pool = [
            topic,
            f"{topic} theory",
            f"applied {topic}",
            f"computational {topic}",
            f"experimental {topic}",
            f"{topic} systems",
            f"{topic} and society",
            f"mathematical {topic}",
            f"{topic} design",
            f"{topic} analysis",
        ]
        if len(base_words) > 1:
            domain_pool.extend(base_words)
        domains = domain_pool

    rng.shuffle(domains)
    domains = domains[:n]

    styles = [
        "analytical", "creative", "critical", "practical",
        "theoretical", "empirical", "holistic", "reductionist",
        "visionary", "methodical",
    ]
    rng.shuffle(styles)

    templates = list(_PERSONA_TEMPLATES)
    rng.shuffle(templates)

    personas: List[PersonaSpec] = []
    for i in range(min(n, len(domains))):
        domain = domains[i]
        style = styles[i % len(styles)]
        tmpl = templates[i % len(templates)]
        desc = tmpl.format(domain=domain)
        personas.append(PersonaSpec(
            description=desc,
            domain=domain,
            style=style,
            prompt_prefix=f"{desc} ",
        ))

    return personas


def ensemble_prompts(
    prompt: str,
    llm_fn: Callable[[str], str],
    n: int = 10,
    k: int = 3,
    seed: Optional[int] = None,
) -> EnsembleResult:
    """Ensemble *n* diverse prompt variants and select best *k* outputs.

    Parameters
    ----------
    prompt : str
        The base prompt.
    llm_fn : callable
        A function ``str -> str`` that calls an LLM and returns its output.
    n : int
        Number of diverse prompt variants to generate.
    k : int
        Number of outputs to keep in the final ensemble.
    seed : int or None
        Random seed.

    Returns
    -------
    EnsembleResult
        Contains the final combined answer, individual outputs, and metrics.
    """
    variants = diversify_prompt(prompt, n_variants=n, seed=seed)

    outputs: List[str] = []
    for v in variants:
        try:
            out = llm_fn(v)
            if out and out.strip():
                outputs.append(out.strip())
        except Exception as exc:
            logger.warning("LLM call failed for variant: %s", exc)

    if not outputs:
        return EnsembleResult(
            final_answer="",
            individual_outputs=[],
            prompts_used=variants,
            diversity_score=0.0,
            agreement_score=0.0,
        )

    # Select diverse subset of size k
    if len(outputs) <= k:
        selected = outputs
    else:
        selected = [outputs[0]]
        remaining = set(range(1, len(outputs)))
        while len(selected) < k and remaining:
            best_idx = -1
            best_dist = -1.0
            for idx in remaining:
                min_d = min(
                    1.0 - _jaccard_tokens(outputs[idx], s) for s in selected
                )
                if min_d > best_dist:
                    best_dist = min_d
                    best_idx = idx
            if best_idx < 0:
                break
            selected.append(outputs[best_idx])
            remaining.discard(best_idx)

    # Combine by majority-vote on token level
    all_tokens = [Counter(_tokenize(s)) for s in selected]
    if all_tokens:
        merged = all_tokens[0]
        for tc in all_tokens[1:]:
            merged += tc
        # Build answer from most common tokens (ordered by frequency)
        common = merged.most_common(200)
        final_answer = " ".join(tok for tok, _ in common)
    else:
        final_answer = selected[0] if selected else ""

    # Agreement: average pairwise Jaccard similarity
    if len(selected) >= 2:
        sims = [
            _jaccard_tokens(selected[i], selected[j])
            for i in range(len(selected))
            for j in range(i + 1, len(selected))
        ]
        agreement = float(np.mean(sims))
    else:
        agreement = 1.0

    return EnsembleResult(
        final_answer=final_answer,
        individual_outputs=outputs,
        prompts_used=variants[:len(outputs)],
        diversity_score=_diversity_score(selected),
        agreement_score=agreement,
    )
