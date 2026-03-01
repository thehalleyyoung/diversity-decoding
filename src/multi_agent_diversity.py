"""
Multi-Agent Diversity — Diverse multi-agent systems.

Build diverse agent teams, assign specializations, run diverse voting,
measure cognitive diversity, and orchestrate debate tournaments. Agents
are LLM-agnostic: they wrap a generation callable and are differentiated
by persona, temperature, and prompt strategy.
"""

from __future__ import annotations

import logging
import math
import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text helpers
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
# Agent personas
# ---------------------------------------------------------------------------

_PERSONAS = [
    {
        "name": "Analyst",
        "style": "analytical",
        "system": "You are a precise, data-driven analyst. Focus on evidence, logic, and quantitative reasoning.",
        "temperature": 0.3,
    },
    {
        "name": "Creative",
        "style": "creative",
        "system": "You are a wildly creative thinker. Generate unconventional ideas and make unexpected connections.",
        "temperature": 0.9,
    },
    {
        "name": "Skeptic",
        "style": "critical",
        "system": "You are a critical skeptic. Question assumptions, find flaws, and stress-test every claim.",
        "temperature": 0.4,
    },
    {
        "name": "Pragmatist",
        "style": "practical",
        "system": "You are a pragmatic problem-solver. Focus on what works, implementation feasibility, and real-world constraints.",
        "temperature": 0.5,
    },
    {
        "name": "Visionary",
        "style": "strategic",
        "system": "You are a big-picture visionary. Think long-term, identify trends, and envision future possibilities.",
        "temperature": 0.7,
    },
    {
        "name": "Devil's Advocate",
        "style": "contrarian",
        "system": "You always argue the opposite position. Challenge the majority view with well-reasoned counterpoints.",
        "temperature": 0.6,
    },
    {
        "name": "Domain Expert",
        "style": "specialist",
        "system": "You are a deep domain expert. Provide detailed technical knowledge and nuanced understanding.",
        "temperature": 0.3,
    },
    {
        "name": "Generalist",
        "style": "broad",
        "system": "You are a broad generalist. Draw connections across disciplines and see the bigger picture.",
        "temperature": 0.6,
    },
    {
        "name": "Ethicist",
        "style": "ethical",
        "system": "You are an ethical thinker. Consider moral implications, fairness, and societal impact of every decision.",
        "temperature": 0.5,
    },
    {
        "name": "Innovator",
        "style": "experimental",
        "system": "You are a bold innovator. Push boundaries, propose experiments, and embrace calculated risks.",
        "temperature": 0.8,
    },
]

# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Agent:
    """A single agent in a diverse team."""
    agent_id: str
    name: str
    persona: str
    style: str
    temperature: float
    gen_fn: Optional[Callable[[str], str]] = None

    def respond(self, prompt: str) -> str:
        """Generate a response using this agent's persona."""
        full_prompt = f"[System: {self.persona}]\n\n{prompt}"
        if self.gen_fn is not None:
            return self.gen_fn(full_prompt)
        return full_prompt

    def __repr__(self) -> str:
        return f"Agent({self.name!r}, style={self.style!r})"


@dataclass
class AgentTeam:
    """A team of diverse agents."""
    agents: List[Agent]
    task: str
    diversity_score: float

    def respond_all(self, prompt: str) -> Dict[str, str]:
        """Get responses from all agents."""
        return {agent.name: agent.respond(prompt) for agent in self.agents}

    @property
    def names(self) -> List[str]:
        return [a.name for a in self.agents]

    def __len__(self) -> int:
        return len(self.agents)

    def __iter__(self) -> Iterator[Agent]:
        return iter(self.agents)


@dataclass
class ConsensusResult:
    """Result of diverse voting."""
    question: str
    votes: Dict[str, str]
    consensus: Optional[str]
    agreement_score: float
    dissenting_views: List[str]
    reasoning: Dict[str, str]

    @property
    def has_consensus(self) -> bool:
        return self.consensus is not None and self.agreement_score > 0.5


@dataclass
class DebateRound:
    """A single round of debate."""
    round_number: int
    topic: str
    arguments: Dict[str, str]
    rebuttals: Dict[str, str]
    round_diversity: float


@dataclass
class TournamentResult:
    """Result of an agent debate tournament."""
    topics: List[str]
    rounds: List[DebateRound]
    agent_scores: Dict[str, float]
    overall_diversity: float
    winner: str

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)


# ---------------------------------------------------------------------------
# Core: diverse_agent_team
# ---------------------------------------------------------------------------


def diverse_agent_team(
    task: str,
    n_agents: int = 5,
    *,
    gen_fn: Optional[Callable[[str], str]] = None,
    custom_personas: Optional[List[Dict[str, Any]]] = None,
    seed: int = 42,
) -> AgentTeam:
    """
    Assemble a team of *n_agents* with maximally diverse cognitive styles.

    Parameters
    ----------
    task : str
        The task or problem the team will work on.
    n_agents : int
        Number of agents in the team.
    gen_fn : callable(prompt) -> str, optional
        Shared LLM generation function for all agents.
    custom_personas : list of dicts, optional
        Custom persona definitions (with keys: name, style, system, temperature).
    """
    rng = random.Random(seed)

    personas = list(custom_personas) if custom_personas else list(_PERSONAS)
    rng.shuffle(personas)

    # select diverse personas using greedy max-distance
    if len(personas) > n_agents:
        selected = _select_diverse_personas(personas, n_agents, seed=seed)
    else:
        selected = personas[:n_agents]

    agents: List[Agent] = []
    for i, p in enumerate(selected):
        agents.append(Agent(
            agent_id=f"agent_{i}",
            name=p.get("name", f"Agent_{i}"),
            persona=p.get("system", ""),
            style=p.get("style", "general"),
            temperature=p.get("temperature", 0.5),
            gen_fn=gen_fn,
        ))

    # measure team diversity
    diversity = _team_style_diversity(agents)

    return AgentTeam(agents=agents, task=task, diversity_score=diversity)


def _select_diverse_personas(
    personas: List[Dict[str, Any]], k: int, seed: int = 42,
) -> List[Dict[str, Any]]:
    """Greedy selection of k most diverse personas by style embedding."""
    # encode personas as simple vectors: temperature + style hash
    def _encode(p: Dict[str, Any]) -> np.ndarray:
        temp = p.get("temperature", 0.5)
        style = p.get("style", "")
        system = p.get("system", "")
        tokens = _tokenize(system)
        # simple bag-of-words hash to vector
        vec = np.zeros(50)
        for tok in tokens:
            idx = hash(tok) % 50
            vec[idx] += 1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vec = np.append(vec, temp)
        return vec

    vecs = [_encode(p) for p in personas]
    n = len(personas)

    # greedy farthest-point selection
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(n))]
    remaining = set(range(n)) - set(selected)

    while len(selected) < k and remaining:
        best_idx = -1
        best_min_dist = -1.0
        for idx in remaining:
            min_dist = min(
                float(np.linalg.norm(vecs[idx] - vecs[s]))
                for s in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return [personas[i] for i in selected]


def _team_style_diversity(agents: List[Agent]) -> float:
    """Measure diversity of agent styles within a team."""
    if len(agents) < 2:
        return 0.0
    styles = [a.style for a in agents]
    unique_ratio = len(set(styles)) / len(styles)
    temp_std = np.std([a.temperature for a in agents])
    persona_div = _avg_pairwise_distance([a.persona for a in agents])
    return float(0.4 * unique_ratio + 0.3 * min(temp_std * 5, 1.0) + 0.3 * persona_div)


# ---------------------------------------------------------------------------
# Agent specialization
# ---------------------------------------------------------------------------


def agent_specialization(
    agents: Union[AgentTeam, List[Agent]],
    task: str,
    *,
    subtasks: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Assign each agent to the subtask that best matches their style.

    If *subtasks* is not provided, generates default subtask categories
    based on the task description.

    Returns dict mapping agent_name -> assigned_subtask.
    """
    if isinstance(agents, AgentTeam):
        agent_list = agents.agents
    else:
        agent_list = agents

    if subtasks is None:
        subtasks = [
            f"analyze {task}",
            f"brainstorm solutions for {task}",
            f"evaluate risks of {task}",
            f"plan implementation of {task}",
            f"review and critique {task}",
        ]

    # match agents to subtasks by style affinity
    style_affinity: Dict[str, str] = {
        "analytical": "analyze",
        "creative": "brainstorm",
        "critical": "evaluate risks",
        "practical": "plan implementation",
        "strategic": "brainstorm",
        "contrarian": "review and critique",
        "specialist": "analyze",
        "broad": "brainstorm",
        "ethical": "evaluate risks",
        "experimental": "brainstorm",
    }

    assignments: Dict[str, str] = {}
    used_subtasks: Set[str] = set()

    # first pass: match by affinity
    for agent in agent_list:
        best_subtask = None
        best_score = -1.0
        affinity_keyword = style_affinity.get(agent.style, "")

        for subtask in subtasks:
            if subtask in used_subtasks and len(used_subtasks) < len(subtasks):
                continue
            # score by keyword overlap
            score = _jaccard(affinity_keyword, subtask)
            if score > best_score:
                best_score = score
                best_subtask = subtask

        if best_subtask:
            assignments[agent.name] = best_subtask
            used_subtasks.add(best_subtask)
        else:
            assignments[agent.name] = subtasks[0]

    return assignments


# ---------------------------------------------------------------------------
# Diverse voting
# ---------------------------------------------------------------------------


def diverse_voting(
    agents: Union[AgentTeam, List[Agent]],
    question: str,
    *,
    options: Optional[List[str]] = None,
) -> ConsensusResult:
    """
    Have all agents vote on *question* and synthesize a consensus.

    If *options* is None, agents provide free-form answers that are
    clustered to find consensus.

    Returns a ConsensusResult with votes, consensus, agreement score,
    and dissenting views.
    """
    if isinstance(agents, AgentTeam):
        agent_list = agents.agents
    else:
        agent_list = agents

    votes: Dict[str, str] = {}
    reasoning: Dict[str, str] = {}

    for agent in agent_list:
        if options:
            prompt = (
                f"Question: {question}\n"
                f"Options: {', '.join(options)}\n"
                f"Choose one option and explain briefly."
            )
        else:
            prompt = f"Answer concisely: {question}"

        response = agent.respond(prompt)
        votes[agent.name] = response
        reasoning[agent.name] = response

    # find consensus
    if options:
        # count which option appears most in responses
        option_counts: Counter = Counter()
        for vote in votes.values():
            vote_lower = vote.lower()
            for opt in options:
                if opt.lower() in vote_lower:
                    option_counts[opt] += 1
                    break

        if option_counts:
            consensus_option, count = option_counts.most_common(1)[0]
            agreement = count / len(agent_list)
            consensus = consensus_option if agreement > 0.3 else None
        else:
            consensus = None
            agreement = 0.0
    else:
        # cluster free-form responses
        response_list = list(votes.values())
        diversity = _avg_pairwise_distance(response_list)
        agreement = max(0.0, 1.0 - diversity)
        consensus = response_list[0] if agreement > 0.5 else None

    # identify dissenters
    dissenting: List[str] = []
    if consensus:
        for name, vote in votes.items():
            if options:
                if consensus.lower() not in vote.lower():
                    dissenting.append(f"{name}: {vote[:100]}")
            else:
                sim = _jaccard(consensus, vote)
                if sim < 0.3:
                    dissenting.append(f"{name}: {vote[:100]}")

    return ConsensusResult(
        question=question,
        votes=votes,
        consensus=consensus,
        agreement_score=agreement,
        dissenting_views=dissenting,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Cognitive diversity score
# ---------------------------------------------------------------------------


def cognitive_diversity_score(agents: Union[AgentTeam, List[Agent]]) -> float:
    """
    Compute a cognitive diversity score for a group of agents.

    Measures diversity across multiple dimensions:
    - Style diversity (unique thinking styles)
    - Temperature diversity (exploration vs exploitation spread)
    - Persona diversity (textual diversity of system prompts)
    - Complementarity (how well styles cover the cognitive space)

    Returns a score in [0, 1] where 1 = maximally diverse.
    """
    if isinstance(agents, AgentTeam):
        agent_list = agents.agents
    else:
        agent_list = agents

    if len(agent_list) < 2:
        return 0.0

    n = len(agent_list)

    # style diversity: ratio of unique styles
    styles = [a.style for a in agent_list]
    style_div = len(set(styles)) / n

    # temperature diversity: normalized std
    temps = np.array([a.temperature for a in agent_list])
    temp_range = temps.max() - temps.min()
    temp_div = min(temp_range / 0.8, 1.0)  # 0.8 range = max diversity

    # persona textual diversity
    personas = [a.persona for a in agent_list]
    persona_div = _avg_pairwise_distance(personas)

    # complementarity: coverage of cognitive space
    cognitive_dimensions = [
        "analytical", "creative", "critical", "practical",
        "strategic", "ethical", "experimental", "broad",
    ]
    covered = sum(1 for dim in cognitive_dimensions if any(dim in s for s in styles))
    complementarity = covered / len(cognitive_dimensions)

    # weighted combination
    score = (
        0.25 * style_div
        + 0.20 * temp_div
        + 0.30 * persona_div
        + 0.25 * complementarity
    )

    return float(min(1.0, score))


# ---------------------------------------------------------------------------
# Agent debate tournament
# ---------------------------------------------------------------------------


def agent_debate_tournament(
    agents: Union[AgentTeam, List[Agent]],
    topics: List[str],
    rounds: int = 3,
    *,
    seed: int = 42,
) -> TournamentResult:
    """
    Run a debate tournament where agents argue diverse perspectives.

    Each round, agents present arguments and rebuttals on a topic.
    Agents are scored on argument diversity and persuasiveness.

    Parameters
    ----------
    agents : AgentTeam or list of Agent
    topics : list of debate topics
    rounds : int
        Number of debate rounds per topic.
    """
    rng = random.Random(seed)

    if isinstance(agents, AgentTeam):
        agent_list = agents.agents
    else:
        agent_list = agents

    all_rounds: List[DebateRound] = []
    agent_scores: Dict[str, float] = {a.name: 0.0 for a in agent_list}

    for topic_idx, topic in enumerate(topics):
        for round_num in range(rounds):
            # Phase 1: arguments
            arguments: Dict[str, str] = {}
            for agent in agent_list:
                if round_num == 0:
                    prompt = f"Present your argument on: {topic}"
                else:
                    prev_round = all_rounds[-1] if all_rounds else None
                    prev_args = ""
                    if prev_round:
                        prev_args = "\n".join(
                            f"- {name}: {arg[:100]}"
                            for name, arg in prev_round.arguments.items()
                        )
                    prompt = (
                        f"Continue the debate on: {topic}\n"
                        f"Previous arguments:\n{prev_args}\n"
                        f"Present a new or refined argument."
                    )
                arguments[agent.name] = agent.respond(prompt)

            # Phase 2: rebuttals
            rebuttals: Dict[str, str] = {}
            for agent in agent_list:
                other_args = {
                    name: arg for name, arg in arguments.items()
                    if name != agent.name
                }
                if other_args:
                    target_name = rng.choice(list(other_args.keys()))
                    rebuttal_prompt = (
                        f"Respond to {target_name}'s argument on '{topic}':\n"
                        f"{other_args[target_name][:200]}\n"
                        f"Present your rebuttal."
                    )
                    rebuttals[agent.name] = agent.respond(rebuttal_prompt)
                else:
                    rebuttals[agent.name] = ""

            # Score round diversity
            all_texts = list(arguments.values()) + list(rebuttals.values())
            round_diversity = _avg_pairwise_distance(
                [t for t in all_texts if t]
            )

            # score agents: unique contribution + rebuttal quality
            for agent in agent_list:
                arg = arguments.get(agent.name, "")
                reb = rebuttals.get(agent.name, "")
                # uniqueness: distance from other arguments
                other_args = [a for n, a in arguments.items() if n != agent.name]
                if other_args and arg:
                    uniqueness = _avg_pairwise_distance([arg] + other_args)
                else:
                    uniqueness = 0.5
                # rebuttal length as proxy for engagement
                engagement = min(len(_tokenize(reb)) / 20, 1.0) if reb else 0.0
                agent_scores[agent.name] += 0.6 * uniqueness + 0.4 * engagement

            all_rounds.append(DebateRound(
                round_number=topic_idx * rounds + round_num,
                topic=topic,
                arguments=arguments,
                rebuttals=rebuttals,
                round_diversity=round_diversity,
            ))

    # normalize scores
    max_score = max(agent_scores.values()) if agent_scores else 1.0
    if max_score > 0:
        agent_scores = {k: v / max_score for k, v in agent_scores.items()}

    overall_diversity = float(np.mean([r.round_diversity for r in all_rounds])) if all_rounds else 0.0
    winner = max(agent_scores, key=agent_scores.get) if agent_scores else ""  # type: ignore[arg-type]

    return TournamentResult(
        topics=topics,
        rounds=all_rounds,
        agent_scores=agent_scores,
        overall_diversity=overall_diversity,
        winner=winner,
    )
