"""
Human evaluation framework for diversity assessment.

Create evaluation tasks, run pairwise comparisons, collect diversity
annotations, compute inter-annotator agreement, and correlate human
scores with automatic metrics. All implementations are self-contained
with numpy/scipy.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import random
import time
from collections import Counter, defaultdict
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
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvalCriterion:
    """A single evaluation criterion."""
    name: str
    description: str
    scale_min: float = 1.0
    scale_max: float = 5.0
    weight: float = 1.0


@dataclass
class EvalItem:
    """A single item to evaluate (a response or set of responses)."""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalTask:
    """A complete evaluation task ready for annotators."""
    task_id: str
    items: List[EvalItem]
    criteria: List[EvalCriterion]
    instructions: str
    created_at: float = field(default_factory=time.time)
    pairwise_pairs: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def n_criteria(self) -> int:
        return len(self.criteria)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instructions": self.instructions,
            "criteria": [
                {"name": c.name, "description": c.description,
                 "scale": [c.scale_min, c.scale_max], "weight": c.weight}
                for c in self.criteria
            ],
            "items": [
                {"id": it.id, "text": it.text, "metadata": it.metadata}
                for it in self.items
            ],
            "pairs": self.pairwise_pairs,
        }


@dataclass
class Annotation:
    """A single annotator's scores for one item."""
    annotator_id: str
    item_id: str
    scores: Dict[str, float]  # criterion_name -> score
    timestamp: float = field(default_factory=time.time)
    notes: str = ""


@dataclass
class PairwiseJudgment:
    """Annotator judgment for a pair of items."""
    annotator_id: str
    item_a_id: str
    item_b_id: str
    winner: str  # item_a_id, item_b_id, or "tie"
    confidence: float = 1.0  # 0-1
    criterion: str = "overall"


@dataclass
class RankingResult:
    """Result of pairwise comparison ranking."""
    ranked_ids: List[str]
    scores: Dict[str, float]  # id -> Bradley-Terry score
    win_matrix: np.ndarray
    n_comparisons: int
    consistency: float  # fraction of transitive triples


@dataclass
class AnnotationResult:
    """Aggregated annotation results."""
    item_scores: Dict[str, Dict[str, float]]  # item_id -> criterion -> avg score
    item_rankings: Dict[str, int]  # item_id -> rank (1-based)
    n_annotations: int
    annotations: List[Annotation]


@dataclass
class CorrelationResult:
    """Correlation between human and automatic scores."""
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float
    n_items: int
    human_scores: List[float]
    auto_scores: List[float]


# ---------------------------------------------------------------------------
# Task creation
# ---------------------------------------------------------------------------


def create_eval_task(
    responses: List[str],
    criteria: Optional[List[Dict[str, Any]]] = None,
    instructions: str = "",
    include_pairwise: bool = True,
    max_pairs: int = 50,
    seed: Optional[int] = None,
) -> EvalTask:
    """Create an evaluation task from a list of responses.

    Parameters
    ----------
    responses : list of str
        Texts to evaluate.
    criteria : list of dict, optional
        Each dict should have ``name``, ``description``, and optionally
        ``scale_min``, ``scale_max``, ``weight``.
    instructions : str
        Instructions for annotators.
    include_pairwise : bool
        If True, generate pairwise comparison pairs.
    max_pairs : int
        Maximum number of pairwise pairs to generate.
    """
    if criteria is None:
        criteria = [
            {"name": "quality", "description": "Overall response quality"},
            {"name": "diversity", "description": "How different this response is from others"},
            {"name": "relevance", "description": "How relevant the response is to the prompt"},
        ]

    eval_criteria = [
        EvalCriterion(
            name=c["name"],
            description=c.get("description", ""),
            scale_min=c.get("scale_min", 1.0),
            scale_max=c.get("scale_max", 5.0),
            weight=c.get("weight", 1.0),
        )
        for c in criteria
    ]

    items = []
    for i, text in enumerate(responses):
        item_id = hashlib.md5(f"{i}:{text[:100]}".encode()).hexdigest()[:8]
        items.append(EvalItem(id=item_id, text=text, metadata={"index": i}))

    pairs: List[Tuple[int, int]] = []
    if include_pairwise and len(items) >= 2:
        all_pairs = list(itertools.combinations(range(len(items)), 2))
        rng = random.Random(seed)
        if len(all_pairs) > max_pairs:
            rng.shuffle(all_pairs)
            pairs = all_pairs[:max_pairs]
        else:
            pairs = all_pairs

    task_id = hashlib.md5(
        f"{len(responses)}:{time.time()}".encode()
    ).hexdigest()[:12]

    if not instructions:
        instructions = (
            "Rate each response on the given criteria. "
            "Use the full scale range. For pairwise comparisons, "
            "choose which response is better or mark as tie."
        )

    return EvalTask(
        task_id=task_id,
        items=items,
        criteria=eval_criteria,
        instructions=instructions,
        pairwise_pairs=pairs,
    )


# ---------------------------------------------------------------------------
# Pairwise comparison & Bradley-Terry ranking
# ---------------------------------------------------------------------------


def pairwise_comparison(
    responses: List[str],
    judgments: Optional[List[PairwiseJudgment]] = None,
    judge_fn: Optional[Callable[[str, str], str]] = None,
    n_iterations: int = 100,
) -> RankingResult:
    """Rank responses via pairwise comparisons using Bradley-Terry model.

    Provide either pre-collected *judgments* or a *judge_fn* that takes
    two texts and returns the id of the winner (or ``"tie"``).

    Parameters
    ----------
    responses : list of str
        Texts to rank.
    judgments : list of PairwiseJudgment, optional
        Pre-collected judgments.
    judge_fn : callable, optional
        ``(text_a, text_b) -> "a" | "b" | "tie"`` function for automatic
        pairwise judging.
    n_iterations : int
        Bradley-Terry fitting iterations.
    """
    n = len(responses)
    ids = [f"r{i}" for i in range(n)]
    id2idx = {rid: i for i, rid in enumerate(ids)}

    wins = np.zeros((n, n))

    if judgments:
        for j in judgments:
            a_idx = id2idx.get(j.item_a_id)
            b_idx = id2idx.get(j.item_b_id)
            if a_idx is None or b_idx is None:
                continue
            if j.winner == j.item_a_id:
                wins[a_idx, b_idx] += j.confidence
            elif j.winner == j.item_b_id:
                wins[b_idx, a_idx] += j.confidence
            else:  # tie
                wins[a_idx, b_idx] += 0.5 * j.confidence
                wins[b_idx, a_idx] += 0.5 * j.confidence
    elif judge_fn is not None:
        for i in range(n):
            for j_idx in range(i + 1, n):
                result = judge_fn(responses[i], responses[j_idx])
                if result == "a":
                    wins[i, j_idx] += 1
                elif result == "b":
                    wins[j_idx, i] += 1
                else:
                    wins[i, j_idx] += 0.5
                    wins[j_idx, i] += 0.5

    # Bradley-Terry MLE via iterative algorithm
    scores = np.ones(n)
    for _ in range(n_iterations):
        new_scores = np.zeros(n)
        for i in range(n):
            w_i = wins[i].sum()
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                n_ij = wins[i, j] + wins[j, i]
                if n_ij > 0:
                    denom += n_ij / (scores[i] + scores[j])
            new_scores[i] = w_i / denom if denom > 0 else scores[i]
        # Normalize
        s = new_scores.sum()
        if s > 0:
            new_scores /= s
        scores = new_scores

    # Compute consistency (transitivity)
    n_transitive = 0
    n_triples = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                n_triples += 1
                s = [scores[i], scores[j], scores[k]]
                idx = [i, j, k]
                ranked = sorted(zip(s, idx), reverse=True)
                a, b, c = ranked[0][1], ranked[1][1], ranked[2][1]
                if wins[a, b] >= wins[b, a] and wins[b, c] >= wins[c, b]:
                    n_transitive += 1

    consistency = n_transitive / n_triples if n_triples > 0 else 1.0

    ranked_indices = np.argsort(-scores)
    ranked_ids = [ids[i] for i in ranked_indices]
    score_dict = {ids[i]: float(scores[i]) for i in range(n)}

    return RankingResult(
        ranked_ids=ranked_ids,
        scores=score_dict,
        win_matrix=wins,
        n_comparisons=int(wins.sum()),
        consistency=consistency,
    )


# ---------------------------------------------------------------------------
# Diversity annotation
# ---------------------------------------------------------------------------


def diversity_annotation_interface(
    responses: List[str],
    annotations: Optional[List[Annotation]] = None,
) -> AnnotationResult:
    """Aggregate diversity annotations for a set of responses.

    If *annotations* is None, creates a template result with zero scores
    that can be filled in later.
    """
    ids = [
        hashlib.md5(f"{i}:{r[:50]}".encode()).hexdigest()[:8]
        for i, r in enumerate(responses)
    ]

    if annotations is None:
        # Return empty template
        item_scores = {
            rid: {"diversity": 0.0, "quality": 0.0, "relevance": 0.0}
            for rid in ids
        }
        return AnnotationResult(
            item_scores=item_scores,
            item_rankings={rid: i + 1 for i, rid in enumerate(ids)},
            n_annotations=0,
            annotations=[],
        )

    # Aggregate scores per item per criterion
    score_accum: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ann in annotations:
        for crit, score in ann.scores.items():
            score_accum[ann.item_id][crit].append(score)

    item_scores: Dict[str, Dict[str, float]] = {}
    for item_id, crits in score_accum.items():
        item_scores[item_id] = {
            crit: float(np.mean(vals)) for crit, vals in crits.items()
        }

    # Rank by average across all criteria
    avg_scores = {
        item_id: float(np.mean(list(crits.values())))
        for item_id, crits in item_scores.items()
    }
    sorted_items = sorted(avg_scores, key=lambda x: -avg_scores[x])
    item_rankings = {item_id: rank + 1 for rank, item_id in enumerate(sorted_items)}

    return AnnotationResult(
        item_scores=item_scores,
        item_rankings=item_rankings,
        n_annotations=len(annotations),
        annotations=annotations,
    )


# ---------------------------------------------------------------------------
# Inter-annotator agreement
# ---------------------------------------------------------------------------


def compute_inter_annotator_agreement(
    annotations: List[Annotation],
    method: str = "krippendorff",
) -> float:
    """Compute inter-annotator agreement.

    Parameters
    ----------
    annotations : list of Annotation
        All annotations across annotators.
    method : str
        ``"krippendorff"`` (Krippendorff's alpha) or ``"cohen"`` (Cohen's
        kappa, only for 2 annotators).

    Returns
    -------
    float
        Agreement score (0 = chance, 1 = perfect agreement).
    """
    if len(annotations) < 2:
        return 0.0

    # Build annotator × item score matrix
    annotators = sorted({a.annotator_id for a in annotations})
    items = sorted({a.item_id for a in annotations})

    if len(annotators) < 2:
        return 1.0  # only one annotator

    # Average across criteria for each annotation
    scores: Dict[Tuple[str, str], float] = {}
    for ann in annotations:
        avg = float(np.mean(list(ann.scores.values()))) if ann.scores else 0.0
        scores[(ann.annotator_id, ann.item_id)] = avg

    if method == "cohen" and len(annotators) == 2:
        return _cohens_kappa(annotators, items, scores)

    return _krippendorff_alpha(annotators, items, scores)


def _cohens_kappa(
    annotators: List[str],
    items: List[str],
    scores: Dict[Tuple[str, str], float],
) -> float:
    """Cohen's weighted kappa for two annotators."""
    a1, a2 = annotators[0], annotators[1]
    shared_items = [
        it for it in items
        if (a1, it) in scores and (a2, it) in scores
    ]
    if len(shared_items) < 2:
        return 0.0

    x = np.array([scores[(a1, it)] for it in shared_items])
    y = np.array([scores[(a2, it)] for it in shared_items])

    # Weighted kappa using squared difference weights
    n = len(shared_items)
    max_diff = max(x.max() - x.min(), y.max() - y.min(), 1.0)

    observed = np.mean((x - y) ** 2) / (max_diff ** 2)
    # Expected under independence
    expected = 0.0
    for xi in x:
        for yj in y:
            expected += (xi - yj) ** 2
    expected /= (n * n * max_diff ** 2)

    if expected == 0:
        return 1.0
    return 1.0 - observed / expected


def _krippendorff_alpha(
    annotators: List[str],
    items: List[str],
    scores: Dict[Tuple[str, str], float],
) -> float:
    """Krippendorff's alpha for interval data."""
    # Collect all paired observations
    pairs: List[Tuple[float, float]] = []
    for item in items:
        item_scores = [
            scores[(a, item)] for a in annotators if (a, item) in scores
        ]
        if len(item_scores) < 2:
            continue
        for i in range(len(item_scores)):
            for j in range(i + 1, len(item_scores)):
                pairs.append((item_scores[i], item_scores[j]))

    if not pairs:
        return 0.0

    # Observed disagreement
    d_o = np.mean([(a - b) ** 2 for a, b in pairs])

    # Expected disagreement (all values pooled)
    all_vals = [v for pair in pairs for v in pair]
    n_vals = len(all_vals)
    if n_vals < 2:
        return 0.0

    mean_val = np.mean(all_vals)
    d_e = np.var(all_vals)

    if d_e == 0:
        return 1.0

    return 1.0 - d_o / d_e


# ---------------------------------------------------------------------------
# Correlation with automatic metrics
# ---------------------------------------------------------------------------


def correlate_with_metrics(
    human_scores: List[float],
    auto_scores: List[float],
) -> CorrelationResult:
    """Compute correlation between human judgments and automatic metrics.

    Returns Pearson, Spearman, and Kendall correlations with p-values.
    """
    if len(human_scores) != len(auto_scores):
        raise ValueError("Score lists must have equal length")
    n = len(human_scores)
    if n < 3:
        return CorrelationResult(
            pearson_r=0.0, pearson_p=1.0,
            spearman_rho=0.0, spearman_p=1.0,
            kendall_tau=0.0, kendall_p=1.0,
            n_items=n,
            human_scores=list(human_scores),
            auto_scores=list(auto_scores),
        )

    h = np.array(human_scores, dtype=float)
    a = np.array(auto_scores, dtype=float)

    # Handle constant arrays
    if np.std(h) == 0 or np.std(a) == 0:
        return CorrelationResult(
            pearson_r=0.0, pearson_p=1.0,
            spearman_rho=0.0, spearman_p=1.0,
            kendall_tau=0.0, kendall_p=1.0,
            n_items=n,
            human_scores=list(human_scores),
            auto_scores=list(auto_scores),
        )

    pr, pp = sp_stats.pearsonr(h, a)
    sr, sp = sp_stats.spearmanr(h, a)
    kt, kp = sp_stats.kendalltau(h, a)

    return CorrelationResult(
        pearson_r=float(pr),
        pearson_p=float(pp),
        spearman_rho=float(sr),
        spearman_p=float(sp),
        kendall_tau=float(kt),
        kendall_p=float(kp),
        n_items=n,
        human_scores=list(human_scores),
        auto_scores=list(auto_scores),
    )
