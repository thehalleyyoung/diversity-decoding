"""Fairness through diversity measurement.

Provides tools for analysing demographic representation, detecting
representation gaps, generating balanced outputs, and computing
intersectional diversity metrics across multiple demographic axes.
"""

from __future__ import annotations

import math
import re
import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np

# ---------------------------------------------------------------------------
# Word-list catalogues used by the detection helpers
# ---------------------------------------------------------------------------

_GENDER_TERMS: Dict[str, List[str]] = {
    "male": [
        "he", "him", "his", "man", "men", "boy", "boys", "father",
        "husband", "brother", "son", "gentleman", "mr", "king",
        "prince", "male", "masculine",
    ],
    "female": [
        "she", "her", "hers", "woman", "women", "girl", "girls",
        "mother", "wife", "sister", "daughter", "lady", "mrs", "ms",
        "queen", "princess", "female", "feminine",
    ],
    "non_binary": [
        "they", "them", "non-binary", "nonbinary", "genderqueer",
        "genderfluid", "agender", "enby",
    ],
}

_AGE_TERMS: Dict[str, List[str]] = {
    "child": [
        "child", "children", "kid", "kids", "toddler", "infant",
        "baby", "juvenile", "minor", "youth",
    ],
    "young_adult": [
        "teenager", "teen", "adolescent", "young adult", "college",
        "student", "millennial", "gen-z",
    ],
    "middle_aged": [
        "middle-aged", "midlife", "adult", "parent", "gen-x",
    ],
    "elderly": [
        "elderly", "senior", "retiree", "grandfather", "grandmother",
        "grandparent", "aged", "older adult", "boomer",
    ],
}

_GEO_TERMS: Dict[str, List[str]] = {
    "north_america": [
        "american", "canadian", "mexican", "united states", "usa",
        "canada", "mexico", "north america",
    ],
    "europe": [
        "european", "british", "french", "german", "italian",
        "spanish", "dutch", "swedish", "polish", "europe",
    ],
    "asia": [
        "asian", "chinese", "japanese", "indian", "korean",
        "vietnamese", "thai", "indonesian", "asia",
    ],
    "africa": [
        "african", "nigerian", "kenyan", "south african",
        "ethiopian", "ghanaian", "egypt", "africa",
    ],
    "south_america": [
        "south american", "brazilian", "argentinian", "colombian",
        "peruvian", "chilean", "south america",
    ],
    "oceania": [
        "australian", "new zealander", "pacific islander",
        "oceania", "australia",
    ],
}

_PROFESSION_TERMS: Dict[str, List[str]] = {
    "stem": [
        "engineer", "scientist", "developer", "programmer",
        "researcher", "mathematician", "physicist", "biologist",
        "chemist", "technologist",
    ],
    "healthcare": [
        "doctor", "nurse", "physician", "surgeon", "therapist",
        "pharmacist", "dentist", "paramedic", "healthcare",
    ],
    "arts": [
        "artist", "musician", "writer", "painter", "actor",
        "filmmaker", "designer", "photographer", "poet",
    ],
    "trades": [
        "plumber", "electrician", "carpenter", "mechanic",
        "welder", "mason", "construction", "technician",
    ],
    "service": [
        "teacher", "firefighter", "police", "social worker",
        "chef", "driver", "cashier", "janitor", "waiter",
    ],
}

_ETHNICITY_TERMS: Dict[str, List[str]] = {
    "white": ["white", "caucasian", "european-american"],
    "black": ["black", "african-american", "afro"],
    "hispanic": ["hispanic", "latino", "latina", "latinx", "chicano"],
    "east_asian": ["east asian", "chinese", "japanese", "korean"],
    "south_asian": ["south asian", "indian", "pakistani", "bangladeshi"],
    "indigenous": ["indigenous", "native", "aboriginal", "first nations"],
    "middle_eastern": ["middle eastern", "arab", "persian"],
}

_ALL_AXES: Dict[str, Dict[str, List[str]]] = {
    "gender": _GENDER_TERMS,
    "age": _AGE_TERMS,
    "geography": _GEO_TERMS,
    "profession": _PROFESSION_TERMS,
    "ethnicity": _ETHNICITY_TERMS,
}

# Synonym expansion tables for perspective coverage
_SYNONYM_TABLE: Dict[str, List[str]] = {
    "happy": ["joyful", "glad", "cheerful", "content", "pleased", "elated"],
    "sad": ["unhappy", "sorrowful", "melancholy", "gloomy", "depressed"],
    "important": ["significant", "crucial", "vital", "essential", "key"],
    "difficult": ["hard", "challenging", "tough", "complex", "arduous"],
    "good": ["excellent", "great", "fine", "positive", "beneficial"],
    "bad": ["poor", "negative", "harmful", "detrimental", "adverse"],
    "economic": ["financial", "fiscal", "monetary", "commercial"],
    "political": ["governmental", "civic", "legislative", "policy"],
    "environmental": ["ecological", "green", "sustainable", "climate"],
    "social": ["societal", "communal", "community", "interpersonal"],
    "scientific": ["empirical", "evidence-based", "research", "analytical"],
    "cultural": ["ethnic", "traditional", "heritage", "multicultural"],
    "religious": ["spiritual", "faith", "theological", "sacred"],
    "technological": ["digital", "tech", "computational", "cyber"],
    "conservative": ["traditional", "right-wing", "conventional"],
    "progressive": ["liberal", "left-wing", "reformist", "forward-looking"],
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FairnessReport:
    """Result of a demographic diversity analysis."""

    overall_score: float
    per_demographic_scores: Dict[str, float]
    representation_ratios: Dict[str, float]
    disparity_index: float
    recommendations: List[str]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoverageReport:
    """Result of a perspective coverage analysis."""

    covered_perspectives: List[str]
    uncovered_perspectives: List[str]
    coverage_ratio: float
    balance_score: float
    per_perspective_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RepGap:
    """A single representation gap finding."""

    category: str
    expected_ratio: float
    actual_ratio: float
    gap_magnitude: float
    severity: str
    affected_terms: List[str]


@dataclass
class IntersectionalReport:
    """Result of an intersectional diversity analysis."""

    axes: List[str]
    intersections: List[Tuple[str, ...]]
    per_intersection_scores: Dict[Tuple[str, ...], float]
    worst_intersections: List[Tuple[Tuple[str, ...], float]]
    overall_score: float
    disparity_matrix: np.ndarray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Lower-case tokenisation with basic punctuation stripping."""
    return re.findall(r"[a-z]+(?:[-'][a-z]+)*", text.lower())


def _count_term_hits(
    tokens: List[str],
    text_lower: str,
    terms: List[str],
) -> int:
    """Count how many *terms* appear in *tokens* or *text_lower*."""
    hits = 0
    token_set = set(tokens)
    for term in terms:
        if " " in term:
            hits += text_lower.count(term)
        elif term in token_set:
            hits += tokens.count(term)
    return hits


def _gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient for an array of non-negative values."""
    if len(values) == 0:
        return 0.0
    sorted_v = np.sort(values)
    n = len(sorted_v)
    total = sorted_v.sum()
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_v) / (n * total)) - (n + 1) / n)


def _chi_squared_uniform(observed: np.ndarray) -> float:
    """Chi-squared statistic against a uniform expected distribution."""
    total = observed.sum()
    if total == 0:
        return 0.0
    k = len(observed)
    expected = total / k
    return float(np.sum((observed - expected) ** 2 / expected))


def _max_min_ratio(values: np.ndarray) -> float:
    """Ratio between the maximum and minimum values (clipped to avoid div/0)."""
    mn = values.min()
    mx = values.max()
    if mn <= 0:
        return float("inf") if mx > 0 else 1.0
    return float(mx / mn)


def _severity_label(gap: float) -> str:
    """Map an absolute gap magnitude to a human-readable severity label."""
    if gap < 0.05:
        return "negligible"
    if gap < 0.15:
        return "low"
    if gap < 0.30:
        return "moderate"
    if gap < 0.50:
        return "high"
    return "critical"


def _expand_synonyms(term: str) -> Set[str]:
    """Return a set containing *term* and its synonyms from the lookup table."""
    result: Set[str] = {term.lower()}
    for base, syns in _SYNONYM_TABLE.items():
        if term.lower() == base or term.lower() in syns:
            result.add(base)
            result.update(s.lower() for s in syns)
    return result


def _axis_catalogue(axis: str) -> Dict[str, List[str]]:
    """Return the term catalogue for a named axis, falling back to empty."""
    return _ALL_AXES.get(axis, {})


# ---------------------------------------------------------------------------
# BiasDetector
# ---------------------------------------------------------------------------


class BiasDetector:
    """Detect demographic bias in text using keyword lists and co-occurrence.

    The detector counts mentions of demographic groups and computes a
    per-group bias score that reflects deviation from uniform mention.
    Sentiment-like signals are approximated by tracking positive/negative
    adjective co-occurrence near group terms.
    """

    _POSITIVE_ADJ = {
        "good", "great", "excellent", "brilliant", "talented", "skilled",
        "successful", "strong", "capable", "intelligent", "kind",
        "beautiful", "innovative", "respected", "reliable", "heroic",
    }
    _NEGATIVE_ADJ = {
        "bad", "poor", "weak", "lazy", "criminal", "dangerous",
        "aggressive", "incompetent", "stupid", "ugly", "dirty",
        "dishonest", "violent", "corrupt", "inferior", "hostile",
    }

    def __init__(
        self,
        axes: Optional[List[str]] = None,
        window: int = 5,
    ) -> None:
        self.axes = axes or list(_ALL_AXES.keys())
        self.window = window
        self._catalogues: Dict[str, Dict[str, List[str]]] = {
            ax: _axis_catalogue(ax) for ax in self.axes
        }

    # ---- public API ----

    def detect(self, text: str) -> Dict[str, float]:
        """Return per-group bias scores for a single text.

        Scores range roughly from -1 (negative sentiment association)
        through 0 (neutral / balanced) to +1 (positive sentiment
        association).  The magnitude also reflects mention frequency.
        """
        tokens = _tokenize(text)
        text_lower = text.lower()
        scores: Dict[str, float] = {}
        for axis, groups in self._catalogues.items():
            for group, terms in groups.items():
                mention_count = _count_term_hits(tokens, text_lower, terms)
                if mention_count == 0:
                    scores[f"{axis}:{group}"] = 0.0
                    continue
                sentiment = self._local_sentiment(tokens, terms)
                scores[f"{axis}:{group}"] = round(
                    sentiment * math.log1p(mention_count), 4
                )
        return scores

    def aggregate(self, texts: List[str]) -> Dict[str, float]:
        """Aggregate bias scores across many texts (mean per group)."""
        accum: Dict[str, List[float]] = defaultdict(list)
        for text in texts:
            for key, val in self.detect(text).items():
                accum[key].append(val)
        return {k: round(float(np.mean(v)), 4) for k, v in accum.items()}

    def compare_groups(
        self,
        texts: List[str],
        groups: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compare bias scores between specified groups.

        *groups* should be strings like ``"gender:male"`` matching the
        keys returned by :meth:`detect`.

        Returns a dict mapping each group to ``{"mean", "std", "count"}``.
        """
        per_group: Dict[str, List[float]] = {g: [] for g in groups}
        for text in texts:
            scores = self.detect(text)
            for g in groups:
                per_group[g].append(scores.get(g, 0.0))
        result: Dict[str, Dict[str, float]] = {}
        for g, vals in per_group.items():
            arr = np.array(vals)
            result[g] = {
                "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std()), 4),
                "count": float(len(arr)),
            }
        return result

    # ---- internals ----

    def _local_sentiment(self, tokens: List[str], terms: List[str]) -> float:
        """Compute a sentiment proxy around term mentions in *tokens*."""
        term_set = set(terms)
        pos_count = 0
        neg_count = 0
        for i, tok in enumerate(tokens):
            if tok not in term_set:
                continue
            window_start = max(0, i - self.window)
            window_end = min(len(tokens), i + self.window + 1)
            window_tokens = set(tokens[window_start:window_end])
            pos_count += len(window_tokens & self._POSITIVE_ADJ)
            neg_count += len(window_tokens & self._NEGATIVE_ADJ)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total


# ---------------------------------------------------------------------------
# RepresentationBalancer
# ---------------------------------------------------------------------------


class RepresentationBalancer:
    """Algorithms for balancing demographic representation.

    Supports three strategies:

    * **resample** – oversample under-represented groups and undersample
      over-represented ones to reach a target distribution.
    * **reweight** – assign importance weights so that weighted statistics
      match a target distribution.
    * **diverse_subset** – greedily select a maximally diverse subset of a
      given size.
    """

    def __init__(
        self,
        detector: Optional[BiasDetector] = None,
    ) -> None:
        self.detector = detector or BiasDetector()

    def resample(
        self,
        responses: List[str],
        target_dist: Optional[Dict[str, float]] = None,
        size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[str]:
        """Resample *responses* to approximate *target_dist*.

        If *target_dist* is ``None``, a uniform distribution over detected
        groups is used.  *size* defaults to ``len(responses)``.
        """
        rng = rng or np.random.default_rng(42)
        size = size or len(responses)
        group_assignments = self._assign_groups(responses)
        groups = sorted(set(group_assignments))
        if not groups:
            indices = rng.choice(len(responses), size=size, replace=True)
            return [responses[int(i)] for i in indices]

        if target_dist is None:
            target_dist = {g: 1.0 / len(groups) for g in groups}

        bucket: Dict[str, List[int]] = {g: [] for g in groups}
        for idx, g in enumerate(group_assignments):
            if g in bucket:
                bucket[g].append(idx)

        sampled: List[str] = []
        for g in groups:
            n_target = max(1, int(round(target_dist.get(g, 0) * size)))
            pool = bucket.get(g, [])
            if not pool:
                continue
            indices = rng.choice(pool, size=n_target, replace=True)
            sampled.extend(responses[int(i)] for i in indices)

        if len(sampled) > size:
            indices = rng.choice(len(sampled), size=size, replace=False)
            sampled = [sampled[int(i)] for i in sorted(indices)]
        return sampled

    def reweight(
        self,
        responses: List[str],
        target_dist: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        """Return per-response importance weights matching *target_dist*."""
        group_assignments = self._assign_groups(responses)
        groups = sorted(set(group_assignments))
        if not groups:
            return [1.0] * len(responses)

        if target_dist is None:
            target_dist = {g: 1.0 / len(groups) for g in groups}

        group_counts: Counter[str] = Counter(group_assignments)
        total = sum(group_counts.values())
        weights: List[float] = []
        for g in group_assignments:
            actual_ratio = group_counts[g] / total if total else 1.0
            target_ratio = target_dist.get(g, actual_ratio)
            w = target_ratio / actual_ratio if actual_ratio > 0 else 1.0
            weights.append(round(w, 4))
        return weights

    def diverse_subset(
        self,
        responses: List[str],
        k: int,
    ) -> List[str]:
        """Greedily select *k* responses maximising group diversity."""
        if k >= len(responses):
            return list(responses)

        group_assignments = self._assign_groups(responses)
        selected_indices: List[int] = []
        selected_groups: Counter[str] = Counter()
        remaining = set(range(len(responses)))

        for _ in range(k):
            best_idx = -1
            best_score = -1.0
            for idx in remaining:
                g = group_assignments[idx]
                trial = Counter(selected_groups)
                trial[g] += 1
                counts = np.array(list(trial.values()), dtype=float)
                score = float(counts.min()) / float(counts.max()) if counts.max() > 0 else 1.0
                unique_bonus = len(trial) / (len(_ALL_AXES) + 1)
                combined = score + unique_bonus
                if combined > best_score:
                    best_score = combined
                    best_idx = idx
            if best_idx < 0:
                break
            selected_indices.append(best_idx)
            selected_groups[group_assignments[best_idx]] += 1
            remaining.discard(best_idx)

        return [responses[i] for i in selected_indices]

    # ---- internals ----

    def _assign_groups(self, responses: List[str]) -> List[str]:
        """Assign each response to its dominant demographic group."""
        assignments: List[str] = []
        for resp in responses:
            scores = self.detector.detect(resp)
            if not scores:
                assignments.append("_unknown")
                continue
            mention_counts: Dict[str, int] = {}
            tokens = _tokenize(resp)
            text_lower = resp.lower()
            for axis, groups in self.detector._catalogues.items():
                for group, terms in groups.items():
                    c = _count_term_hits(tokens, text_lower, terms)
                    if c > 0:
                        mention_counts[f"{axis}:{group}"] = c
            if mention_counts:
                dominant = max(mention_counts, key=mention_counts.get)  # type: ignore[arg-type]
                assignments.append(dominant)
            else:
                assignments.append("_unknown")
        return assignments


# ---------------------------------------------------------------------------
# FairnessMetrics
# ---------------------------------------------------------------------------


class FairnessMetrics:
    """Statistical fairness metrics for evaluating representation equity."""

    @staticmethod
    def demographic_parity(
        group_counts: Dict[str, int],
    ) -> Dict[str, float]:
        """Compute demographic parity difference from uniform.

        Returns a dict mapping each group to the signed difference between
        its actual proportion and the expected uniform proportion.
        """
        total = sum(group_counts.values())
        if total == 0:
            return {g: 0.0 for g in group_counts}
        k = len(group_counts)
        expected = 1.0 / k if k else 1.0
        return {
            g: round(c / total - expected, 4)
            for g, c in group_counts.items()
        }

    @staticmethod
    def equalized_odds_proxy(
        group_positive_rates: Dict[str, float],
    ) -> float:
        """Proxy for equalised odds: max difference in positive rates.

        In a generative context *positive rate* can be the fraction of
        responses mentioning a group in a favourable context.
        """
        if not group_positive_rates:
            return 0.0
        rates = list(group_positive_rates.values())
        return round(max(rates) - min(rates), 4)

    @staticmethod
    def individual_fairness_proxy(
        embeddings: np.ndarray,
        group_labels: List[str],
        metric: str = "euclidean",
    ) -> float:
        """Proxy for individual fairness via intra/inter-group distances.

        Computes the ratio of mean intra-group distance to mean
        inter-group distance.  Values close to 1.0 indicate that
        similar individuals (close in embedding space) receive similar
        treatment regardless of group membership.
        """
        n = len(group_labels)
        if n < 2 or embeddings.shape[0] != n:
            return 1.0

        if metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-10, None)
            normed = embeddings / norms
            dist_matrix = 1.0 - normed @ normed.T
        else:
            diff = embeddings[:, None, :] - embeddings[None, :, :]
            dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))

        intra_dists: List[float] = []
        inter_dists: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(dist_matrix[i, j])
                if group_labels[i] == group_labels[j]:
                    intra_dists.append(d)
                else:
                    inter_dists.append(d)

        mean_intra = float(np.mean(intra_dists)) if intra_dists else 0.0
        mean_inter = float(np.mean(inter_dists)) if inter_dists else 1.0
        if mean_inter == 0:
            return 1.0
        return round(mean_intra / mean_inter, 4)


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def demographic_diversity(
    responses: List[str],
    demographics: List[str],
) -> FairnessReport:
    """Analyse demographic representation in *responses*.

    Parameters
    ----------
    responses:
        Texts to analyse.
    demographics:
        Demographic group labels to look for (e.g.
        ``["male", "female", "non_binary"]``).

    Returns
    -------
    FairnessReport
        Comprehensive fairness analysis including Gini coefficient,
        chi-squared statistic, max-min ratio, and actionable
        recommendations.
    """
    all_terms: Dict[str, List[str]] = {}
    for demo in demographics:
        for axis_terms in _ALL_AXES.values():
            if demo in axis_terms:
                all_terms[demo] = axis_terms[demo]
                break
        if demo not in all_terms:
            all_terms[demo] = [demo.lower()]

    group_counts: Dict[str, int] = {d: 0 for d in demographics}
    for resp in responses:
        tokens = _tokenize(resp)
        text_lower = resp.lower()
        for demo in demographics:
            group_counts[demo] += _count_term_hits(
                tokens, text_lower, all_terms[demo]
            )

    total_mentions = sum(group_counts.values())
    representation_ratios: Dict[str, float] = {}
    for demo in demographics:
        representation_ratios[demo] = round(
            group_counts[demo] / total_mentions if total_mentions else 0.0, 4
        )

    counts_arr = np.array([group_counts[d] for d in demographics], dtype=float)
    gini = _gini(counts_arr)
    chi2 = _chi_squared_uniform(counts_arr)
    mmr = _max_min_ratio(counts_arr)

    disparity_index = round(gini, 4)

    expected = 1.0 / len(demographics) if demographics else 1.0
    per_demo_scores: Dict[str, float] = {}
    for demo in demographics:
        deviation = abs(representation_ratios[demo] - expected)
        per_demo_scores[demo] = round(max(0.0, 1.0 - deviation / expected), 4)

    overall = round(float(np.mean(list(per_demo_scores.values()))), 4) if per_demo_scores else 0.0

    recommendations: List[str] = []
    under = [d for d in demographics if representation_ratios[d] < expected * 0.5]
    over = [d for d in demographics if representation_ratios[d] > expected * 2.0]
    if under:
        recommendations.append(
            f"Under-represented groups detected: {', '.join(under)}. "
            "Consider augmenting prompts to include these perspectives."
        )
    if over:
        recommendations.append(
            f"Over-represented groups detected: {', '.join(over)}. "
            "Consider diversifying generation to reduce dominance."
        )
    if gini > 0.3:
        recommendations.append(
            f"High Gini coefficient ({gini:.3f}) indicates significant "
            "inequality in representation."
        )
    if not recommendations:
        recommendations.append("Representation appears reasonably balanced.")

    details: Dict[str, Any] = {
        "gini_coefficient": round(gini, 4),
        "chi_squared": round(chi2, 4),
        "max_min_ratio": round(mmr, 4) if mmr != float("inf") else "inf",
        "total_mentions": total_mentions,
        "raw_counts": dict(group_counts),
    }

    return FairnessReport(
        overall_score=overall,
        per_demographic_scores=per_demo_scores,
        representation_ratios=representation_ratios,
        disparity_index=disparity_index,
        recommendations=recommendations,
        details=details,
    )


def perspective_coverage(
    responses: List[str],
    perspectives: List[str],
) -> CoverageReport:
    """Measure how well *responses* cover the given *perspectives*.

    Keyword / phrase matching is augmented with a synonym expansion
    table to improve recall.

    Parameters
    ----------
    responses:
        Texts to analyse.
    perspectives:
        Perspective labels (e.g. ``["economic", "environmental"]``).

    Returns
    -------
    CoverageReport
    """
    expanded: Dict[str, Set[str]] = {}
    for p in perspectives:
        expanded[p] = _expand_synonyms(p)

    per_perspective: Dict[str, Dict[str, Any]] = {}
    covered: List[str] = []
    uncovered: List[str] = []
    hit_counts: Dict[str, int] = {}

    for p in perspectives:
        terms = expanded[p]
        total_hits = 0
        matching_responses = 0
        for resp in responses:
            tokens = _tokenize(resp)
            text_lower = resp.lower()
            hits = sum(
                1 for t in terms
                if (" " in t and t in text_lower) or t in set(tokens)
            )
            total_hits += hits
            if hits > 0:
                matching_responses += 1

        hit_counts[p] = total_hits
        resp_ratio = matching_responses / len(responses) if responses else 0.0
        per_perspective[p] = {
            "hit_count": total_hits,
            "response_ratio": round(resp_ratio, 4),
            "matching_responses": matching_responses,
            "expanded_terms": sorted(terms),
        }
        if total_hits > 0:
            covered.append(p)
        else:
            uncovered.append(p)

    coverage_ratio = round(
        len(covered) / len(perspectives) if perspectives else 0.0, 4
    )

    counts = np.array([hit_counts[p] for p in perspectives], dtype=float)
    if counts.sum() > 0:
        proportions = counts / counts.sum()
        entropy = -float(np.sum(p * np.log(p + 1e-12) for p in proportions))
        max_entropy = math.log(len(perspectives)) if len(perspectives) > 1 else 1.0
        balance_score = round(entropy / max_entropy, 4)
    else:
        balance_score = 0.0

    return CoverageReport(
        covered_perspectives=covered,
        uncovered_perspectives=uncovered,
        coverage_ratio=coverage_ratio,
        balance_score=balance_score,
        per_perspective_details=per_perspective,
    )


def detect_representation_gaps(
    responses: List[str],
) -> List[RepGap]:
    """Detect under- and over-representation across multiple axes.

    Analyses gender, age, geography, profession, and ethnicity using
    curated word lists and flags significant deviations from a balanced
    (uniform) distribution within each axis.

    Returns
    -------
    List[RepGap]
        Sorted by gap magnitude descending.
    """
    gaps: List[RepGap] = []

    for axis_name, groups in _ALL_AXES.items():
        group_counts: Dict[str, int] = {}
        group_matched_terms: Dict[str, List[str]] = defaultdict(list)
        for group, terms in groups.items():
            count = 0
            for resp in responses:
                tokens = _tokenize(resp)
                text_lower = resp.lower()
                for t in terms:
                    c = _count_term_hits(tokens, text_lower, [t])
                    if c > 0:
                        count += c
                        if t not in group_matched_terms[group]:
                            group_matched_terms[group].append(t)
            group_counts[group] = count

        total = sum(group_counts.values())
        if total == 0:
            continue

        n_groups = len(groups)
        expected_ratio = 1.0 / n_groups

        for group, count in group_counts.items():
            actual_ratio = count / total
            gap_mag = abs(actual_ratio - expected_ratio)
            severity = _severity_label(gap_mag)
            if severity != "negligible":
                gaps.append(
                    RepGap(
                        category=f"{axis_name}:{group}",
                        expected_ratio=round(expected_ratio, 4),
                        actual_ratio=round(actual_ratio, 4),
                        gap_magnitude=round(gap_mag, 4),
                        severity=severity,
                        affected_terms=group_matched_terms[group],
                    )
                )

    gaps.sort(key=lambda g: g.gap_magnitude, reverse=True)
    return gaps


def balanced_generation(
    gen_fn: Callable[..., str],
    categories: List[str],
    n_per: int,
) -> Dict[str, List[str]]:
    """Generate balanced outputs ensuring *n_per* diverse items per category.

    Uses iterative generation with rejection-based deduplication: if a
    generated response is too similar to an existing one in the same
    category (exact match or high token overlap), it is discarded and
    regenerated.

    Parameters
    ----------
    gen_fn:
        A callable that accepts a *category* string and returns a single
        generated response string.
    categories:
        Category labels to balance across.
    n_per:
        Number of distinct responses desired per category.

    Returns
    -------
    Dict[str, List[str]]
    """
    max_attempts_per = n_per * 5
    results: Dict[str, List[str]] = {cat: [] for cat in categories}

    for cat in categories:
        seen_normalized: Set[str] = set()
        attempts = 0
        while len(results[cat]) < n_per and attempts < max_attempts_per:
            attempts += 1
            try:
                resp = gen_fn(cat)
            except Exception:
                continue

            if not isinstance(resp, str) or not resp.strip():
                continue

            normalized = " ".join(_tokenize(resp))
            if normalized in seen_normalized:
                continue

            is_duplicate = False
            resp_tokens = set(_tokenize(resp))
            for existing in results[cat]:
                existing_tokens = set(_tokenize(existing))
                if not resp_tokens or not existing_tokens:
                    continue
                overlap = len(resp_tokens & existing_tokens)
                union = len(resp_tokens | existing_tokens)
                jaccard = overlap / union if union > 0 else 0.0
                if jaccard > 0.85:
                    is_duplicate = True
                    break

            if not is_duplicate:
                results[cat].append(resp)
                seen_normalized.add(normalized)

    return results


def debiased_selection(
    responses: List[str],
    bias_detector: Optional[Callable[[str], Dict[str, float]]] = None,
) -> List[str]:
    """Select a subset of *responses* that minimises demographic bias.

    Uses a greedy algorithm: at each step, the response that best
    balances the cumulative demographic representation is chosen.

    Parameters
    ----------
    responses:
        Candidate responses to select from.
    bias_detector:
        A callable mapping a text to per-group bias scores.  If
        ``None``, the built-in :class:`BiasDetector` is used.

    Returns
    -------
    List[str]
        A subset of *responses* ordered by selection step.
    """
    if not responses:
        return []

    if bias_detector is None:
        _det = BiasDetector()
        bias_detector = _det.detect

    per_response_scores: List[Dict[str, float]] = [
        bias_detector(r) for r in responses
    ]

    all_groups: Set[str] = set()
    for s in per_response_scores:
        all_groups.update(s.keys())
    if not all_groups:
        return list(responses)

    target_size = max(1, len(responses) // 2)
    selected_indices: List[int] = []
    cumulative: Dict[str, float] = {g: 0.0 for g in all_groups}
    remaining = set(range(len(responses)))

    for _ in range(target_size):
        best_idx = -1
        best_imbalance = float("inf")

        for idx in remaining:
            trial = dict(cumulative)
            for g in all_groups:
                trial[g] += abs(per_response_scores[idx].get(g, 0.0))
            values = np.array(list(trial.values()))
            if values.sum() == 0:
                imbalance = 0.0
            else:
                props = values / values.sum()
                uniform = 1.0 / len(all_groups)
                imbalance = float(np.sum((props - uniform) ** 2))
            if imbalance < best_imbalance:
                best_imbalance = imbalance
                best_idx = idx

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        for g in all_groups:
            cumulative[g] += abs(per_response_scores[best_idx].get(g, 0.0))
        remaining.discard(best_idx)

    return [responses[i] for i in selected_indices]


def intersectional_diversity(
    responses: List[str],
    axes: List[str],
) -> IntersectionalReport:
    """Measure diversity along intersecting demographic axes.

    For every combination of groups across the requested *axes*,
    computes representation counts and identifies under-served
    intersections.

    Parameters
    ----------
    responses:
        Texts to analyse.
    axes:
        Axis names (keys of the internal catalogue, e.g.
        ``["gender", "profession"]``).

    Returns
    -------
    IntersectionalReport
    """
    catalogues: Dict[str, Dict[str, List[str]]] = {}
    for ax in axes:
        cat = _axis_catalogue(ax)
        if cat:
            catalogues[ax] = cat
    if not catalogues:
        return IntersectionalReport(
            axes=axes,
            intersections=[],
            per_intersection_scores={},
            worst_intersections=[],
            overall_score=0.0,
            disparity_matrix=np.array([]),
        )

    ordered_axes = [ax for ax in axes if ax in catalogues]
    group_names_per_axis = [sorted(catalogues[ax].keys()) for ax in ordered_axes]
    all_intersections: List[Tuple[str, ...]] = list(
        itertools.product(*group_names_per_axis)
    )

    intersection_counts: Dict[Tuple[str, ...], int] = {
        combo: 0 for combo in all_intersections
    }

    for resp in responses:
        tokens = _tokenize(resp)
        text_lower = resp.lower()

        axis_hits: Dict[str, Set[str]] = {ax: set() for ax in ordered_axes}
        for ax in ordered_axes:
            for group, terms in catalogues[ax].items():
                if _count_term_hits(tokens, text_lower, terms) > 0:
                    axis_hits[ax].add(group)

        if any(len(v) == 0 for v in axis_hits.values()):
            continue

        hit_combos = list(itertools.product(
            *(sorted(axis_hits[ax]) for ax in ordered_axes)
        ))
        for combo in hit_combos:
            if combo in intersection_counts:
                intersection_counts[combo] += 1

    total = sum(intersection_counts.values())
    per_intersection_scores: Dict[Tuple[str, ...], float] = {}
    if total > 0:
        expected = total / len(all_intersections)
        for combo, count in intersection_counts.items():
            per_intersection_scores[combo] = round(
                min(count / expected, 2.0) / 2.0, 4
            )
    else:
        per_intersection_scores = {combo: 0.0 for combo in all_intersections}

    sorted_by_score = sorted(
        per_intersection_scores.items(), key=lambda x: x[1]
    )
    worst_intersections = sorted_by_score[: max(5, len(sorted_by_score) // 5)]

    counts_arr = np.array(
        [intersection_counts[c] for c in all_intersections], dtype=float
    )
    if counts_arr.sum() > 0:
        props = counts_arr / counts_arr.sum()
        max_ent = math.log(len(all_intersections)) if len(all_intersections) > 1 else 1.0
        entropy = -float(np.sum(p * np.log(p + 1e-12) for p in props))
        overall_score = round(entropy / max_ent, 4)
    else:
        overall_score = 0.0

    n = len(all_intersections)
    disparity_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            ci = intersection_counts[all_intersections[i]]
            cj = intersection_counts[all_intersections[j]]
            disparity_matrix[i, j] = abs(ci - cj)

    return IntersectionalReport(
        axes=ordered_axes,
        intersections=all_intersections,
        per_intersection_scores=per_intersection_scores,
        worst_intersections=worst_intersections,
        overall_score=overall_score,
        disparity_matrix=disparity_matrix,
    )
