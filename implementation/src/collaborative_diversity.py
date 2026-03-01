"""
Collaborative Diversity — Multi-user diversity measurement and optimization.

Merge contributions from multiple users while maximizing diversity, identify
unique contributions via MinHash/LSH, score team diversity across multiple
axes, assign tasks for maximum output diversity, and track diversity in
real-time collaboration sessions.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
import time
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
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_NUM_HASHES = 128
_DEFAULT_NGRAM_SIZE = 3
_NEAR_DUPLICATE_THRESHOLD = 0.8
_NOVELTY_THRESHOLD = 0.6
_DIVERSITY_WEIGHTS = {
    "within": 0.25,
    "between": 0.35,
    "coverage": 0.25,
    "style": 0.15,
}

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Assignment:
    """Result of a task-to-member assignment optimized for diversity."""

    task_assignments: Dict[str, List[str]]
    diversity_score: float
    coverage_score: float
    balance_score: float
    rationale: str


@dataclass
class DiversityTracker:
    """Tracks diversity evolution in a collaboration session."""

    session_id: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_diversity: float = 0.0
    trend: List[float] = field(default_factory=list)
    contributors: Set[str] = field(default_factory=set)
    timestamps: List[float] = field(default_factory=list)


@dataclass
class ContributionProfile:
    """Statistical profile of a single contributor's outputs."""

    contributor_id: str
    unique_ratio: float
    overlap_with_others: float
    topic_distribution: Dict[str, float] = field(default_factory=dict)
    style_signature: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokenization with punctuation stripping."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _char_ngrams(text: str, n: int = _DEFAULT_NGRAM_SIZE) -> Set[str]:
    """Extract character-level n-grams from *text*."""
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _word_ngrams(text: str, n: int = 2) -> Set[str]:
    """Extract word-level n-grams from *text*."""
    tokens = _tokenize(text)
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _entropy(distribution: np.ndarray) -> float:
    """Shannon entropy of a probability distribution (in nats)."""
    d = distribution[distribution > 0]
    return float(-np.sum(d * np.log(d)))


def _build_vocab(texts: List[str]) -> Dict[str, int]:
    """Build a word-to-index vocabulary from a list of texts."""
    vocab: Dict[str, int] = {}
    for text in texts:
        for tok in _tokenize(text):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def _text_to_bow(text: str, vocab: Dict[str, int]) -> np.ndarray:
    """Convert text to a bag-of-words vector given a vocabulary."""
    vec = np.zeros(len(vocab), dtype=np.float64)
    for tok in _tokenize(text):
        if tok in vocab:
            vec[vocab[tok]] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# MinHashSignature
# ---------------------------------------------------------------------------


class MinHashSignature:
    """MinHash implementation for fast approximate Jaccard similarity.

    Uses universal hash functions of the form ``(a*x + b) mod p`` where *p*
    is a large prime and *a*, *b* are random coefficients.
    """

    _LARGE_PRIME = 2**31 - 1

    def __init__(self, num_hashes: int = _DEFAULT_NUM_HASHES, seed: int = 42) -> None:
        self.num_hashes = num_hashes
        rng = random.Random(seed)
        self._a = [rng.randint(1, self._LARGE_PRIME - 1) for _ in range(num_hashes)]
        self._b = [rng.randint(0, self._LARGE_PRIME - 1) for _ in range(num_hashes)]

    def _hash_token(self, token: str) -> int:
        """Deterministic hash for a token string."""
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)

    def compute(self, text: str, num_hashes: int | None = None) -> np.ndarray:
        """Compute the MinHash signature for *text*.

        Parameters
        ----------
        text:
            Input string.
        num_hashes:
            Override the default number of hash functions.

        Returns
        -------
        np.ndarray of shape ``(num_hashes,)`` with uint64 dtype.
        """
        n = num_hashes if num_hashes is not None else self.num_hashes
        shingles = _char_ngrams(text)
        if not shingles:
            return np.full(n, self._LARGE_PRIME, dtype=np.uint64)

        hashed_shingles = [self._hash_token(s) for s in shingles]
        sig = np.full(n, np.iinfo(np.uint64).max, dtype=np.uint64)
        for h_val in hashed_shingles:
            for i in range(n):
                val = (self._a[i % self.num_hashes] * h_val + self._b[i % self.num_hashes]) % self._LARGE_PRIME
                if val < sig[i]:
                    sig[i] = val
        return sig

    @staticmethod
    def similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have the same length")
        return float(np.mean(sig1 == sig2))


# ---------------------------------------------------------------------------
# ContributionAnalyzer
# ---------------------------------------------------------------------------


class ContributionAnalyzer:
    """Analyze individual contributions for novelty, topic distribution,
    and writing style.

    Style is captured via simple proxy features: average sentence length,
    vocabulary richness (type-token ratio), punctuation density, and
    function-word ratio.
    """

    _FUNCTION_WORDS: Set[str] = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "of", "in", "to",
        "for", "with", "on", "at", "from", "by", "about", "as", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "no", "only", "own", "same", "than",
        "too", "very", "just", "because", "if", "when", "while", "although",
        "it", "its", "this", "that", "these", "those", "i", "you", "he",
        "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    }

    def __init__(self, minhash: MinHashSignature | None = None) -> None:
        self._minhash = minhash or MinHashSignature()

    # -- style helpers ---------------------------------------------------

    @staticmethod
    def _avg_sentence_length(text: str) -> float:
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        return float(np.mean([len(s.split()) for s in sentences]))

    @staticmethod
    def _type_token_ratio(tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    @staticmethod
    def _punctuation_density(text: str) -> float:
        if not text:
            return 0.0
        punct = sum(1 for ch in text if ch in ".,;:!?-\"'()[]{}…")
        return punct / len(text)

    def _function_word_ratio(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        return sum(1 for t in tokens if t in self._FUNCTION_WORDS) / len(tokens)

    def _compute_style(self, text: str) -> Dict[str, float]:
        tokens = _tokenize(text)
        return {
            "avg_sentence_length": self._avg_sentence_length(text),
            "type_token_ratio": self._type_token_ratio(tokens),
            "punctuation_density": self._punctuation_density(text),
            "function_word_ratio": self._function_word_ratio(tokens),
        }

    # -- topic distribution via simple word clusters ----------------------

    @staticmethod
    def _topic_distribution(texts: List[str], num_topics: int = 10) -> Dict[str, float]:
        """Estimate a topic distribution by hashing words into buckets."""
        counts = np.zeros(num_topics, dtype=np.float64)
        for text in texts:
            for tok in _tokenize(text):
                bucket = int(hashlib.sha256(tok.encode()).hexdigest(), 16) % num_topics
                counts[bucket] += 1.0
        total = counts.sum()
        if total > 0:
            counts /= total
        return {f"topic_{i}": float(counts[i]) for i in range(num_topics)}

    # -- public API -------------------------------------------------------

    def analyze(self, contributor_id: str, texts: List[str]) -> ContributionProfile:
        """Build a :class:`ContributionProfile` for a contributor.

        Parameters
        ----------
        contributor_id:
            Unique identifier for the contributor.
        texts:
            List of text contributions.

        Returns
        -------
        ContributionProfile with computed statistics.
        """
        if not texts:
            return ContributionProfile(
                contributor_id=contributor_id,
                unique_ratio=0.0,
                overlap_with_others=0.0,
            )

        sigs = [self._minhash.compute(t) for t in texts]
        unique_count = 0
        for i, s in enumerate(sigs):
            is_unique = True
            for j, s2 in enumerate(sigs):
                if i != j and MinHashSignature.similarity(s, s2) > _NEAR_DUPLICATE_THRESHOLD:
                    is_unique = False
                    break
            if is_unique:
                unique_count += 1

        combined_text = " ".join(texts)
        style = self._compute_style(combined_text)
        topics = self._topic_distribution(texts)

        return ContributionProfile(
            contributor_id=contributor_id,
            unique_ratio=unique_count / len(texts) if texts else 0.0,
            overlap_with_others=0.0,
            topic_distribution=topics,
            style_signature=style,
        )

    def compare(
        self, profile1: ContributionProfile, profile2: ContributionProfile
    ) -> float:
        """Compute a divergence score between two contributor profiles.

        Uses Jensen-Shannon divergence on topic distributions and Euclidean
        distance on style signatures.  Returns a value in ``[0, 1]`` where
        higher means more different.
        """
        # Topic divergence (JS divergence)
        all_topics = sorted(set(profile1.topic_distribution) | set(profile2.topic_distribution))
        if not all_topics:
            return 0.0
        p = np.array([profile1.topic_distribution.get(t, 0.0) for t in all_topics])
        q = np.array([profile2.topic_distribution.get(t, 0.0) for t in all_topics])
        p_sum, q_sum = p.sum(), q.sum()
        if p_sum > 0:
            p /= p_sum
        if q_sum > 0:
            q /= q_sum
        m = 0.5 * (p + q)
        # avoid log(0)
        eps = 1e-12
        p_safe = np.clip(p, eps, None)
        q_safe = np.clip(q, eps, None)
        m_safe = np.clip(m, eps, None)
        js = 0.5 * np.sum(p_safe * np.log(p_safe / m_safe)) + 0.5 * np.sum(q_safe * np.log(q_safe / m_safe))
        js_norm = float(np.clip(js / np.log(2), 0.0, 1.0))

        # Style distance (normalised Euclidean)
        style_keys = sorted(set(profile1.style_signature) | set(profile2.style_signature))
        if style_keys:
            s1 = np.array([profile1.style_signature.get(k, 0.0) for k in style_keys])
            s2 = np.array([profile2.style_signature.get(k, 0.0) for k in style_keys])
            max_dist = np.sqrt(len(style_keys))
            style_dist = float(np.linalg.norm(s1 - s2) / max_dist) if max_dist > 0 else 0.0
        else:
            style_dist = 0.0

        return 0.6 * js_norm + 0.4 * min(style_dist, 1.0)

    def diversity_contribution(
        self,
        profile: ContributionProfile,
        team_profiles: List[ContributionProfile],
    ) -> float:
        """Measure how much diversity *profile* adds to the team.

        Returns a score in ``[0, 1]`` — higher means the contributor is more
        different from the rest of the team.
        """
        if not team_profiles:
            return 1.0
        distances = [self.compare(profile, tp) for tp in team_profiles if tp.contributor_id != profile.contributor_id]
        if not distances:
            return 1.0
        return float(np.mean(distances))


# ---------------------------------------------------------------------------
# CollaborationOptimizer
# ---------------------------------------------------------------------------


class CollaborationOptimizer:
    """Optimize team collaboration for diversity.

    Uses greedy algorithms over similarity matrices to allocate tasks,
    suggest next contributors, and detect echo-chamber dynamics.
    """

    def __init__(self, minhash: MinHashSignature | None = None) -> None:
        self._minhash = minhash or MinHashSignature()
        self._analyzer = ContributionAnalyzer(self._minhash)

    def optimize_assignment(
        self,
        tasks: List[str],
        members: List[str],
        similarity_matrix: np.ndarray | None = None,
    ) -> Dict[str, List[str]]:
        """Assign *tasks* to *members* to maximise output diversity.

        If *similarity_matrix* is ``None``, pairwise Jaccard similarity over
        task n-grams is computed automatically.  The algorithm greedily assigns
        each task to the member whose current basket is *least similar* to it,
        balancing load across members.

        Returns
        -------
        Dict mapping member id → list of assigned tasks.
        """
        if not tasks or not members:
            return {m: [] for m in members}

        n_tasks = len(tasks)
        n_members = len(members)

        if similarity_matrix is None:
            ngrams = [_word_ngrams(t) for t in tasks]
            similarity_matrix = np.zeros((n_tasks, n_tasks), dtype=np.float64)
            for i in range(n_tasks):
                for j in range(i + 1, n_tasks):
                    sim = _jaccard(ngrams[i], ngrams[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        assignment: Dict[str, List[int]] = {m: [] for m in members}

        task_order = list(range(n_tasks))
        random.Random(42).shuffle(task_order)

        for idx in task_order:
            best_member: str | None = None
            best_score = float("inf")
            for m in members:
                current_indices = assignment[m]
                load_penalty = len(current_indices) / max(n_tasks / n_members, 1)
                if not current_indices:
                    sim_to_basket = 0.0
                else:
                    sim_to_basket = float(np.mean([similarity_matrix[idx, ci] for ci in current_indices]))
                score = sim_to_basket + 0.3 * load_penalty
                if score < best_score:
                    best_score = score
                    best_member = m
            if best_member is not None:
                assignment[best_member].append(idx)

        return {m: [tasks[i] for i in indices] for m, indices in assignment.items()}

    def suggest_next_contributor(
        self, history: List[Tuple[str, str]]
    ) -> str | None:
        """Suggest which contributor should go next to maximize diversity.

        Parameters
        ----------
        history:
            List of ``(contributor_id, text)`` tuples in chronological order.

        Returns
        -------
        The contributor id whose recent contributions are most different from
        the last few entries, or ``None`` if history is empty.
        """
        if not history:
            return None

        contributors: Dict[str, List[str]] = defaultdict(list)
        for cid, text in history:
            contributors[cid].append(text)

        recent_window = [text for _, text in history[-5:]]
        recent_ngrams = set()
        for t in recent_window:
            recent_ngrams |= _char_ngrams(t)

        best_cid: str | None = None
        lowest_sim = float("inf")
        for cid, texts in contributors.items():
            member_ngrams: Set[str] = set()
            for t in texts:
                member_ngrams |= _char_ngrams(t)
            sim = _jaccard(member_ngrams, recent_ngrams)
            if sim < lowest_sim:
                lowest_sim = sim
                best_cid = cid
        return best_cid

    def detect_echo_chamber(
        self, contributions: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Detect echo-chamber dynamics in team contributions.

        Computes pairwise similarity between contributors and flags pairs
        whose similarity exceeds a threshold, indicating potential echo-
        chamber behaviour.

        Returns
        -------
        Dict with keys:
        - ``is_echo_chamber``: bool — whether echo-chamber was detected
        - ``avg_pairwise_similarity``: mean pairwise Jaccard similarity
        - ``flagged_pairs``: list of (member_a, member_b, similarity) tuples
        - ``recommendation``: actionable string
        """
        if len(contributions) < 2:
            return {
                "is_echo_chamber": False,
                "avg_pairwise_similarity": 0.0,
                "flagged_pairs": [],
                "recommendation": "Need at least two contributors to assess.",
            }

        member_sigs: Dict[str, np.ndarray] = {}
        for cid, texts in contributions.items():
            combined = " ".join(texts)
            member_sigs[cid] = self._minhash.compute(combined)

        pairwise_sims: List[float] = []
        flagged: List[Tuple[str, str, float]] = []
        members = list(member_sigs.keys())
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                sim = MinHashSignature.similarity(member_sigs[members[i]], member_sigs[members[j]])
                pairwise_sims.append(sim)
                if sim > _NEAR_DUPLICATE_THRESHOLD:
                    flagged.append((members[i], members[j], round(sim, 4)))

        avg_sim = float(np.mean(pairwise_sims)) if pairwise_sims else 0.0
        is_echo = avg_sim > 0.6 or len(flagged) > 0

        if is_echo:
            rec = (
                f"High similarity detected (avg={avg_sim:.2f}). "
                "Consider assigning distinct sub-topics or bringing in new perspectives."
            )
        else:
            rec = "Team contributions are sufficiently diverse."

        return {
            "is_echo_chamber": is_echo,
            "avg_pairwise_similarity": round(avg_sim, 4),
            "flagged_pairs": flagged,
            "recommendation": rec,
        }


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


class SessionManager:
    """Manage collaboration sessions with running diversity statistics.

    Supports multiple concurrent sessions via a dict-based store.  Each
    session tracks contributions, running diversity scores, per-contributor
    profiles, and trend data.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._minhash = MinHashSignature()
        self._analyzer = ContributionAnalyzer(self._minhash)

    def create_session(self, session_id: str) -> DiversityTracker:
        """Create and return a new :class:`DiversityTracker`."""
        tracker = DiversityTracker(session_id=session_id)
        self._sessions[session_id] = {
            "tracker": tracker,
            "contributions": defaultdict(list),
            "profiles": {},
        }
        return tracker

    def get_session(self, session_id: str) -> DiversityTracker | None:
        """Retrieve the tracker for an existing session."""
        entry = self._sessions.get(session_id)
        return entry["tracker"] if entry else None

    def add_contribution(self, session_id: str, contributor: str, text: str) -> None:
        """Add a contribution to an existing session and update statistics."""
        entry = self._sessions.get(session_id)
        if entry is None:
            raise ValueError(f"Session {session_id!r} does not exist.")

        tracker: DiversityTracker = entry["tracker"]
        contribs: Dict[str, List[str]] = entry["contributions"]

        contribs[contributor].append(text)
        tracker.contributors.add(contributor)
        tracker.timestamps.append(time.time())
        tracker.history.append({"contributor": contributor, "text": text})

        # Recompute diversity score
        all_texts: List[str] = []
        for texts in contribs.values():
            all_texts.extend(texts)

        score = collaboration_diversity_score(dict(contribs))
        tracker.current_diversity = score
        tracker.trend.append(score)

        # Update profiles
        for cid, texts in contribs.items():
            entry["profiles"][cid] = self._analyzer.analyze(cid, texts)

    def get_current_stats(self, session_id: str) -> Dict[str, Any]:
        """Return current session statistics."""
        entry = self._sessions.get(session_id)
        if entry is None:
            raise ValueError(f"Session {session_id!r} does not exist.")

        tracker: DiversityTracker = entry["tracker"]
        profiles: Dict[str, ContributionProfile] = entry["profiles"]

        per_contributor = {}
        for cid, profile in profiles.items():
            per_contributor[cid] = {
                "unique_ratio": profile.unique_ratio,
                "topic_entropy": _entropy(
                    np.array(list(profile.topic_distribution.values()))
                ) if profile.topic_distribution else 0.0,
                "contributions_count": len(entry["contributions"][cid]),
            }

        return {
            "session_id": session_id,
            "current_diversity": tracker.current_diversity,
            "num_contributions": len(tracker.history),
            "num_contributors": len(tracker.contributors),
            "per_contributor": per_contributor,
        }

    def get_trend(self, session_id: str) -> Dict[str, Any]:
        """Return diversity trend data for a session.

        Includes the raw trend series, a moving average, and a flag for
        whether diminishing returns have been detected (diversity increase
        slowing below 1% over the last 5 contributions).
        """
        entry = self._sessions.get(session_id)
        if entry is None:
            raise ValueError(f"Session {session_id!r} does not exist.")

        tracker: DiversityTracker = entry["tracker"]
        trend = tracker.trend

        if len(trend) < 2:
            return {
                "trend": trend,
                "moving_average": trend[:],
                "diminishing_returns": False,
            }

        window = 5
        ma: List[float] = []
        for i in range(len(trend)):
            start = max(0, i - window + 1)
            ma.append(float(np.mean(trend[start : i + 1])))

        # Detect diminishing returns
        recent = trend[-window:]
        if len(recent) >= 2:
            deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
            avg_delta = float(np.mean(deltas))
            diminishing = avg_delta < 0.01
        else:
            diminishing = False

        return {
            "trend": trend,
            "moving_average": ma,
            "diminishing_returns": diminishing,
        }

    def list_sessions(self) -> List[str]:
        """Return all active session ids."""
        return list(self._sessions.keys())


# ---------------------------------------------------------------------------
# Module-level session manager singleton
# ---------------------------------------------------------------------------

_session_manager = SessionManager()


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def merge_diverse_contributions(contributions: Dict[str, List[str]]) -> List[str]:
    """Merge contributions from multiple users, maximizing diversity.

    Algorithm:
    1. Flatten all contributions and compute pairwise n-gram Jaccard
       similarity.
    2. Remove near-duplicate texts (Jaccard > threshold) across
       contributors, keeping the earliest occurrence.
    3. Greedily select the most diverse subset: start with the text that
       has the lowest average similarity to all others, then iteratively
       pick the text most dissimilar to the already-selected set.

    Parameters
    ----------
    contributions:
        Mapping of contributor id → list of text contributions.

    Returns
    -------
    Merged list of diverse contributions.
    """
    if not contributions:
        return []

    all_items: List[Tuple[str, str]] = []  # (contributor, text)
    for cid, texts in contributions.items():
        for t in texts:
            all_items.append((cid, t))

    if not all_items:
        return []

    n = len(all_items)
    ngram_sets = [_char_ngrams(item[1]) for item in all_items]

    # Pairwise similarity matrix
    sim_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            s = _jaccard(ngram_sets[i], ngram_sets[j])
            sim_matrix[i, j] = s
            sim_matrix[j, i] = s

    # Remove near-duplicates across different contributors
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            if all_items[i][0] != all_items[j][0] and sim_matrix[i, j] > _NEAR_DUPLICATE_THRESHOLD:
                keep[j] = False

    surviving = [i for i in range(n) if keep[i]]
    if not surviving:
        return [all_items[0][1]]

    # Greedy diversity selection
    avg_sims = np.array([
        float(np.mean([sim_matrix[i, j] for j in surviving if j != i]))
        if len(surviving) > 1 else 0.0
        for i in surviving
    ])
    first_idx = surviving[int(np.argmin(avg_sims))]
    selected = [first_idx]
    remaining = set(surviving) - {first_idx}

    while remaining:
        best_idx: int | None = None
        best_min_sim = float("inf")
        for candidate in remaining:
            min_sim_to_selected = min(sim_matrix[candidate, s] for s in selected)
            max_div = 1.0 - min_sim_to_selected
            if min_sim_to_selected < best_min_sim:
                best_min_sim = min_sim_to_selected
                best_idx = candidate
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return [all_items[i][1] for i in selected]


def identify_unique_contributions(
    contributors: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Identify each contributor's unique contributions.

    Uses MinHash + locality-sensitive hashing for approximate nearest
    neighbour detection.  A contribution is deemed *unique* if its
    maximum MinHash similarity with any contribution from another
    contributor is below the novelty threshold.

    Parameters
    ----------
    contributors:
        Mapping of contributor id → list of text contributions.

    Returns
    -------
    Mapping of contributor id → list of contributions with novelty above
    the threshold.
    """
    if not contributors:
        return {}

    mh = MinHashSignature()

    # Compute signatures for every contribution
    all_sigs: List[Tuple[str, int, np.ndarray]] = []
    for cid, texts in contributors.items():
        for idx, text in enumerate(texts):
            sig = mh.compute(text)
            all_sigs.append((cid, idx, sig))

    # LSH-style bucketing: divide signature into bands
    num_bands = 16
    rows_per_band = _DEFAULT_NUM_HASHES // num_bands
    buckets: Dict[int, Dict[str, List[Tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))

    for cid, idx, sig in all_sigs:
        for band in range(num_bands):
            start = band * rows_per_band
            end = start + rows_per_band
            band_hash = hashlib.md5(sig[start:end].tobytes()).hexdigest()
            bucket_key = hash((band, band_hash))
            buckets[bucket_key][cid].append((cid, idx))

    # For each contribution, find candidate neighbours from other contributors
    result: Dict[str, List[str]] = {cid: [] for cid in contributors}

    sig_lookup: Dict[Tuple[str, int], np.ndarray] = {
        (cid, idx): sig for cid, idx, sig in all_sigs
    }

    for cid, texts in contributors.items():
        for idx, text in enumerate(texts):
            my_sig = sig_lookup[(cid, idx)]
            candidate_sigs: List[np.ndarray] = []

            # Gather candidates from LSH buckets
            for band in range(num_bands):
                start = band * rows_per_band
                end = start + rows_per_band
                band_hash = hashlib.md5(my_sig[start:end].tobytes()).hexdigest()
                bucket_key = hash((band, band_hash))
                for other_cid, members in buckets[bucket_key].items():
                    if other_cid != cid:
                        for oc, oi in members:
                            candidate_sigs.append(sig_lookup[(oc, oi)])

            # Check if any candidate is too similar
            is_unique = True
            seen: Set[int] = set()
            for csig in candidate_sigs:
                sig_id = id(csig)
                if sig_id in seen:
                    continue
                seen.add(sig_id)
                sim = MinHashSignature.similarity(my_sig, csig)
                if sim > _NOVELTY_THRESHOLD:
                    is_unique = False
                    break

            # Fallback: brute-force check against all other contributors
            if is_unique and not candidate_sigs:
                for other_cid, other_texts in contributors.items():
                    if other_cid == cid:
                        continue
                    for oi, _ in enumerate(other_texts):
                        other_sig = sig_lookup[(other_cid, oi)]
                        sim = MinHashSignature.similarity(my_sig, other_sig)
                        if sim > _NOVELTY_THRESHOLD:
                            is_unique = False
                            break
                    if not is_unique:
                        break

            if is_unique:
                result[cid].append(text)

    return result


def collaboration_diversity_score(team_outputs: Dict[str, List[str]]) -> float:
    """Score how diverse a team's combined output is.

    The final score is a weighted combination of four factors:
    1. **Within-member diversity** — average intra-contributor pairwise
       distance (via n-gram Jaccard).
    2. **Between-member diversity** — average inter-contributor pairwise
       distance.
    3. **Topic coverage breadth** — entropy of the aggregate topic
       distribution.
    4. **Style variation** — variance of style signatures across members.

    Parameters
    ----------
    team_outputs:
        Mapping of contributor id → list of text contributions.

    Returns
    -------
    A diversity score in ``[0, 1]``.
    """
    if not team_outputs:
        return 0.0

    members = list(team_outputs.keys())
    if len(members) == 0:
        return 0.0

    # -- 1. Within-member diversity ----------------------------------------
    within_scores: List[float] = []
    for cid in members:
        texts = team_outputs[cid]
        if len(texts) < 2:
            within_scores.append(0.0)
            continue
        ngrams = [_char_ngrams(t) for t in texts]
        dists = []
        for i in range(len(ngrams)):
            for j in range(i + 1, len(ngrams)):
                dists.append(1.0 - _jaccard(ngrams[i], ngrams[j]))
        within_scores.append(float(np.mean(dists)) if dists else 0.0)
    avg_within = float(np.mean(within_scores)) if within_scores else 0.0

    # -- 2. Between-member diversity ---------------------------------------
    if len(members) < 2:
        avg_between = 0.0
    else:
        member_ngrams: Dict[str, Set[str]] = {}
        for cid in members:
            combined = " ".join(team_outputs[cid])
            member_ngrams[cid] = _char_ngrams(combined)
        between_dists: List[float] = []
        for m1, m2 in combinations(members, 2):
            between_dists.append(1.0 - _jaccard(member_ngrams[m1], member_ngrams[m2]))
        avg_between = float(np.mean(between_dists))

    # -- 3. Topic coverage breadth -----------------------------------------
    all_texts: List[str] = []
    for texts in team_outputs.values():
        all_texts.extend(texts)
    topic_dist = ContributionAnalyzer._topic_distribution(all_texts)
    topic_arr = np.array(list(topic_dist.values()))
    max_entropy = np.log(len(topic_arr)) if len(topic_arr) > 1 else 1.0
    topic_entropy = _entropy(topic_arr) / max_entropy if max_entropy > 0 else 0.0

    # -- 4. Style variation ------------------------------------------------
    analyzer = ContributionAnalyzer()
    style_vecs: List[np.ndarray] = []
    for cid in members:
        combined = " ".join(team_outputs[cid])
        style = analyzer._compute_style(combined)
        style_vecs.append(np.array(list(style.values())))

    if len(style_vecs) >= 2:
        style_matrix = np.stack(style_vecs)
        col_stds = np.std(style_matrix, axis=0)
        style_variation = float(np.mean(col_stds))
        style_variation = min(style_variation, 1.0)
    else:
        style_variation = 0.0

    # -- Weighted combination ----------------------------------------------
    score = (
        _DIVERSITY_WEIGHTS["within"] * avg_within
        + _DIVERSITY_WEIGHTS["between"] * avg_between
        + _DIVERSITY_WEIGHTS["coverage"] * topic_entropy
        + _DIVERSITY_WEIGHTS["style"] * style_variation
    )
    return float(np.clip(score, 0.0, 1.0))


def assign_diverse_tasks(
    tasks: List[str], team_members: List[str]
) -> Assignment:
    """Assign tasks to team members to maximize output diversity.

    Uses task similarity analysis to build a pairwise similarity matrix,
    then delegates to :class:`CollaborationOptimizer` for greedy
    assignment.  Also computes coverage and balance scores.

    Parameters
    ----------
    tasks:
        List of task descriptions.
    team_members:
        List of team member identifiers.

    Returns
    -------
    An :class:`Assignment` with the optimal task allocation.
    """
    if not tasks or not team_members:
        return Assignment(
            task_assignments={m: [] for m in team_members},
            diversity_score=0.0,
            coverage_score=0.0,
            balance_score=0.0,
            rationale="No tasks or team members provided.",
        )

    optimizer = CollaborationOptimizer()
    task_assignments = optimizer.optimize_assignment(tasks, team_members)

    # Compute diversity score: average dissimilarity between tasks within each member
    all_ngrams = [_word_ngrams(t) for t in tasks]
    task_idx: Dict[str, int] = {}
    for i, t in enumerate(tasks):
        task_idx[t] = i

    intra_dists: List[float] = []
    for member, assigned in task_assignments.items():
        if len(assigned) < 2:
            continue
        indices = [task_idx.get(t, 0) for t in assigned]
        for a, b in combinations(indices, 2):
            intra_dists.append(1.0 - _jaccard(all_ngrams[a], all_ngrams[b]))

    diversity_score = float(np.mean(intra_dists)) if intra_dists else 0.0

    # Coverage: fraction of tasks assigned
    total_assigned = sum(len(v) for v in task_assignments.values())
    coverage_score = total_assigned / len(tasks) if tasks else 0.0

    # Balance: 1 - normalised std of load distribution
    loads = [len(v) for v in task_assignments.values()]
    if len(loads) > 1 and np.mean(loads) > 0:
        balance_score = 1.0 - float(np.std(loads) / np.mean(loads))
        balance_score = max(balance_score, 0.0)
    else:
        balance_score = 1.0

    ideal_load = len(tasks) / len(team_members)
    rationale_parts = [
        f"Assigned {len(tasks)} tasks to {len(team_members)} members.",
        f"Ideal load per member: {ideal_load:.1f}.",
        f"Diversity score: {diversity_score:.3f}.",
        f"Balance score: {balance_score:.3f}.",
    ]
    if diversity_score < 0.3:
        rationale_parts.append(
            "Tasks are fairly similar; consider rephrasing for clearer differentiation."
        )

    return Assignment(
        task_assignments=task_assignments,
        diversity_score=round(diversity_score, 4),
        coverage_score=round(coverage_score, 4),
        balance_score=round(balance_score, 4),
        rationale=" ".join(rationale_parts),
    )


def real_time_diversity_tracker(session_id: str) -> DiversityTracker:
    """Create (or retrieve) a real-time diversity tracker for a session.

    The returned :class:`DiversityTracker` is backed by the module-level
    :class:`SessionManager`. Callers interact with the tracker through
    the session manager's ``add_contribution``, ``get_current_stats``,
    and ``get_trend`` methods.

    Parameters
    ----------
    session_id:
        Unique identifier for the collaboration session.

    Returns
    -------
    A :class:`DiversityTracker` instance associated with the session.
    """
    existing = _session_manager.get_session(session_id)
    if existing is not None:
        return existing
    return _session_manager.create_session(session_id)


# ---------------------------------------------------------------------------
# Convenience wrappers for the session manager
# ---------------------------------------------------------------------------


def add_session_contribution(session_id: str, contributor: str, text: str) -> None:
    """Add a contribution to a tracked session."""
    _session_manager.add_contribution(session_id, contributor, text)


def get_session_stats(session_id: str) -> Dict[str, Any]:
    """Return current statistics for a tracked session."""
    return _session_manager.get_current_stats(session_id)


def get_session_trend(session_id: str) -> Dict[str, Any]:
    """Return trend data for a tracked session."""
    return _session_manager.get_trend(session_id)
