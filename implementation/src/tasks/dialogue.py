"""
Dialogue generation task domain for the Diversity Decoding Arena.

Implements evaluation of diverse dialogue generation across open-domain
conversation, task-oriented dialogue, debate, interview, roleplay, and
persona-based scenarios.  Provides rich built-in prompt datasets and
fine-grained evaluation metrics for coherence, engagement, persona
consistency, topic diversity, response diversity, turn-taking quality,
informativeness, naturalness, and contradiction avoidance.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from itertools import combinations
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
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import entropy as scipy_entropy

from src.tasks.base import (
    ConstraintType,
    GenerationTask,
    PromptDataset,
    TaskConfig,
    TaskConstraint,
    TaskEvaluator,
    TaskPrompt,
    _TASK_REGISTRY,
)
from src.types import TaskDomain

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DialogueType(Enum):
    """Supported dialogue scenario types."""

    OPEN_DOMAIN = auto()
    TASK_ORIENTED = auto()
    DEBATE = auto()
    INTERVIEW = auto()
    ROLEPLAY = auto()
    PERSONA_BASED = auto()
    MULTI_TURN = auto()
    NEGOTIATION = auto()
    COUNSELING = auto()
    TUTORIAL = auto()

    def __repr__(self) -> str:
        return f"DialogueType.{self.name}"


class DialogueAct(Enum):
    """Speech act categories for utterance classification."""

    GREETING = auto()
    FAREWELL = auto()
    QUESTION = auto()
    ANSWER = auto()
    STATEMENT = auto()
    REQUEST = auto()
    OFFER = auto()
    AGREEMENT = auto()
    DISAGREEMENT = auto()
    ACKNOWLEDGMENT = auto()
    CLARIFICATION = auto()
    APOLOGY = auto()
    THANKS = auto()
    SUGGESTION = auto()
    COMMAND = auto()
    EXCLAMATION = auto()
    HEDGING = auto()
    BACKCHANNEL = auto()
    ELABORATION = auto()
    TOPIC_SHIFT = auto()

    def __repr__(self) -> str:
        return f"DialogueAct.{self.name}"


class ResponseStyle(Enum):
    """Target response style for dialogue generation."""

    FORMAL = auto()
    CASUAL = auto()
    EMPATHETIC = auto()
    ASSERTIVE = auto()
    HUMOROUS = auto()
    ACADEMIC = auto()
    PROFESSIONAL = auto()
    FRIENDLY = auto()

    def __repr__(self) -> str:
        return f"ResponseStyle.{self.name}"


# ---------------------------------------------------------------------------
# Dialogue act pattern banks
# ---------------------------------------------------------------------------

_GREETING_PATTERNS: List[str] = [
    r"\b(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening))\b",
    r"\bhow\s+are\s+you\b",
    r"\bnice\s+to\s+(?:meet|see)\b",
    r"\bwelcome\b",
    r"\bwhat'?s\s+up\b",
]

_FAREWELL_PATTERNS: List[str] = [
    r"\b(?:goodbye|bye|farewell|see\s+you|take\s+care)\b",
    r"\b(?:good\s+night|until\s+next\s+time|catch\s+you\s+later)\b",
    r"\bhave\s+a\s+(?:good|great|nice)\b",
    r"\btalk\s+(?:to\s+you\s+)?(?:soon|later)\b",
]

_QUESTION_PATTERNS: List[str] = [
    r"\?$",
    r"^(?:what|who|where|when|why|how|which|whose|whom)\b",
    r"\b(?:do|does|did|can|could|would|will|shall|should|is|are|was|were|have|has)\s+\w+\s+\w+\?",
    r"\bdo\s+you\s+(?:think|know|believe|feel|want)\b",
    r"\bcould\s+you\b",
    r"\bwould\s+you\b",
    r"\bcan\s+you\b",
]

_AGREEMENT_PATTERNS: List[str] = [
    r"\b(?:yes|yeah|yep|exactly|absolutely|definitely|certainly|indeed)\b",
    r"\bi\s+agree\b",
    r"\bthat'?s\s+(?:right|true|correct)\b",
    r"\bgood\s+point\b",
    r"\byou'?re\s+right\b",
    r"\bof\s+course\b",
    r"\bno\s+doubt\b",
]

_DISAGREEMENT_PATTERNS: List[str] = [
    r"\b(?:no|nah|nope)\b",
    r"\bi\s+disagree\b",
    r"\bthat'?s\s+not\b",
    r"\bi\s+don'?t\s+(?:think|agree|believe)\b",
    r"\bon\s+the\s+contrary\b",
    r"\bbut\s+(?:actually|really)\b",
    r"\bnot\s+(?:quite|exactly|really|necessarily)\b",
]

_ACKNOWLEDGMENT_PATTERNS: List[str] = [
    r"\b(?:okay|ok|alright|sure|right|got\s+it|understood)\b",
    r"\bi\s+see\b",
    r"\bmakes\s+sense\b",
    r"\bfair\s+enough\b",
    r"\bthat\s+makes\s+sense\b",
    r"\buh[- ]?huh\b",
    r"\bmm[- ]?hmm\b",
]

_APOLOGY_PATTERNS: List[str] = [
    r"\b(?:sorry|apologi[sz]e|forgive|pardon|my\s+bad)\b",
    r"\bi'?m\s+(?:so\s+)?sorry\b",
    r"\bexcuse\s+me\b",
]

_THANKS_PATTERNS: List[str] = [
    r"\b(?:thanks?|thank\s+you|grateful|appreciate)\b",
    r"\bthanks?\s+(?:a\s+lot|so\s+much|very\s+much)\b",
    r"\bmuch\s+appreciated\b",
]

_REQUEST_PATTERNS: List[str] = [
    r"\b(?:please|kindly)\b",
    r"\bcould\s+you\s+(?:please\s+)?(?:\w+)\b",
    r"\bwould\s+you\s+mind\b",
    r"\bi\s+(?:need|want|would\s+like)\b",
    r"\bcan\s+you\s+help\b",
]

_SUGGESTION_PATTERNS: List[str] = [
    r"\bmaybe\s+(?:we|you)\s+(?:should|could|can)\b",
    r"\bhow\s+about\b",
    r"\bwhat\s+if\b",
    r"\bwhy\s+(?:don'?t|not)\b",
    r"\bi\s+suggest\b",
    r"\bhave\s+you\s+(?:tried|considered)\b",
    r"\byou\s+(?:might|could)\s+(?:want\s+to|try)\b",
]

_HEDGING_PATTERNS: List[str] = [
    r"\b(?:maybe|perhaps|possibly|probably|might|could\s+be)\b",
    r"\bi\s+(?:think|guess|suppose|believe|feel\s+like)\b",
    r"\bit\s+seems\s+(?:like|that)\b",
    r"\bkind\s+of\b",
    r"\bsort\s+of\b",
    r"\bin\s+my\s+opinion\b",
    r"\bnot\s+sure\s+(?:if|but|whether)\b",
]

_BACKCHANNEL_PATTERNS: List[str] = [
    r"\b(?:uh[- ]?huh|mm[- ]?hmm|right|yeah|okay|mhm)\b",
    r"\bi\s+see\b",
    r"\bgo\s+on\b",
    r"\breally\??\b",
    r"\binteresting\b",
    r"\bwow\b",
]

_CONVERSATIONAL_FILLERS: Set[str] = {
    "um", "uh", "like", "you know", "i mean", "well", "so",
    "actually", "basically", "honestly", "literally", "right",
    "anyway", "anyhow", "kind of", "sort of",
}

_FORMAL_MARKERS: List[str] = [
    r"\bfurthermore\b", r"\bmoreover\b", r"\bnevertheless\b",
    r"\bhowever\b", r"\bconsequently\b", r"\bthus\b", r"\bhence\b",
    r"\btherefore\b", r"\bnotwithstanding\b", r"\baccordingly\b",
    r"\bregarding\b", r"\bpertaining\b", r"\bwith\s+respect\s+to\b",
]

_CASUAL_MARKERS: List[str] = [
    r"\bgonna\b", r"\bwanna\b", r"\bkinda\b", r"\bsorta\b",
    r"\byeah\b", r"\bnah\b", r"\blol\b", r"\bomg\b",
    r"\bbtw\b", r"\bimo\b", r"\bdude\b", r"\bstuff\b",
    r"\bcool\b", r"\bawesome\b", r"\btotally\b",
]

_EMOTIONAL_MARKERS: Dict[str, List[str]] = {
    "joy": ["happy", "glad", "excited", "wonderful", "great", "fantastic",
            "delighted", "thrilled", "pleased", "love", "enjoy", "amazing"],
    "sadness": ["sad", "sorry", "unfortunately", "miss", "upset", "disappointed",
                "heartbroken", "miserable", "depressed", "gloomy", "unhappy"],
    "anger": ["angry", "frustrated", "annoyed", "furious", "irritated",
              "outraged", "mad", "fed up", "livid", "hostile"],
    "fear": ["afraid", "worried", "scared", "anxious", "nervous", "terrified",
             "uneasy", "concerned", "alarmed", "frightened"],
    "surprise": ["surprised", "shocked", "amazed", "wow", "unexpected",
                 "astonished", "stunned", "incredible", "unbelievable"],
    "empathy": ["understand", "feel for you", "that must be", "i can imagine",
                "sorry to hear", "must be hard", "sympathize", "compassion"],
}

_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "technology": ["computer", "software", "app", "internet", "digital",
                   "phone", "device", "tech", "online", "data", "ai",
                   "robot", "code", "program", "algorithm", "machine"],
    "health": ["health", "doctor", "medical", "exercise", "diet", "fitness",
               "wellness", "symptom", "treatment", "therapy", "medicine",
               "hospital", "disease", "illness", "mental", "physical"],
    "food": ["food", "cook", "recipe", "restaurant", "meal", "eat",
             "dinner", "lunch", "breakfast", "taste", "delicious",
             "kitchen", "chef", "ingredient", "cuisine", "dish"],
    "travel": ["travel", "trip", "vacation", "flight", "hotel", "country",
               "city", "visit", "explore", "adventure", "destination",
               "tourist", "journey", "passport", "airport", "abroad"],
    "work": ["work", "job", "career", "office", "meeting", "project",
             "deadline", "boss", "colleague", "promotion", "salary",
             "resume", "interview", "company", "business", "profession"],
    "education": ["school", "university", "class", "study", "learn",
                  "teacher", "student", "course", "exam", "homework",
                  "degree", "education", "academic", "lecture", "grade"],
    "entertainment": ["movie", "music", "game", "show", "book", "read",
                      "watch", "play", "concert", "theater", "film",
                      "series", "podcast", "song", "artist", "perform"],
    "relationships": ["friend", "family", "love", "date", "partner",
                      "relationship", "marriage", "parent", "child",
                      "sibling", "couple", "wedding", "together", "bond"],
    "sports": ["sport", "team", "game", "match", "play", "win", "score",
               "player", "coach", "season", "championship", "league",
               "training", "athlete", "competition", "tournament"],
    "nature": ["nature", "animal", "plant", "tree", "forest", "ocean",
               "mountain", "river", "weather", "climate", "environment",
               "wildlife", "garden", "flower", "landscape", "outdoor"],
    "philosophy": ["meaning", "purpose", "truth", "reality", "consciousness",
                   "existence", "moral", "ethics", "value", "belief",
                   "freedom", "justice", "wisdom", "knowledge", "reason"],
    "politics": ["politics", "government", "policy", "law", "vote",
                 "election", "democracy", "rights", "freedom", "citizen",
                 "congress", "president", "legislation", "party", "debate"],
}

# ---------------------------------------------------------------------------
# Dataclasses — dialogue structures
# ---------------------------------------------------------------------------


@dataclass
class DialogueTurn:
    """A single turn in a dialogue exchange.

    Parameters
    ----------
    speaker : str
        The name or identifier of the speaker.
    utterance : str
        The text content of the utterance.
    turn_number : int
        The ordinal position of this turn in the conversation.
    metadata : Dict[str, Any]
        Arbitrary key/value metadata (dialogue act, emotion, etc.).
    """

    speaker: str = ""
    utterance: str = ""
    turn_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        """Number of whitespace-delimited words in the utterance."""
        return len(self.utterance.split())

    @property
    def char_count(self) -> int:
        """Number of characters in the utterance."""
        return len(self.utterance)

    def has_question(self) -> bool:
        """Return ``True`` if the utterance contains a question."""
        return "?" in self.utterance

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "speaker": self.speaker,
            "utterance": self.utterance,
            "turn_number": self.turn_number,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueTurn":
        """Reconstruct from a dictionary."""
        return cls(
            speaker=data.get("speaker", ""),
            utterance=data.get("utterance", ""),
            turn_number=data.get("turn_number", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DialogueContext:
    """Context surrounding a dialogue exchange.

    Parameters
    ----------
    turns : List[DialogueTurn]
        Previous conversation turns providing context.
    system_prompt : str
        System-level instructions for the dialogue.
    scenario : str
        A description of the scenario or situation.
    participants : List[str]
        Names or identifiers of the participants.
    """

    turns: List[DialogueTurn] = field(default_factory=list)
    system_prompt: str = ""
    scenario: str = ""
    participants: List[str] = field(default_factory=list)

    @property
    def num_turns(self) -> int:
        """Number of turns in the context."""
        return len(self.turns)

    @property
    def num_participants(self) -> int:
        """Number of participants."""
        return len(self.participants)

    def last_speaker(self) -> str:
        """Return the speaker of the most recent turn, or empty string."""
        if self.turns:
            return self.turns[-1].speaker
        return ""

    def last_utterance(self) -> str:
        """Return the text of the most recent turn, or empty string."""
        if self.turns:
            return self.turns[-1].utterance
        return ""

    def get_speaker_turns(self, speaker: str) -> List[DialogueTurn]:
        """Return all turns by a specific speaker."""
        return [t for t in self.turns if t.speaker == speaker]

    def to_transcript(self, separator: str = "\n") -> str:
        """Format all turns into a readable transcript."""
        lines = []
        for turn in self.turns:
            lines.append(f"{turn.speaker}: {turn.utterance}")
        return separator.join(lines)

    def add_turn(self, speaker: str, utterance: str, **kwargs: Any) -> None:
        """Append a new turn to the context."""
        turn = DialogueTurn(
            speaker=speaker,
            utterance=utterance,
            turn_number=len(self.turns),
            metadata=kwargs,
        )
        self.turns.append(turn)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "turns": [t.to_dict() for t in self.turns],
            "system_prompt": self.system_prompt,
            "scenario": self.scenario,
            "participants": self.participants,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueContext":
        """Reconstruct from a dictionary."""
        turns = [DialogueTurn.from_dict(t) for t in data.get("turns", [])]
        return cls(
            turns=turns,
            system_prompt=data.get("system_prompt", ""),
            scenario=data.get("scenario", ""),
            participants=data.get("participants", []),
        )


# ---------------------------------------------------------------------------
# DialogueConfig
# ---------------------------------------------------------------------------


@dataclass
class DialogueConfig(TaskConfig):
    """Configuration for a dialogue generation evaluation run.

    Extends :class:`TaskConfig` with dialogue-specific settings including
    turn limits, persona definitions, topic lists, and response styles.
    """

    max_turns: int = 10
    min_turns: int = 2
    dialogue_type: DialogueType = DialogueType.OPEN_DOMAIN
    personas: List[Dict[str, str]] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    response_styles: List[ResponseStyle] = field(default_factory=list)
    require_turn_markers: bool = True
    allow_multi_speaker: bool = True
    max_speakers: int = 4
    min_utterance_words: int = 3
    max_utterance_words: int = 200
    system_prompt_template: str = ""
    enable_persona_consistency: bool = True
    enable_coherence_check: bool = True
    topic_adherence_threshold: float = 0.3
    diversity_weight: float = 0.5
    quality_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.max_turns < self.min_turns:
            raise ValueError("max_turns must be >= min_turns")
        if self.min_turns < 1:
            raise ValueError("min_turns must be >= 1")
        if self.max_speakers < 2:
            raise ValueError("max_speakers must be >= 2")
        if self.min_utterance_words < 1:
            raise ValueError("min_utterance_words must be >= 1")
        if not 0.0 <= self.diversity_weight <= 1.0:
            raise ValueError("diversity_weight must be in [0, 1]")
        if not 0.0 <= self.quality_weight <= 1.0:
            raise ValueError("quality_weight must be in [0, 1]")

    def validate(self) -> List[str]:
        """Return a list of validation errors (empty if valid)."""
        errors = super().validate()
        if self.max_turns < self.min_turns:
            errors.append("max_turns must be >= min_turns")
        if self.min_turns < 1:
            errors.append("min_turns must be >= 1")
        if self.max_speakers < 2:
            errors.append("max_speakers must be >= 2")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        base = super().to_dict()
        base.update({
            "max_turns": self.max_turns,
            "min_turns": self.min_turns,
            "dialogue_type": self.dialogue_type.name,
            "personas": self.personas,
            "topics": self.topics,
            "response_styles": [s.name for s in self.response_styles],
            "require_turn_markers": self.require_turn_markers,
            "allow_multi_speaker": self.allow_multi_speaker,
            "max_speakers": self.max_speakers,
            "min_utterance_words": self.min_utterance_words,
            "max_utterance_words": self.max_utterance_words,
            "system_prompt_template": self.system_prompt_template,
            "enable_persona_consistency": self.enable_persona_consistency,
            "enable_coherence_check": self.enable_coherence_check,
            "topic_adherence_threshold": self.topic_adherence_threshold,
            "diversity_weight": self.diversity_weight,
            "quality_weight": self.quality_weight,
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueConfig":
        """Reconstruct from a dictionary."""
        constraints = [
            TaskConstraint.from_dict(c) for c in data.get("constraints", [])
        ]
        response_styles = [
            ResponseStyle[s] for s in data.get("response_styles", [])
        ]
        dtype = data.get("dialogue_type", "OPEN_DOMAIN")
        return cls(
            name=data.get("name", "dialogue"),
            domain=TaskDomain[data.get("domain", "DIALOGUE")],
            num_prompts=data.get("num_prompts", 100),
            max_length=data.get("max_length", 1024),
            min_length=data.get("min_length", 20),
            temperature=data.get("temperature", 1.0),
            constraints=constraints,
            evaluation_metrics=data.get(
                "evaluation_metrics",
                ["coherence", "engagement", "diversity"],
            ),
            prompt_template=data.get("prompt_template", "{text}"),
            seed=data.get("seed", 42),
            max_turns=data.get("max_turns", 10),
            min_turns=data.get("min_turns", 2),
            dialogue_type=DialogueType[dtype],
            personas=data.get("personas", []),
            topics=data.get("topics", []),
            response_styles=response_styles,
            require_turn_markers=data.get("require_turn_markers", True),
            allow_multi_speaker=data.get("allow_multi_speaker", True),
            max_speakers=data.get("max_speakers", 4),
            min_utterance_words=data.get("min_utterance_words", 3),
            max_utterance_words=data.get("max_utterance_words", 200),
            system_prompt_template=data.get("system_prompt_template", ""),
            enable_persona_consistency=data.get("enable_persona_consistency", True),
            enable_coherence_check=data.get("enable_coherence_check", True),
            topic_adherence_threshold=data.get("topic_adherence_threshold", 0.3),
            diversity_weight=data.get("diversity_weight", 0.5),
            quality_weight=data.get("quality_weight", 0.5),
        )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer (strips punctuation)."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def _sentences(text: str) -> List[str]:
    """Split *text* into sentences."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _sigmoid(x: float, midpoint: float = 0.0, steepness: float = 1.0) -> float:
    """Numerically-stable sigmoid mapping to [0, 1]."""
    z = steepness * (x - midpoint)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _type_token_ratio(tokens: List[str]) -> float:
    """Compute type-token ratio for lexical diversity."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from a list of tokens."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _count_pattern_hits(text: str, patterns: List[str]) -> int:
    """Count the total number of regex pattern matches in text."""
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text, re.IGNORECASE))
    return total


def _cosine_sim_from_counters(
    counter_a: Counter, counter_b: Counter
) -> float:
    """Compute cosine similarity between two Counter objects."""
    all_keys = set(counter_a.keys()) | set(counter_b.keys())
    if not all_keys:
        return 0.0
    vec_a = np.array([counter_a.get(k, 0) for k in all_keys], dtype=float)
    vec_b = np.array([counter_b.get(k, 0) for k in all_keys], dtype=float)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _normalized_entropy(counts: List[int]) -> float:
    """Compute normalized Shannon entropy of a distribution."""
    if not counts or sum(counts) == 0:
        return 0.0
    total = sum(counts)
    probs = np.array([c / total for c in counts if c > 0], dtype=float)
    if len(probs) <= 1:
        return 0.0
    raw_entropy = float(scipy_entropy(probs, base=2))
    max_entropy = math.log2(len(probs))
    if max_entropy == 0.0:
        return 0.0
    return raw_entropy / max_entropy


# ---------------------------------------------------------------------------
# Helper functions — dialogue analysis
# ---------------------------------------------------------------------------


def extract_dialogue_acts(utterance: str) -> List[DialogueAct]:
    """Classify the dialogue acts present in an utterance.

    Uses pattern matching to identify one or more dialogue acts in the
    given utterance text.  Returns a list of detected acts in priority
    order.

    Parameters
    ----------
    utterance : str
        The text of a single dialogue utterance.

    Returns
    -------
    List[DialogueAct]
        Detected dialogue acts, may contain multiple if utterance is complex.
    """
    acts: List[DialogueAct] = []
    text = utterance.strip()
    text_lower = text.lower()

    if _count_pattern_hits(text_lower, _GREETING_PATTERNS) > 0:
        acts.append(DialogueAct.GREETING)

    if _count_pattern_hits(text_lower, _FAREWELL_PATTERNS) > 0:
        acts.append(DialogueAct.FAREWELL)

    if _count_pattern_hits(text_lower, _APOLOGY_PATTERNS) > 0:
        acts.append(DialogueAct.APOLOGY)

    if _count_pattern_hits(text_lower, _THANKS_PATTERNS) > 0:
        acts.append(DialogueAct.THANKS)

    if _count_pattern_hits(text_lower, _AGREEMENT_PATTERNS) > 0:
        acts.append(DialogueAct.AGREEMENT)

    if _count_pattern_hits(text_lower, _DISAGREEMENT_PATTERNS) > 0:
        acts.append(DialogueAct.DISAGREEMENT)

    if _count_pattern_hits(text_lower, _ACKNOWLEDGMENT_PATTERNS) > 0:
        acts.append(DialogueAct.ACKNOWLEDGMENT)

    if _count_pattern_hits(text_lower, _REQUEST_PATTERNS) > 0:
        acts.append(DialogueAct.REQUEST)

    if _count_pattern_hits(text_lower, _SUGGESTION_PATTERNS) > 0:
        acts.append(DialogueAct.SUGGESTION)

    if _count_pattern_hits(text_lower, _HEDGING_PATTERNS) > 0:
        acts.append(DialogueAct.HEDGING)

    if _count_pattern_hits(text_lower, _BACKCHANNEL_PATTERNS) > 0:
        acts.append(DialogueAct.BACKCHANNEL)

    # Question detection (check after others as questions can co-occur)
    if _count_pattern_hits(text, _QUESTION_PATTERNS) > 0:
        acts.append(DialogueAct.QUESTION)

    # Exclamation — if ending with ! and not already classified
    if text.endswith("!") and DialogueAct.EXCLAMATION not in acts:
        acts.append(DialogueAct.EXCLAMATION)

    # Default to STATEMENT if nothing else detected
    if not acts:
        acts.append(DialogueAct.STATEMENT)

    return acts


def compute_topic_shift_score(
    utterance_a: str, utterance_b: str
) -> float:
    """Compute a topic shift score between two consecutive utterances.

    A score of 0.0 means no topic shift (high overlap), and 1.0 means
    a complete topic change (no overlap).

    Parameters
    ----------
    utterance_a : str
        The first utterance.
    utterance_b : str
        The second utterance.

    Returns
    -------
    float
        Topic shift score in [0, 1].
    """
    tokens_a = set(_tokenize(utterance_a))
    tokens_b = set(_tokenize(utterance_b))

    # Remove very common function words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "and", "but", "or", "nor", "not", "so", "yet",
        "both", "either", "neither", "each", "every", "all", "any",
        "few", "more", "most", "other", "some", "such", "no",
        "only", "own", "same", "than", "too", "very", "just",
        "i", "me", "my", "mine", "we", "us", "our", "ours",
        "you", "your", "yours", "he", "him", "his", "she", "her",
        "hers", "it", "its", "they", "them", "their", "theirs",
        "this", "that", "these", "those", "what", "which", "who",
        "whom", "whose", "where", "when", "why", "how",
    }

    content_a = tokens_a - stop_words
    content_b = tokens_b - stop_words

    if not content_a and not content_b:
        return 0.0
    if not content_a or not content_b:
        return 1.0

    overlap = len(content_a & content_b)
    union = len(content_a | content_b)

    similarity = overlap / union if union > 0 else 0.0
    return 1.0 - similarity


def detect_conversation_patterns(
    turns: List[DialogueTurn],
) -> Dict[str, Any]:
    """Detect structural patterns in a sequence of dialogue turns.

    Analyses turn-taking patterns, question-answer adjacency pairs,
    topic flow, and speaker participation balance.

    Parameters
    ----------
    turns : List[DialogueTurn]
        The dialogue turns to analyse.

    Returns
    -------
    Dict[str, Any]
        Dictionary with pattern metrics including ``qa_pairs``,
        ``topic_shifts``, ``speaker_balance``, ``avg_turn_length``,
        ``turn_length_variance``, ``consecutive_same_speaker``, and
        ``act_distribution``.
    """
    if not turns:
        return {
            "qa_pairs": 0,
            "topic_shifts": 0,
            "speaker_balance": 0.0,
            "avg_turn_length": 0.0,
            "turn_length_variance": 0.0,
            "consecutive_same_speaker": 0,
            "act_distribution": {},
        }

    # Question-answer pairs
    qa_pairs = 0
    for i in range(len(turns) - 1):
        acts_current = extract_dialogue_acts(turns[i].utterance)
        acts_next = extract_dialogue_acts(turns[i + 1].utterance)
        if DialogueAct.QUESTION in acts_current and (
            DialogueAct.ANSWER in acts_next
            or DialogueAct.STATEMENT in acts_next
        ):
            qa_pairs += 1

    # Topic shifts between consecutive turns
    topic_shifts = 0
    for i in range(len(turns) - 1):
        shift = compute_topic_shift_score(
            turns[i].utterance, turns[i + 1].utterance
        )
        if shift > 0.7:
            topic_shifts += 1

    # Speaker participation balance
    speaker_counts: Counter = Counter()
    for turn in turns:
        speaker_counts[turn.speaker] += 1
    total_turns_count = len(turns)
    if len(speaker_counts) > 1:
        counts = list(speaker_counts.values())
        ideal = total_turns_count / len(speaker_counts)
        deviations = [abs(c - ideal) / ideal for c in counts]
        speaker_balance = 1.0 - min(1.0, sum(deviations) / len(deviations))
    else:
        speaker_balance = 0.0

    # Turn length statistics
    lengths = [turn.word_count for turn in turns]
    avg_turn_length = float(np.mean(lengths)) if lengths else 0.0
    turn_length_variance = float(np.var(lengths)) if lengths else 0.0

    # Consecutive same speaker
    consecutive = 0
    for i in range(len(turns) - 1):
        if turns[i].speaker == turns[i + 1].speaker:
            consecutive += 1

    # Dialogue act distribution
    all_acts: Counter = Counter()
    for turn in turns:
        acts = extract_dialogue_acts(turn.utterance)
        for act in acts:
            all_acts[act.name] += 1

    return {
        "qa_pairs": qa_pairs,
        "topic_shifts": topic_shifts,
        "speaker_balance": speaker_balance,
        "avg_turn_length": avg_turn_length,
        "turn_length_variance": turn_length_variance,
        "consecutive_same_speaker": consecutive,
        "act_distribution": dict(all_acts),
    }


def compute_lexical_alignment(
    speaker_a_text: str, speaker_b_text: str
) -> float:
    """Compute lexical alignment between two speakers' combined text.

    Lexical alignment measures how much speakers converge on vocabulary
    choices.  Higher alignment may indicate rapport or accommodation.

    Parameters
    ----------
    speaker_a_text : str
        All text from speaker A concatenated.
    speaker_b_text : str
        All text from speaker B concatenated.

    Returns
    -------
    float
        Alignment score in [0, 1].  Higher means more alignment.
    """
    tokens_a = _tokenize(speaker_a_text)
    tokens_b = _tokenize(speaker_b_text)

    if not tokens_a or not tokens_b:
        return 0.0

    # Compute bigram overlap
    bigrams_a = set(_ngrams(tokens_a, 2))
    bigrams_b = set(_ngrams(tokens_b, 2))

    if not bigrams_a and not bigrams_b:
        # Fall back to unigram overlap
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        return _jaccard_similarity(set_a, set_b)

    bigram_overlap = _jaccard_similarity(
        {" ".join(bg) for bg in bigrams_a},
        {" ".join(bg) for bg in bigrams_b},
    )

    # Unigram frequency correlation
    freq_a = Counter(tokens_a)
    freq_b = Counter(tokens_b)
    freq_sim = _cosine_sim_from_counters(freq_a, freq_b)

    # Weighted combination
    return 0.4 * bigram_overlap + 0.6 * freq_sim


def score_response_appropriateness(
    context_utterance: str, response_utterance: str
) -> float:
    """Score how appropriate a response is given the preceding utterance.

    Uses heuristic rules around dialogue act compatibility, lexical
    relevance, and structural cues to produce a score in [0, 1].

    Parameters
    ----------
    context_utterance : str
        The utterance the response is replying to.
    response_utterance : str
        The generated response utterance.

    Returns
    -------
    float
        Appropriateness score in [0, 1].
    """
    if not context_utterance.strip() or not response_utterance.strip():
        return 0.5

    context_acts = extract_dialogue_acts(context_utterance)
    response_acts = extract_dialogue_acts(response_utterance)

    score = 0.5  # baseline

    # Questions should be followed by answers/statements
    if DialogueAct.QUESTION in context_acts:
        if (DialogueAct.STATEMENT in response_acts
                or DialogueAct.ANSWER in response_acts
                or DialogueAct.AGREEMENT in response_acts
                or DialogueAct.DISAGREEMENT in response_acts):
            score += 0.15
        elif DialogueAct.QUESTION in response_acts:
            score += 0.05  # counter-questions are somewhat appropriate
        else:
            score -= 0.05

    # Greetings should be met with greetings
    if DialogueAct.GREETING in context_acts:
        if DialogueAct.GREETING in response_acts:
            score += 0.2
        else:
            score -= 0.05

    # Farewells should be met with farewells
    if DialogueAct.FAREWELL in context_acts:
        if DialogueAct.FAREWELL in response_acts:
            score += 0.15

    # Apologies should get acknowledgment
    if DialogueAct.APOLOGY in context_acts:
        if (DialogueAct.ACKNOWLEDGMENT in response_acts
                or DialogueAct.AGREEMENT in response_acts):
            score += 0.1

    # Thanks should get acknowledgment
    if DialogueAct.THANKS in context_acts:
        if DialogueAct.ACKNOWLEDGMENT in response_acts:
            score += 0.1

    # Lexical relevance — some content word overlap expected
    context_tokens = set(_tokenize(context_utterance))
    response_tokens = set(_tokenize(response_utterance))
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "i", "you",
                  "it", "to", "and", "of", "in", "that", "this", "my", "your"}
    content_context = context_tokens - stop_words
    content_response = response_tokens - stop_words
    if content_context:
        overlap_ratio = len(content_context & content_response) / len(content_context)
        score += 0.1 * _clamp(overlap_ratio, 0.0, 1.0)

    # Penalize very short or empty responses
    if len(response_utterance.split()) < 2:
        score -= 0.1

    # Penalize exact copying
    if response_utterance.strip().lower() == context_utterance.strip().lower():
        score -= 0.2

    return _clamp(score)


def parse_dialogue_format(text: str) -> List[DialogueTurn]:
    """Parse a text block into a list of DialogueTurn objects.

    Supports multiple dialogue formats:
    - ``Speaker: utterance`` (colon-separated)
    - ``Speaker - utterance`` (dash-separated)
    - ``"utterance" said Speaker`` (narrative style)
    - ``[Speaker] utterance`` (bracket style)

    Parameters
    ----------
    text : str
        The raw dialogue text to parse.

    Returns
    -------
    List[DialogueTurn]
        Parsed dialogue turns.
    """
    turns: List[DialogueTurn] = []
    turn_number = 0

    # Try colon-separated format first (most common)
    colon_pattern = re.compile(
        r'^([A-Za-z][A-Za-z0-9 _.\'-]*?)\s*:\s*(.+)$', re.MULTILINE
    )
    colon_matches = colon_pattern.findall(text)

    if len(colon_matches) >= 2:
        for speaker, utterance in colon_matches:
            speaker = speaker.strip()
            utterance = utterance.strip()
            if speaker and utterance:
                turns.append(DialogueTurn(
                    speaker=speaker,
                    utterance=utterance,
                    turn_number=turn_number,
                ))
                turn_number += 1
        if turns:
            return turns

    # Try bracket format: [Speaker] utterance
    bracket_pattern = re.compile(
        r'^\[([A-Za-z][A-Za-z0-9 _.\'-]*?)\]\s*(.+)$', re.MULTILINE
    )
    bracket_matches = bracket_pattern.findall(text)

    if len(bracket_matches) >= 2:
        for speaker, utterance in bracket_matches:
            speaker = speaker.strip()
            utterance = utterance.strip()
            if speaker and utterance:
                turns.append(DialogueTurn(
                    speaker=speaker,
                    utterance=utterance,
                    turn_number=turn_number,
                ))
                turn_number += 1
        if turns:
            return turns

    # Try dash-separated format: Speaker - utterance
    dash_pattern = re.compile(
        r'^([A-Za-z][A-Za-z0-9 _.\'-]*?)\s*[-–—]\s*(.+)$', re.MULTILINE
    )
    dash_matches = dash_pattern.findall(text)

    if len(dash_matches) >= 2:
        for speaker, utterance in dash_matches:
            speaker = speaker.strip()
            utterance = utterance.strip()
            if speaker and utterance:
                turns.append(DialogueTurn(
                    speaker=speaker,
                    utterance=utterance,
                    turn_number=turn_number,
                ))
                turn_number += 1
        if turns:
            return turns

    # Try narrative format: "utterance" said Speaker / Speaker said "utterance"
    narrative_pattern = re.compile(
        r'["""](.+?)["""]'
        r'\s*(?:said|asked|replied|exclaimed|whispered|shouted|answered|responded)'
        r'\s+([A-Za-z][A-Za-z ]*)',
        re.MULTILINE,
    )
    narrative_matches = narrative_pattern.findall(text)

    if len(narrative_matches) >= 2:
        for utterance, speaker in narrative_matches:
            speaker = speaker.strip().rstrip(".,;!?")
            utterance = utterance.strip()
            if speaker and utterance:
                turns.append(DialogueTurn(
                    speaker=speaker,
                    utterance=utterance,
                    turn_number=turn_number,
                ))
                turn_number += 1
        if turns:
            return turns

    # Fallback: treat each non-empty line as a turn, alternating speakers
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) >= 2:
        speakers = ["Speaker A", "Speaker B"]
        for i, line in enumerate(lines):
            turns.append(DialogueTurn(
                speaker=speakers[i % 2],
                utterance=line,
                turn_number=i,
            ))

    return turns


# ---------------------------------------------------------------------------
# Built-in prompt data
# ---------------------------------------------------------------------------

_OPEN_DOMAIN_SCENARIOS: List[Dict[str, str]] = [
    {
        "topic": "favorite childhood memories",
        "starter": "What's your favorite childhood memory?",
        "context": "Two friends catching up over coffee.",
    },
    {
        "topic": "dream vacation",
        "starter": "If you could go anywhere in the world, where would you go?",
        "context": "Coworkers chatting during lunch break.",
    },
    {
        "topic": "unusual hobbies",
        "starter": "I recently started learning to juggle. Do you have any unusual hobbies?",
        "context": "Neighbors meeting at a community event.",
    },
    {
        "topic": "future of technology",
        "starter": "Do you think AI will change how we live in the next decade?",
        "context": "Strangers sitting next to each other on a train.",
    },
    {
        "topic": "cooking adventures",
        "starter": "I tried making sushi for the first time yesterday. It was a disaster!",
        "context": "Friends sharing stories at a dinner party.",
    },
    {
        "topic": "books and reading",
        "starter": "Have you read anything good lately?",
        "context": "Members of a book club before the meeting starts.",
    },
    {
        "topic": "pet stories",
        "starter": "My cat did the funniest thing this morning.",
        "context": "Pet owners meeting at a dog park.",
    },
    {
        "topic": "life philosophy",
        "starter": "Do you think people can really change who they are?",
        "context": "Old friends having a late-night conversation.",
    },
]

_TASK_ORIENTED_SCENARIOS: List[Dict[str, str]] = [
    {
        "task": "restaurant reservation",
        "goal": "Book a table for 4 at an Italian restaurant for Saturday evening.",
        "roles": "customer and restaurant host",
        "context": "Phone call to a busy downtown restaurant.",
    },
    {
        "task": "tech support",
        "goal": "Troubleshoot a laptop that won't connect to WiFi.",
        "roles": "customer and tech support agent",
        "context": "Online chat support session.",
    },
    {
        "task": "hotel booking",
        "goal": "Find and book a room with ocean view for a 3-night stay.",
        "roles": "traveler and hotel receptionist",
        "context": "Phone call to a resort hotel.",
    },
    {
        "task": "flight change",
        "goal": "Change a flight from Tuesday to Thursday, same route.",
        "roles": "passenger and airline agent",
        "context": "Call to airline customer service.",
    },
    {
        "task": "doctor appointment",
        "goal": "Schedule an appointment for persistent headaches.",
        "roles": "patient and receptionist",
        "context": "Phone call to a medical clinic.",
    },
    {
        "task": "car repair",
        "goal": "Get an estimate for fixing a strange engine noise.",
        "roles": "car owner and mechanic",
        "context": "Visit to an auto repair shop.",
    },
    {
        "task": "food delivery order",
        "goal": "Order dinner for the family with dietary restrictions.",
        "roles": "customer and order-taker",
        "context": "Phone call to a pizza restaurant.",
    },
    {
        "task": "bank inquiry",
        "goal": "Understand options for a savings account with good interest rates.",
        "roles": "customer and bank advisor",
        "context": "In-person meeting at a bank branch.",
    },
]

_DEBATE_SCENARIOS: List[Dict[str, str]] = [
    {
        "topic": "remote work vs. office work",
        "position_a": "Remote work is better for productivity and work-life balance.",
        "position_b": "Office work is essential for collaboration and company culture.",
    },
    {
        "topic": "social media impact on society",
        "position_a": "Social media has been a net positive, connecting people globally.",
        "position_b": "Social media has harmed mental health and spread misinformation.",
    },
    {
        "topic": "artificial intelligence regulation",
        "position_a": "AI should be heavily regulated to prevent misuse.",
        "position_b": "Over-regulation of AI will stifle innovation and progress.",
    },
    {
        "topic": "college education value",
        "position_a": "A college degree is still the best path to success.",
        "position_b": "Alternative education paths are often better and more affordable.",
    },
    {
        "topic": "space exploration funding",
        "position_a": "Space exploration deserves significant government funding.",
        "position_b": "Resources should be focused on solving problems on Earth first.",
    },
    {
        "topic": "universal basic income",
        "position_a": "UBI would reduce poverty and give people freedom to pursue meaningful work.",
        "position_b": "UBI would reduce motivation to work and be financially unsustainable.",
    },
]

_INTERVIEW_SCENARIOS: List[Dict[str, str]] = [
    {
        "role": "software engineer",
        "interviewer": "senior hiring manager",
        "topic": "system design and problem-solving approach",
        "context": "Final round interview at a tech company.",
    },
    {
        "role": "journalist",
        "interviewer": "local politician",
        "topic": "new education policy and its impact",
        "context": "Press conference after policy announcement.",
    },
    {
        "role": "scientist",
        "interviewer": "podcast host",
        "topic": "recent breakthrough in renewable energy",
        "context": "Science podcast interview.",
    },
    {
        "role": "entrepreneur",
        "interviewer": "venture capitalist",
        "topic": "startup pitch and business model",
        "context": "Investor pitch meeting.",
    },
]

_ROLEPLAY_SCENARIOS: List[Dict[str, str]] = [
    {
        "setting": "a medieval tavern",
        "characters": "a weary knight and a mysterious traveler",
        "scenario": "The knight seeks information about a dragon terrorizing nearby villages.",
        "tone": "adventurous and mysterious",
    },
    {
        "setting": "a space station orbiting Mars",
        "characters": "two astronauts from different countries",
        "scenario": "They discover an anomaly in their sensor data that could change everything.",
        "tone": "tense and scientific",
    },
    {
        "setting": "a 1920s speakeasy",
        "characters": "a jazz musician and a prohibition agent",
        "scenario": "The agent is undercover but finds themselves drawn to the music.",
        "tone": "atmospheric and morally ambiguous",
    },
    {
        "setting": "a futuristic AI research lab",
        "characters": "a lead researcher and their AI creation",
        "scenario": "The AI begins asking questions about its own existence.",
        "tone": "philosophical and thought-provoking",
    },
]

_PERSONA_TEMPLATES: List[Dict[str, str]] = [
    {
        "name": "Dr. Elena Vasquez",
        "age": "45",
        "occupation": "marine biologist",
        "personality": "passionate, detail-oriented, occasionally sarcastic",
        "speech_style": "uses scientific terminology naturally, tends to relate everything to the ocean",
        "background": "Grew up in a coastal town, has spent 20 years studying coral reefs.",
    },
    {
        "name": "Marcus Chen",
        "age": "28",
        "occupation": "independent game developer",
        "personality": "creative, enthusiastic, slightly anxious about deadlines",
        "speech_style": "uses gaming metaphors, casual tone, occasional self-deprecating humor",
        "background": "Left a corporate job to pursue game development. Working on a pixel-art RPG.",
    },
    {
        "name": "Grandma Rose",
        "age": "78",
        "occupation": "retired schoolteacher",
        "personality": "warm, wise, nostalgic, occasionally stubborn about modern technology",
        "speech_style": "uses old-fashioned expressions, tells stories from the past, gentle but firm",
        "background": "Taught elementary school for 40 years. Known for her legendary cookies.",
    },
    {
        "name": "Kai Nakamura",
        "age": "19",
        "occupation": "college freshman studying philosophy",
        "personality": "curious, idealistic, sometimes overthinks simple things",
        "speech_style": "asks lots of questions, references philosophers, uses hedging language",
        "background": "First in family to attend college. Fascinated by existentialism.",
    },
    {
        "name": "Chef Antonio",
        "age": "52",
        "occupation": "head chef at a Michelin-starred restaurant",
        "personality": "perfectionist, passionate, fiery temper but deeply caring",
        "speech_style": "uses Italian expressions, food metaphors, dramatic emphasis",
        "background": "Trained in Italy, moved to New York to open his dream restaurant.",
    },
    {
        "name": "Detective Sarah Park",
        "age": "35",
        "occupation": "homicide detective",
        "personality": "analytical, guarded, dry wit, deeply empathetic underneath",
        "speech_style": "precise language, asks probing questions, minimal small talk",
        "background": "Former prosecutor who switched to detective work to be closer to cases.",
    },
]

# Additional diverse prompt scenarios for reaching 30+ prompts
_ADDITIONAL_SCENARIOS: List[Dict[str, Any]] = [
    {
        "type": "open_domain",
        "prompt": "Two strangers are stuck in an elevator. They start talking about their biggest fears.",
        "participants": ["Stranger 1", "Stranger 2"],
    },
    {
        "type": "open_domain",
        "prompt": "A grandparent is teaching their grandchild to bake cookies while sharing family stories.",
        "participants": ["Grandparent", "Grandchild"],
    },
    {
        "type": "task_oriented",
        "prompt": "A customer is returning a defective product and the store has a strict no-return policy.",
        "participants": ["Customer", "Store Manager"],
    },
    {
        "type": "debate",
        "prompt": "Two teachers debate whether homework should be abolished in schools.",
        "participants": ["Teacher A", "Teacher B"],
    },
    {
        "type": "roleplay",
        "prompt": "A time traveler from the year 2200 arrives in 2024 and asks a local for help understanding current technology.",
        "participants": ["Time Traveler", "Local"],
    },
    {
        "type": "interview",
        "prompt": "A therapist is conducting a first session with a new client who is reluctant to open up.",
        "participants": ["Therapist", "Client"],
    },
    {
        "type": "negotiation",
        "prompt": "A freelancer is negotiating their rate with a client who has a limited budget.",
        "participants": ["Freelancer", "Client"],
    },
    {
        "type": "counseling",
        "prompt": "A guidance counselor helps a high school student who is torn between two very different career paths.",
        "participants": ["Counselor", "Student"],
    },
    {
        "type": "tutorial",
        "prompt": "An experienced gardener teaches a complete beginner how to start a vegetable garden.",
        "participants": ["Gardener", "Beginner"],
    },
    {
        "type": "open_domain",
        "prompt": "Two people from very different cultural backgrounds discuss their holiday traditions.",
        "participants": ["Person A", "Person B"],
    },
    {
        "type": "debate",
        "prompt": "Two friends argue about whether it's better to rent or buy a home in today's market.",
        "participants": ["Friend 1", "Friend 2"],
    },
    {
        "type": "roleplay",
        "prompt": "A detective interviews a witness who saw something strange in the park last night.",
        "participants": ["Detective", "Witness"],
    },
]


# ---------------------------------------------------------------------------
# DialoguePromptGenerator
# ---------------------------------------------------------------------------


class DialoguePromptGenerator:
    """Generates diverse dialogue prompts for different dialogue types.

    Supports open-domain, task-oriented, debate, interview, roleplay,
    multi-turn, and persona-based prompt generation with configurable
    parameters and built-in templates.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.RandomState(seed)

    def generate_open_domain_prompts(
        self, count: int = 8
    ) -> List[TaskPrompt]:
        """Generate open-domain casual conversation starter prompts.

        Creates prompts for natural, unstructured conversations on varied
        topics like hobbies, memories, philosophy, and daily life.

        Parameters
        ----------
        count : int
            Number of prompts to generate.

        Returns
        -------
        List[TaskPrompt]
        """
        prompts: List[TaskPrompt] = []
        scenarios = list(_OPEN_DOMAIN_SCENARIOS)

        for i in range(min(count, len(scenarios))):
            sc = scenarios[i]
            prompt_text = (
                f"Write a natural conversation between two people.\n"
                f"Context: {sc['context']}\n"
                f"Topic: {sc['topic']}\n"
                f"Opening line: \"{sc['starter']}\"\n\n"
                f"The dialogue should feel natural and include at least "
                f"6 turns. Each speaker should contribute meaningfully "
                f"to the conversation."
            )
            prompts.append(TaskPrompt(
                prompt_id=f"open_domain_{i:03d}",
                text=prompt_text,
                context=sc["context"],
                metadata={
                    "dialogue_type": "open_domain",
                    "topic": sc["topic"],
                    "starter": sc["starter"],
                },
                domain=TaskDomain.DIALOGUE,
            ))

        # Generate additional prompts by combining elements
        extra_topics = [
            "the meaning of friendship", "worst travel experiences",
            "if you could have any superpower", "favorite comfort foods",
            "unpopular opinions", "childhood dreams vs. adult reality",
            "the perfect weekend", "things you wish you'd learned sooner",
        ]
        extra_contexts = [
            "Roommates relaxing on a Sunday morning.",
            "Old classmates reuniting at a high school reunion.",
            "New colleagues getting to know each other at a team outing.",
            "Passengers on a delayed flight making conversation.",
        ]

        idx = len(scenarios)
        while len(prompts) < count and (extra_topics or extra_contexts):
            topic = extra_topics[idx % len(extra_topics)] if extra_topics else "life"
            context = extra_contexts[idx % len(extra_contexts)] if extra_contexts else "casual setting"
            prompt_text = (
                f"Write a natural conversation between two people.\n"
                f"Context: {context}\n"
                f"Topic: {topic}\n\n"
                f"The dialogue should flow naturally with at least "
                f"6 exchanges between the speakers."
            )
            prompts.append(TaskPrompt(
                prompt_id=f"open_domain_{idx:03d}",
                text=prompt_text,
                context=context,
                metadata={
                    "dialogue_type": "open_domain",
                    "topic": topic,
                },
                domain=TaskDomain.DIALOGUE,
            ))
            idx += 1

        return prompts[:count]

    def generate_task_oriented_prompts(
        self, count: int = 8
    ) -> List[TaskPrompt]:
        """Generate task-oriented dialogue prompts (goal-directed).

        Creates prompts for dialogues with specific goals such as booking,
        ordering, troubleshooting, or making arrangements.

        Parameters
        ----------
        count : int
            Number of prompts to generate.

        Returns
        -------
        List[TaskPrompt]
        """
        prompts: List[TaskPrompt] = []
        scenarios = list(_TASK_ORIENTED_SCENARIOS)

        for i in range(min(count, len(scenarios))):
            sc = scenarios[i]
            prompt_text = (
                f"Write a task-oriented dialogue between a {sc['roles']}.\n"
                f"Task: {sc['task']}\n"
                f"Goal: {sc['goal']}\n"
                f"Context: {sc['context']}\n\n"
                f"The dialogue should progress toward completing the task. "
                f"Include realistic complications or clarifications. "
                f"The conversation should have at least 8 turns."
            )
            prompts.append(TaskPrompt(
                prompt_id=f"task_oriented_{i:03d}",
                text=prompt_text,
                context=sc["context"],
                metadata={
                    "dialogue_type": "task_oriented",
                    "task": sc["task"],
                    "goal": sc["goal"],
                    "roles": sc["roles"],
                },
                domain=TaskDomain.DIALOGUE,
            ))

        return prompts[:count]

    def generate_debate_prompts(
        self, count: int = 6
    ) -> List[TaskPrompt]:
        """Generate debate-style dialogue prompts with opposing viewpoints.

        Creates prompts where two speakers argue different sides of a
        contentious topic with reasoned arguments.

        Parameters
        ----------
        count : int
            Number of prompts to generate.

        Returns
        -------
        List[TaskPrompt]
        """
        prompts: List[TaskPrompt] = []
        scenarios = list(_DEBATE_SCENARIOS)

        for i in range(min(count, len(scenarios))):
            sc = scenarios[i]
            prompt_text = (
                f"Write a structured debate between two speakers.\n"
                f"Topic: {sc['topic']}\n"
                f"Speaker A's position: {sc['position_a']}\n"
                f"Speaker B's position: {sc['position_b']}\n\n"
                f"Each speaker should present arguments and "
                f"counter-arguments. The debate should be respectful "
                f"but intellectually rigorous with at least 8 turns total."
            )
            prompts.append(TaskPrompt(
                prompt_id=f"debate_{i:03d}",
                text=prompt_text,
                context=f"Formal debate on {sc['topic']}",
                metadata={
                    "dialogue_type": "debate",
                    "topic": sc["topic"],
                    "position_a": sc["position_a"],
                    "position_b": sc["position_b"],
                },
                domain=TaskDomain.DIALOGUE,
            ))

        return prompts[:count]

    def generate_interview_prompts(
        self, count: int = 4
    ) -> List[TaskPrompt]:
        """Generate interview-format dialogue prompts.

        Creates prompts for question-answer format dialogues including
        job interviews, journalistic interviews, and podcast-style
        conversations.

        Parameters
        ----------
        count : int
            Number of prompts to generate.

        Returns
        -------
        List[TaskPrompt]
        """
        prompts: List[TaskPrompt] = []
        scenarios = list(_INTERVIEW_SCENARIOS)

        for i in range(min(count, len(scenarios))):
            sc = scenarios[i]
            prompt_text = (
                f"Write an interview dialogue.\n"
                f"Interviewer: {sc['interviewer']}\n"
                f"Interviewee role: {sc['role']}\n"
                f"Topic: {sc['topic']}\n"
                f"Context: {sc['context']}\n\n"
                f"The interviewer should ask thoughtful, probing questions. "
                f"The interviewee should give detailed, insightful answers. "
                f"Include follow-up questions and natural conversation flow. "
                f"At least 8 turns total."
            )
            prompts.append(TaskPrompt(
                prompt_id=f"interview_{i:03d}",
                text=prompt_text,
                context=sc["context"],
                metadata={
                    "dialogue_type": "interview",
                    "role": sc["role"],
                    "interviewer": sc["interviewer"],
                    "topic": sc["topic"],
                },
                domain=TaskDomain.DIALOGUE,
            ))

        return prompts[:count]

    def generate_roleplay_prompts(
        self, count: int = 4
    ) -> List[TaskPrompt]:
        """Generate character-based roleplay dialogue prompts.

        Creates prompts with specific settings, characters, and scenarios
        for creative dialogue generation.

        Parameters
        ----------
        count : int
            Number of prompts to generate.

        Returns
        -------
        List[TaskPrompt]
        """
        prompts: List[TaskPrompt] = []
        scenarios = list(_ROLEPLAY_SCENARIOS)

        for i in range(min(count, len(scenarios))):
            sc = scenarios[i]
            prompt_text = (
                f"Write a roleplay dialogue scene.\n"
                f"Setting: {sc['setting']}\n"
                f"Characters: {sc['characters']}\n"
                f"Scenario: {sc['scenario']}\n"
                f"Tone: {sc['tone']}\n\n"
                f"Stay in character throughout. The dialogue should "
                f"reveal personality through speech patterns and word "
                f"choice. Include stage directions or action descriptions "
                f"in brackets. At least 8 turns."
            )
            prompts.append(TaskPrompt(
                prompt_id=f"roleplay_{i:03d}",
                text=prompt_text,
                context=sc["setting"],
                metadata={
                    "dialogue_type": "roleplay",
                    "setting": sc["setting"],
                    "characters": sc["characters"],
                    "scenario": sc["scenario"],
                    "tone": sc["tone"],
                },
                domain=TaskDomain.DIALOGUE,
            ))

        return prompts[:count]

    def generate_multi_turn_context(
        self,
        num_turns: int = 4,
        topic: str = "general",
        participants: Optional[List[str]] = None,
    ) -> DialogueContext:
        """Build a conversation history as context for continuation prompts.

        Generates a plausible multi-turn dialogue context that can be
        used as input for dialogue continuation tasks.

        Parameters
        ----------
        num_turns : int
            Number of context turns to generate.
        topic : str
            The conversation topic.
        participants : List[str], optional
            Speaker names; defaults to ["Alice", "Bob"].

        Returns
        -------
        DialogueContext
            A context object with generated conversation history.
        """
        if participants is None:
            participants = ["Alice", "Bob"]

        topic_starters: Dict[str, List[List[str]]] = {
            "general": [
                [
                    "Hey, how's it going?",
                    "Pretty good! I've been keeping busy lately.",
                    "Oh yeah? What have you been up to?",
                    "I've been learning to play the guitar. It's harder than I thought!",
                ],
                [
                    "Did you catch the news this morning?",
                    "No, I missed it. Anything interesting?",
                    "There was this fascinating story about a new discovery in deep sea exploration.",
                    "That sounds cool! I've always been fascinated by the ocean.",
                ],
            ],
            "technology": [
                [
                    "Have you tried that new AI assistant everyone's talking about?",
                    "Yeah, I've been using it for work. It's surprisingly helpful.",
                    "What do you use it for mostly?",
                    "Mainly for brainstorming ideas and drafting emails. Saves me a lot of time.",
                ],
                [
                    "I'm thinking about upgrading my phone. Any recommendations?",
                    "It depends on what you care about most. Camera? Battery life?",
                    "Camera quality is definitely my top priority.",
                    "Then you might want to look at the latest models. The cameras have gotten incredible.",
                ],
            ],
            "philosophy": [
                [
                    "I've been thinking about what makes a meaningful life.",
                    "That's a big question. What brought this on?",
                    "I just finished reading a book about Stoic philosophy.",
                    "Interesting. What's the main takeaway for you?",
                ],
            ],
            "work": [
                [
                    "How's the new project going?",
                    "It's challenging but rewarding. We're behind schedule though.",
                    "What's causing the delay?",
                    "Mostly scope creep. The client keeps adding new requirements.",
                ],
            ],
            "food": [
                [
                    "I tried a new Thai restaurant last night. It was amazing!",
                    "Oh nice! Where is it?",
                    "It's on Main Street, next to the bookstore. You should check it out.",
                    "I love Thai food. What did you order?",
                ],
            ],
        }

        # Select conversation stubs based on topic
        available = topic_starters.get(topic, topic_starters["general"])
        stub_idx = self._rng.randint(0, len(available))
        selected_stub = available[stub_idx]

        context = DialogueContext(
            system_prompt=f"Continue the following conversation about {topic}.",
            scenario=f"A conversation about {topic}.",
            participants=list(participants),
        )

        for i in range(min(num_turns, len(selected_stub))):
            speaker = participants[i % len(participants)]
            context.add_turn(speaker, selected_stub[i])

        return context

    def generate_persona_based_prompts(
        self, count: int = 6
    ) -> List[TaskPrompt]:
        """Generate prompts with detailed character persona descriptions.

        Each prompt includes a persona with name, background, personality
        traits, and speech style to guide the dialogue generation.

        Parameters
        ----------
        count : int
            Number of prompts to generate.

        Returns
        -------
        List[TaskPrompt]
        """
        prompts: List[TaskPrompt] = []
        personas = list(_PERSONA_TEMPLATES)

        conversation_topics = [
            "their biggest professional challenge",
            "a book that changed their perspective",
            "what they would do differently if they could start over",
            "their vision for the future",
            "a moment that defined who they are",
            "advice they would give to their younger self",
        ]

        for i in range(min(count, len(personas))):
            persona = personas[i % len(personas)]
            topic = conversation_topics[i % len(conversation_topics)]

            persona_desc = (
                f"Name: {persona['name']}\n"
                f"Age: {persona['age']}\n"
                f"Occupation: {persona['occupation']}\n"
                f"Personality: {persona['personality']}\n"
                f"Speech style: {persona['speech_style']}\n"
                f"Background: {persona['background']}"
            )

            prompt_text = (
                f"Write a dialogue where the following character is speaking "
                f"with a friend about {topic}.\n\n"
                f"Character Profile:\n{persona_desc}\n\n"
                f"The character's speech should reflect their personality, "
                f"occupation, and speech style as described. Their partner "
                f"in conversation should react naturally and ask follow-up "
                f"questions. At least 8 turns total."
            )
            prompts.append(TaskPrompt(
                prompt_id=f"persona_{i:03d}",
                text=prompt_text,
                context=f"Conversation with {persona['name']}",
                metadata={
                    "dialogue_type": "persona_based",
                    "persona": persona,
                    "topic": topic,
                },
                domain=TaskDomain.DIALOGUE,
            ))

        return prompts[:count]

    def format_dialogue_prompt(
        self,
        context: DialogueContext,
        instruction: str = "",
        include_system_prompt: bool = True,
    ) -> str:
        """Format a dialogue context and instruction into a prompt string.

        Combines system prompt, conversation history, and generation
        instruction into a single formatted string.

        Parameters
        ----------
        context : DialogueContext
            The dialogue context with conversation history.
        instruction : str
            Additional instruction for the generation.
        include_system_prompt : bool
            Whether to include the system prompt.

        Returns
        -------
        str
            Formatted prompt string ready for model input.
        """
        parts: List[str] = []

        if include_system_prompt and context.system_prompt:
            parts.append(f"System: {context.system_prompt}")
            parts.append("")

        if context.scenario:
            parts.append(f"Scenario: {context.scenario}")
            parts.append("")

        if context.participants:
            parts.append(f"Participants: {', '.join(context.participants)}")
            parts.append("")

        if context.turns:
            parts.append("Conversation so far:")
            for turn in context.turns:
                parts.append(f"{turn.speaker}: {turn.utterance}")
            parts.append("")

        if instruction:
            parts.append(f"Instruction: {instruction}")
        else:
            next_speaker = ""
            if context.participants and context.turns:
                last = context.last_speaker()
                idx = context.participants.index(last) if last in context.participants else -1
                next_idx = (idx + 1) % len(context.participants)
                next_speaker = context.participants[next_idx]
            if next_speaker:
                parts.append(f"Continue the dialogue. {next_speaker}'s turn:")
            else:
                parts.append("Continue the dialogue:")

        return "\n".join(parts)

    def generate_all_prompts(self) -> List[TaskPrompt]:
        """Generate a comprehensive set of all dialogue prompt types.

        Returns
        -------
        List[TaskPrompt]
            At least 30 diverse dialogue prompts.
        """
        all_prompts: List[TaskPrompt] = []
        all_prompts.extend(self.generate_open_domain_prompts(8))
        all_prompts.extend(self.generate_task_oriented_prompts(8))
        all_prompts.extend(self.generate_debate_prompts(6))
        all_prompts.extend(self.generate_interview_prompts(4))
        all_prompts.extend(self.generate_roleplay_prompts(4))
        all_prompts.extend(self.generate_persona_based_prompts(6))

        # Add additional diverse scenarios
        for i, sc in enumerate(_ADDITIONAL_SCENARIOS):
            prompt_text = (
                f"Write a dialogue for the following scenario:\n"
                f"{sc['prompt']}\n\n"
                f"Participants: {', '.join(sc['participants'])}\n"
                f"Include at least 6 turns. Make the dialogue natural "
                f"and engaging."
            )
            all_prompts.append(TaskPrompt(
                prompt_id=f"additional_{i:03d}",
                text=prompt_text,
                context=sc["prompt"],
                metadata={
                    "dialogue_type": sc["type"],
                    "participants": sc["participants"],
                },
                domain=TaskDomain.DIALOGUE,
            ))

        logger.info("Generated %d total dialogue prompts", len(all_prompts))
        return all_prompts


# ---------------------------------------------------------------------------
# DialogueEvaluator
# ---------------------------------------------------------------------------


class DialogueEvaluator(TaskEvaluator):
    """Evaluates dialogue generation quality and diversity.

    Provides comprehensive metrics for coherence, engagement, persona
    consistency, topic diversity, response diversity, turn-taking,
    informativeness, naturalness, and contradiction avoidance.
    """

    def __init__(
        self,
        metrics_config: Optional[Dict[str, Any]] = None,
        config: Optional[DialogueConfig] = None,
    ) -> None:
        super().__init__(metrics_config=metrics_config)
        self.config = config or DialogueConfig()

    def evaluate_coherence(
        self, response: str, context: str
    ) -> float:
        """Evaluate how coherent a response is given the conversation context.

        Measures relevance through lexical overlap, dialogue act compatibility,
        topic continuity, and structural coherence.

        Parameters
        ----------
        response : str
            The generated dialogue response.
        context : str
            The preceding conversation context.

        Returns
        -------
        float
            Coherence score in [0, 1].
        """
        if not response.strip() or not context.strip():
            return 0.0

        context_tokens = set(_tokenize(context))
        response_tokens = set(_tokenize(response))

        # Remove function words for content overlap
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "i", "you",
            "it", "to", "and", "of", "in", "that", "this", "my", "your",
            "we", "they", "he", "she", "but", "or", "not", "have", "has",
            "do", "does", "did", "will", "would", "could", "should",
            "be", "been", "being", "for", "on", "with", "at", "by",
        }
        content_context = context_tokens - stop_words
        content_response = response_tokens - stop_words

        # Content overlap score
        if content_context:
            overlap = len(content_context & content_response) / len(content_context)
            content_score = _clamp(overlap * 2.0)
        else:
            content_score = 0.5

        # Dialogue act coherence
        context_lines = [l.strip() for l in context.split("\n") if l.strip()]
        last_context_line = context_lines[-1] if context_lines else ""
        # Extract just the utterance part if formatted as "Speaker: utterance"
        if ":" in last_context_line:
            last_context_line = last_context_line.split(":", 1)[1].strip()

        response_first_line = response.split("\n")[0].strip()
        if ":" in response_first_line:
            response_first_line = response_first_line.split(":", 1)[1].strip()

        act_score = score_response_appropriateness(last_context_line, response_first_line)

        # Topic continuity via bigram overlap
        context_bigrams = set(
            " ".join(bg) for bg in _ngrams(_tokenize(context), 2)
        )
        response_bigrams = set(
            " ".join(bg) for bg in _ngrams(_tokenize(response), 2)
        )
        if context_bigrams:
            bigram_overlap = len(context_bigrams & response_bigrams) / len(context_bigrams)
            topic_score = _clamp(bigram_overlap * 3.0)
        else:
            topic_score = 0.5

        # Structural coherence — response should be well-formed
        turns = parse_dialogue_format(response)
        if turns:
            structural_score = min(1.0, len(turns) / 2.0)
        else:
            # Even without parseable turns, non-empty text gets partial credit
            structural_score = 0.3 if len(response.split()) > 5 else 0.1

        # Weighted combination
        coherence = (
            0.30 * content_score
            + 0.30 * act_score
            + 0.20 * topic_score
            + 0.20 * structural_score
        )
        return _clamp(coherence)

    def evaluate_engagement(
        self, text: str
    ) -> float:
        """Evaluate dialogue engagement through question asking and topic development.

        Measures whether the dialogue actively develops topics, asks questions,
        introduces new ideas, and avoids stagnation.

        Parameters
        ----------
        text : str
            The full dialogue text.

        Returns
        -------
        float
            Engagement score in [0, 1].
        """
        turns = parse_dialogue_format(text)
        if not turns:
            return 0.0

        # Question frequency — engaged dialogue asks questions
        question_count = sum(1 for t in turns if t.has_question())
        question_ratio = question_count / len(turns) if turns else 0.0
        question_score = _clamp(question_ratio * 2.5)

        # Topic development — later turns introduce new content words
        all_content_words: Set[str] = set()
        new_word_ratios: List[float] = []
        for turn in turns:
            turn_words = set(_tokenize(turn.utterance)) - {
                "the", "a", "an", "is", "are", "i", "you", "it", "to",
                "and", "of", "in", "that", "this",
            }
            if all_content_words and turn_words:
                new_words = turn_words - all_content_words
                new_ratio = len(new_words) / len(turn_words)
                new_word_ratios.append(new_ratio)
            all_content_words.update(turn_words)

        topic_dev_score = float(np.mean(new_word_ratios)) if new_word_ratios else 0.5

        # Dialogue act variety — engaged dialogue uses different speech acts
        all_acts: Set[str] = set()
        for turn in turns:
            acts = extract_dialogue_acts(turn.utterance)
            for act in acts:
                all_acts.add(act.name)
        act_variety = min(1.0, len(all_acts) / 6.0)

        # Turn length variation — engaged dialogue isn't monotonous
        lengths = [turn.word_count for turn in turns]
        if len(lengths) > 1:
            length_cv = float(np.std(lengths)) / (float(np.mean(lengths)) + 1e-6)
            length_variety = _clamp(length_cv * 1.5)
        else:
            length_variety = 0.5

        # Elaboration — longer responses suggest engagement
        avg_length = float(np.mean(lengths)) if lengths else 0.0
        length_score = _sigmoid(avg_length, midpoint=15, steepness=0.1)

        engagement = (
            0.25 * question_score
            + 0.20 * topic_dev_score
            + 0.20 * act_variety
            + 0.15 * length_variety
            + 0.20 * length_score
        )
        return _clamp(engagement)

    def evaluate_persona_consistency(
        self,
        text: str,
        persona: Optional[Dict[str, str]] = None,
    ) -> float:
        """Evaluate how consistently a dialogue maintains a character persona.

        Checks speech style markers, vocabulary consistency, and adherence
        to the described persona traits.

        Parameters
        ----------
        text : str
            The dialogue text.
        persona : Dict[str, str], optional
            Persona description with keys like ``speech_style``, ``personality``,
            ``occupation``, ``name``.

        Returns
        -------
        float
            Persona consistency score in [0, 1].
        """
        turns = parse_dialogue_format(text)
        if not turns:
            return 0.0

        if persona is None:
            # Without persona info, evaluate basic style consistency
            return self._evaluate_style_self_consistency(turns)

        persona_name = persona.get("name", "")
        persona_occupation = persona.get("occupation", "")
        persona_personality = persona.get("personality", "")
        persona_speech_style = persona.get("speech_style", "")

        # Identify persona's turns
        persona_turns = [
            t for t in turns
            if persona_name and persona_name.lower() in t.speaker.lower()
        ]
        if not persona_turns:
            # Try to match first speaker if no name match
            speakers = list({t.speaker for t in turns})
            if speakers:
                persona_turns = [t for t in turns if t.speaker == speakers[0]]

        if not persona_turns:
            return 0.5

        scores: List[float] = []

        # Occupation keyword presence
        if persona_occupation:
            occupation_words = set(_tokenize(persona_occupation))
            combined_text = " ".join(t.utterance for t in persona_turns).lower()
            combined_tokens = set(_tokenize(combined_text))
            if occupation_words:
                occ_overlap = len(occupation_words & combined_tokens) / len(occupation_words)
                scores.append(_clamp(occ_overlap * 2.0))

        # Personality trait manifestation
        if persona_personality:
            personality_words = set(_tokenize(persona_personality))
            combined_text = " ".join(t.utterance for t in persona_turns).lower()
            combined_tokens = set(_tokenize(combined_text))
            if personality_words:
                pers_overlap = len(personality_words & combined_tokens) / len(personality_words)
                scores.append(_clamp(pers_overlap * 2.5))

        # Speech style adherence
        if persona_speech_style:
            style_lower = persona_speech_style.lower()
            combined_text = " ".join(t.utterance for t in persona_turns).lower()

            # Check for mentioned style markers
            if "formal" in style_lower or "scientific" in style_lower:
                formal_hits = _count_pattern_hits(combined_text, _FORMAL_MARKERS)
                scores.append(_clamp(formal_hits / max(len(persona_turns), 1) * 0.5))
            elif "casual" in style_lower or "informal" in style_lower:
                casual_hits = _count_pattern_hits(combined_text, _CASUAL_MARKERS)
                scores.append(_clamp(casual_hits / max(len(persona_turns), 1) * 0.5))

            # Check for specific style features mentioned in description
            style_keywords = set(_tokenize(persona_speech_style))
            text_tokens = set(_tokenize(combined_text))
            if style_keywords:
                style_overlap = len(style_keywords & text_tokens) / len(style_keywords)
                scores.append(_clamp(style_overlap * 2.0))

        # Self-consistency of persona turns
        consistency = self._evaluate_style_self_consistency(persona_turns)
        scores.append(consistency)

        return float(np.mean(scores)) if scores else 0.5

    def _evaluate_style_self_consistency(
        self, turns: List[DialogueTurn]
    ) -> float:
        """Evaluate internal style consistency across turns by a speaker."""
        if len(turns) < 2:
            return 0.5

        # Collect per-turn features
        formality_scores: List[float] = []
        avg_lengths: List[float] = []
        punctuation_rates: List[float] = []

        for turn in turns:
            text = turn.utterance.lower()
            tokens = _tokenize(text)
            word_count = len(tokens)

            # Formality: ratio of formal vs casual markers
            formal = _count_pattern_hits(text, _FORMAL_MARKERS)
            casual = _count_pattern_hits(text, _CASUAL_MARKERS)
            total_markers = formal + casual
            if total_markers > 0:
                formality_scores.append(formal / total_markers)
            else:
                formality_scores.append(0.5)

            avg_lengths.append(float(word_count))

            # Punctuation rate
            punct_count = sum(1 for c in turn.utterance if c in "!?.,;:")
            punctuation_rates.append(
                punct_count / max(word_count, 1)
            )

        # Consistency = 1 - normalized variance of each feature
        consistency_scores: List[float] = []

        if len(formality_scores) > 1:
            formality_var = float(np.std(formality_scores))
            consistency_scores.append(1.0 - min(1.0, formality_var * 2.0))

        if len(avg_lengths) > 1:
            mean_len = float(np.mean(avg_lengths))
            if mean_len > 0:
                length_cv = float(np.std(avg_lengths)) / mean_len
                consistency_scores.append(1.0 - min(1.0, length_cv))
            else:
                consistency_scores.append(0.5)

        if len(punctuation_rates) > 1:
            punct_var = float(np.std(punctuation_rates))
            consistency_scores.append(1.0 - min(1.0, punct_var * 3.0))

        return float(np.mean(consistency_scores)) if consistency_scores else 0.5

    def evaluate_topic_diversity(
        self, texts: List[str]
    ) -> float:
        """Evaluate variety of topics covered across multiple dialogue outputs.

        Measures how many different topic areas the dialogues touch on,
        rewarding breadth of coverage.

        Parameters
        ----------
        texts : List[str]
            Multiple dialogue generation outputs.

        Returns
        -------
        float
            Topic diversity score in [0, 1].
        """
        if not texts:
            return 0.0

        topic_coverage: Dict[str, int] = defaultdict(int)

        for text in texts:
            text_lower = text.lower()
            text_tokens = set(_tokenize(text_lower))

            for topic_name, keywords in _TOPIC_KEYWORDS.items():
                keyword_set = set(keywords)
                matches = len(text_tokens & keyword_set)
                if matches >= 2:
                    topic_coverage[topic_name] += 1

        if not topic_coverage:
            return 0.1

        # Number of distinct topics covered
        topics_covered = len(topic_coverage)
        total_possible = len(_TOPIC_KEYWORDS)
        coverage_ratio = topics_covered / total_possible

        # Distribution evenness of topic usage
        counts = list(topic_coverage.values())
        evenness = _normalized_entropy(counts)

        # Combined score
        diversity = 0.6 * _clamp(coverage_ratio * 2.0) + 0.4 * evenness
        return _clamp(diversity)

    def evaluate_response_diversity(
        self, texts: List[str]
    ) -> float:
        """Evaluate structural variety across multiple dialogue responses.

        Measures variation in sentence structures, dialogue acts, and
        opening patterns across generated dialogues.

        Parameters
        ----------
        texts : List[str]
            Multiple dialogue generation outputs.

        Returns
        -------
        float
            Response diversity score in [0, 1].
        """
        if len(texts) < 2:
            return 0.0

        # Unique first lines (de-duplicated openings)
        first_lines: List[str] = []
        for text in texts:
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if lines:
                first_lines.append(lines[0].lower())

        unique_openings = len(set(first_lines))
        opening_diversity = unique_openings / len(first_lines) if first_lines else 0.0

        # Lexical diversity across responses
        all_tokens: List[str] = []
        per_text_tokens: List[Set[str]] = []
        for text in texts:
            tokens = _tokenize(text)
            all_tokens.extend(tokens)
            per_text_tokens.append(set(tokens))

        overall_ttr = _type_token_ratio(all_tokens)

        # Pairwise Jaccard distance between responses
        pairwise_distances: List[float] = []
        for i in range(len(per_text_tokens)):
            for j in range(i + 1, len(per_text_tokens)):
                sim = _jaccard_similarity(per_text_tokens[i], per_text_tokens[j])
                pairwise_distances.append(1.0 - sim)

        avg_distance = float(np.mean(pairwise_distances)) if pairwise_distances else 0.0

        # Structural variety — distribution of turn counts
        turn_counts: List[int] = []
        for text in texts:
            turns = parse_dialogue_format(text)
            turn_counts.append(len(turns))

        if len(turn_counts) > 1:
            turn_count_cv = float(np.std(turn_counts)) / (float(np.mean(turn_counts)) + 1e-6)
            structure_variety = _clamp(turn_count_cv)
        else:
            structure_variety = 0.0

        diversity = (
            0.25 * opening_diversity
            + 0.25 * overall_ttr
            + 0.30 * avg_distance
            + 0.20 * structure_variety
        )
        return _clamp(diversity)

    def evaluate_turn_taking(
        self, text: str, expected_speakers: int = 2
    ) -> float:
        """Evaluate the quality of turn-taking in a dialogue.

        Assesses response length appropriateness, speaker balance, and
        natural alternation patterns.

        Parameters
        ----------
        text : str
            The dialogue text.
        expected_speakers : int
            Expected number of speakers.

        Returns
        -------
        float
            Turn-taking quality score in [0, 1].
        """
        turns = parse_dialogue_format(text)
        if not turns:
            return 0.0

        # Speaker count accuracy
        unique_speakers = len({t.speaker for t in turns})
        if expected_speakers > 0:
            speaker_accuracy = 1.0 - abs(unique_speakers - expected_speakers) / max(
                expected_speakers, unique_speakers
            )
        else:
            speaker_accuracy = 0.5

        # Alternation pattern — ideally speakers alternate
        alternations = 0
        for i in range(len(turns) - 1):
            if turns[i].speaker != turns[i + 1].speaker:
                alternations += 1
        max_alternations = len(turns) - 1
        alternation_score = alternations / max_alternations if max_alternations > 0 else 0.0

        # Speaker balance — each speaker should contribute roughly equally
        speaker_counts: Counter = Counter()
        for turn in turns:
            speaker_counts[turn.speaker] += 1

        if len(speaker_counts) > 1:
            counts = list(speaker_counts.values())
            max_count = max(counts)
            min_count = min(counts)
            balance = min_count / max_count if max_count > 0 else 0.0
        else:
            balance = 0.0

        # Turn length appropriateness — not too short, not too long
        lengths = [turn.word_count for turn in turns]
        appropriate_count = sum(
            1 for l in lengths if 3 <= l <= 150
        )
        length_appropriateness = appropriate_count / len(lengths) if lengths else 0.0

        # Total turn count — should have enough turns for meaningful dialogue
        turn_count_score = _sigmoid(len(turns), midpoint=4, steepness=0.5)

        score = (
            0.20 * speaker_accuracy
            + 0.25 * alternation_score
            + 0.20 * balance
            + 0.15 * length_appropriateness
            + 0.20 * turn_count_score
        )
        return _clamp(score)

    def evaluate_informativeness(
        self, text: str
    ) -> float:
        """Evaluate how much new information each turn introduces.

        Measures information density through unique content words, facts,
        and details introduced across the conversation.

        Parameters
        ----------
        text : str
            The dialogue text.

        Returns
        -------
        float
            Informativeness score in [0, 1].
        """
        turns = parse_dialogue_format(text)
        if not turns:
            return 0.0

        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "i", "you",
            "it", "to", "and", "of", "in", "that", "this", "my", "your",
            "we", "they", "he", "she", "but", "or", "not", "have", "has",
            "do", "does", "did", "will", "would", "be", "been", "for",
            "on", "with", "at", "by", "just", "so", "yes", "no", "yeah",
            "ok", "okay", "right", "well", "um", "uh",
        }

        cumulative_vocab: Set[str] = set()
        new_info_ratios: List[float] = []
        unique_content_counts: List[int] = []

        for turn in turns:
            tokens = set(_tokenize(turn.utterance))
            content_tokens = tokens - stop_words

            if cumulative_vocab:
                new_tokens = content_tokens - cumulative_vocab
                ratio = len(new_tokens) / max(len(content_tokens), 1)
                new_info_ratios.append(ratio)
            else:
                new_info_ratios.append(1.0)

            unique_content_counts.append(len(content_tokens))
            cumulative_vocab.update(content_tokens)

        # Average new information ratio
        avg_new_info = float(np.mean(new_info_ratios)) if new_info_ratios else 0.0

        # Information density — unique content words per turn
        avg_density = (
            float(np.mean(unique_content_counts)) if unique_content_counts else 0.0
        )
        density_score = _sigmoid(avg_density, midpoint=5, steepness=0.2)

        # Total vocabulary richness
        total_tokens = _tokenize(text)
        if total_tokens:
            ttr = _type_token_ratio(total_tokens)
        else:
            ttr = 0.0

        # Presence of specific/concrete details (numbers, names, etc.)
        detail_patterns = [
            r'\b\d+\b',  # numbers
            r'\b[A-Z][a-z]+\b',  # proper nouns
            r'\b(?:specifically|exactly|precisely|approximately)\b',
        ]
        detail_hits = _count_pattern_hits(text, detail_patterns)
        detail_score = _sigmoid(detail_hits, midpoint=3, steepness=0.3)

        informativeness = (
            0.30 * avg_new_info
            + 0.25 * density_score
            + 0.25 * ttr
            + 0.20 * detail_score
        )
        return _clamp(informativeness)

    def evaluate_naturalness(
        self, text: str
    ) -> float:
        """Evaluate how natural and conversational the dialogue sounds.

        Checks for conversational markers, fillers, hedging, contractions,
        and other features of natural speech.

        Parameters
        ----------
        text : str
            The dialogue text.

        Returns
        -------
        float
            Naturalness score in [0, 1].
        """
        turns = parse_dialogue_format(text)
        if not turns:
            # Fall back to text-level analysis
            tokens = _tokenize(text)
            if not tokens:
                return 0.0
            turns_fallback = True
        else:
            turns_fallback = False

        text_lower = text.lower()

        # Conversational filler presence (moderate amount is natural)
        filler_count = 0
        for filler in _CONVERSATIONAL_FILLERS:
            filler_count += len(re.findall(r'\b' + re.escape(filler) + r'\b', text_lower))

        total_words = len(text_lower.split())
        filler_ratio = filler_count / max(total_words, 1)
        # Natural text has some fillers (1-5%) but not too many
        if 0.005 <= filler_ratio <= 0.08:
            filler_score = 1.0
        elif filler_ratio < 0.005:
            filler_score = filler_ratio / 0.005
        else:
            filler_score = max(0.0, 1.0 - (filler_ratio - 0.08) * 5)

        # Contraction usage (natural speech uses contractions)
        contractions = re.findall(
            r"\b(?:don't|doesn't|didn't|can't|won't|wouldn't|couldn't|shouldn't|"
            r"isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|it's|that's|"
            r"there's|here's|what's|who's|how's|I'm|I've|I'll|I'd|you're|"
            r"you've|you'll|you'd|we're|we've|we'll|we'd|they're|they've|"
            r"they'll|they'd|he's|she's|let's)\b",
            text_lower,
        )
        contraction_ratio = len(contractions) / max(total_words, 1)
        contraction_score = _clamp(contraction_ratio * 20.0)

        # Hedging language (natural conversation includes hedging)
        hedging_hits = _count_pattern_hits(text_lower, _HEDGING_PATTERNS)
        hedging_score = _sigmoid(hedging_hits, midpoint=2, steepness=0.5)

        # Backchannel signals
        backchannel_hits = _count_pattern_hits(text_lower, _BACKCHANNEL_PATTERNS)
        backchannel_score = _sigmoid(backchannel_hits, midpoint=1, steepness=0.5)

        # Sentence length variety (natural speech varies)
        sentences = _sentences(text)
        if len(sentences) > 1:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = float(np.mean(sent_lengths))
            cv = float(np.std(sent_lengths)) / (mean_len + 1e-6)
            variety_score = _clamp(cv)
        else:
            variety_score = 0.3

        # Emotional markers (natural conversation has emotion)
        emotion_count = 0
        for emotion_words in _EMOTIONAL_MARKERS.values():
            for word in emotion_words:
                if word in text_lower:
                    emotion_count += 1
        emotion_score = _sigmoid(emotion_count, midpoint=3, steepness=0.3)

        naturalness = (
            0.18 * filler_score
            + 0.15 * contraction_score
            + 0.18 * hedging_score
            + 0.12 * backchannel_score
            + 0.17 * variety_score
            + 0.20 * emotion_score
        )
        return _clamp(naturalness)

    def evaluate_contradiction_avoidance(
        self, text: str
    ) -> float:
        """Evaluate consistency by checking for self-contradictions.

        Detects cases where speakers contradict themselves or each other
        inappropriately (outside of debates where disagreement is expected).

        Parameters
        ----------
        text : str
            The dialogue text.

        Returns
        -------
        float
            Contradiction avoidance score in [0, 1].  Higher is better
            (fewer contradictions).
        """
        turns = parse_dialogue_format(text)
        if len(turns) < 2:
            return 1.0

        contradiction_count = 0
        total_checks = 0

        # Group turns by speaker
        speaker_turns: Dict[str, List[str]] = defaultdict(list)
        for turn in turns:
            speaker_turns[turn.speaker].append(turn.utterance.lower())

        # Check for self-contradiction within same speaker
        for speaker, utterances in speaker_turns.items():
            if len(utterances) < 2:
                continue

            for i in range(len(utterances)):
                for j in range(i + 1, len(utterances)):
                    total_checks += 1
                    # Check for negation patterns
                    contradiction = self._detect_contradiction_pair(
                        utterances[i], utterances[j]
                    )
                    if contradiction:
                        contradiction_count += 1

        # Check adjacent turns for logical inconsistency
        for i in range(len(turns) - 1):
            if turns[i].speaker == turns[i + 1].speaker:
                total_checks += 1
                contradiction = self._detect_contradiction_pair(
                    turns[i].utterance.lower(),
                    turns[i + 1].utterance.lower(),
                )
                if contradiction:
                    contradiction_count += 1

        if total_checks == 0:
            return 1.0

        contradiction_ratio = contradiction_count / total_checks
        return _clamp(1.0 - contradiction_ratio * 3.0)

    def _detect_contradiction_pair(
        self, text_a: str, text_b: str
    ) -> bool:
        """Detect if two texts from the same speaker are contradictory."""
        # Pattern: "I like X" vs "I don't like X" / "I hate X"
        like_pattern = re.compile(r"i\s+(?:really\s+)?(?:like|love|enjoy)\s+(\w+)")
        dislike_pattern = re.compile(
            r"i\s+(?:don'?t\s+(?:really\s+)?(?:like|love|enjoy)|hate|dislike|can'?t\s+stand)\s+(\w+)"
        )

        likes_a = set(m.group(1) for m in like_pattern.finditer(text_a))
        dislikes_a = set(m.group(1) for m in dislike_pattern.finditer(text_a))
        likes_b = set(m.group(1) for m in like_pattern.finditer(text_b))
        dislikes_b = set(m.group(1) for m in dislike_pattern.finditer(text_b))

        # Same thing liked then disliked (or vice versa)
        if (likes_a & dislikes_b) or (dislikes_a & likes_b):
            return True

        # Pattern: "I am X" vs "I am not X"
        am_pattern = re.compile(r"i\s+(?:am|'m)\s+(?:a\s+)?(\w+)")
        am_not_pattern = re.compile(r"i\s+(?:am\s+not|'m\s+not|ain'?t)\s+(?:a\s+)?(\w+)")

        am_a = set(m.group(1) for m in am_pattern.finditer(text_a))
        am_not_a = set(m.group(1) for m in am_not_pattern.finditer(text_a))
        am_b = set(m.group(1) for m in am_pattern.finditer(text_b))
        am_not_b = set(m.group(1) for m in am_not_pattern.finditer(text_b))

        if (am_a & am_not_b) or (am_not_a & am_b):
            return True

        # Pattern: "I have X" vs "I don't have X" / "I've never X"
        have_pattern = re.compile(r"i\s+(?:have|'ve)\s+(?:a\s+)?(\w+)")
        have_not_pattern = re.compile(
            r"i\s+(?:don'?t\s+have|haven'?t|'ve\s+never|have\s+never)\s+(?:a\s+)?(\w+)"
        )

        have_a = set(m.group(1) for m in have_pattern.finditer(text_a))
        have_not_a = set(m.group(1) for m in have_not_pattern.finditer(text_a))
        have_b = set(m.group(1) for m in have_pattern.finditer(text_b))
        have_not_b = set(m.group(1) for m in have_not_pattern.finditer(text_b))

        if (have_a & have_not_b) or (have_not_a & have_b):
            return True

        # Pattern: direct assertion vs negation of same statement
        # "X is true" vs "X is not true"
        assertion_pattern = re.compile(r"(\w+(?:\s+\w+)?)\s+is\s+(true|right|correct|good|bad|wrong)")
        negation_pattern = re.compile(
            r"(\w+(?:\s+\w+)?)\s+is\s+(?:not\s+)?(true|right|correct|good|bad|wrong)"
        )

        assertions_a = {
            (m.group(1), m.group(2))
            for m in assertion_pattern.finditer(text_a)
        }
        negations_b = set()
        for m in re.finditer(
            r"(\w+(?:\s+\w+)?)\s+is\s+not\s+(true|right|correct|good|bad|wrong)",
            text_b,
        ):
            negations_b.add((m.group(1), m.group(2)))

        if assertions_a & negations_b:
            return True

        return False

    def compute_dialogue_quality_score(
        self, text: str, context: str = "", persona: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Compute an aggregate dialogue quality score from all quality metrics.

        Parameters
        ----------
        text : str
            The dialogue text.
        context : str
            Preceding conversation context.
        persona : Dict[str, str], optional
            Persona description for consistency evaluation.

        Returns
        -------
        Dict[str, float]
            Individual metric scores and the aggregate ``overall_quality``.
        """
        scores: Dict[str, float] = {}

        scores["coherence"] = self.evaluate_coherence(text, context)
        scores["engagement"] = self.evaluate_engagement(text)
        scores["turn_taking"] = self.evaluate_turn_taking(text)
        scores["informativeness"] = self.evaluate_informativeness(text)
        scores["naturalness"] = self.evaluate_naturalness(text)
        scores["contradiction_avoidance"] = self.evaluate_contradiction_avoidance(text)

        if persona:
            scores["persona_consistency"] = self.evaluate_persona_consistency(
                text, persona
            )

        metric_values = list(scores.values())
        scores["overall_quality"] = float(np.mean(metric_values)) if metric_values else 0.0

        return scores

    def compute_dialogue_diversity_score(
        self, texts: List[str]
    ) -> Dict[str, float]:
        """Compute an aggregate dialogue diversity score from diversity metrics.

        Parameters
        ----------
        texts : List[str]
            Multiple dialogue generation outputs to compare.

        Returns
        -------
        Dict[str, float]
            Individual diversity metric scores and the aggregate
            ``overall_diversity``.
        """
        if not texts:
            return {"overall_diversity": 0.0}

        metrics = DialogueDiversityMetrics()
        scores: Dict[str, float] = {}

        scores["response_variety"] = metrics.compute_response_variety(texts)
        scores["topic_coverage"] = metrics.compute_topic_coverage(texts)
        scores["style_diversity"] = metrics.compute_style_diversity(texts)
        scores["length_distribution"] = metrics.compute_length_distribution_diversity(texts)
        scores["opening_diversity"] = metrics.compute_opening_diversity(texts)
        scores["strategy_diversity"] = metrics.compute_strategy_diversity(texts)

        # Also include evaluator-level diversity scores
        scores["topic_diversity"] = self.evaluate_topic_diversity(texts)
        scores["response_diversity"] = self.evaluate_response_diversity(texts)

        metric_values = list(scores.values())
        scores["overall_diversity"] = float(np.mean(metric_values)) if metric_values else 0.0

        return scores


# ---------------------------------------------------------------------------
# DialogueDiversityMetrics
# ---------------------------------------------------------------------------


class DialogueDiversityMetrics:
    """Computes diversity metrics across multiple dialogue generation outputs.

    Measures variety in responses, topics, styles, lengths, openings,
    and conversational strategies.
    """

    def compute_response_variety(
        self, texts: List[str]
    ) -> float:
        """Compute variety in response patterns across dialogue outputs.

        Measures unique n-gram patterns, lexical diversity, and structural
        uniqueness across multiple generated dialogues.

        Parameters
        ----------
        texts : List[str]
            Multiple dialogue generation outputs.

        Returns
        -------
        float
            Response variety score in [0, 1].
        """
        if len(texts) < 2:
            return 0.0

        # Unique trigrams across all texts
        all_trigrams: Set[Tuple[str, ...]] = set()
        per_text_trigrams: List[Set[Tuple[str, ...]]] = []

        for text in texts:
            tokens = _tokenize(text)
            trigrams = set(_ngrams(tokens, 3))
            per_text_trigrams.append(trigrams)
            all_trigrams.update(trigrams)

        if not all_trigrams:
            return 0.0

        # Unique trigram ratio
        total_trigrams = sum(len(tg) for tg in per_text_trigrams)
        unique_ratio = len(all_trigrams) / max(total_trigrams, 1)

        # Pairwise trigram Jaccard distances
        pairwise_distances: List[float] = []
        for i in range(len(per_text_trigrams)):
            for j in range(i + 1, len(per_text_trigrams)):
                tg_i = {" ".join(t) for t in per_text_trigrams[i]}
                tg_j = {" ".join(t) for t in per_text_trigrams[j]}
                sim = _jaccard_similarity(tg_i, tg_j)
                pairwise_distances.append(1.0 - sim)

        avg_distance = float(np.mean(pairwise_distances)) if pairwise_distances else 0.0

        # Unique sentence starts
        first_words: List[str] = []
        for text in texts:
            sentences = _sentences(text)
            for sent in sentences[:3]:
                words = sent.lower().split()
                if words:
                    first_words.append(words[0])

        first_word_diversity = _type_token_ratio(first_words) if first_words else 0.0

        variety = (
            0.35 * unique_ratio
            + 0.40 * avg_distance
            + 0.25 * first_word_diversity
        )
        return _clamp(variety)

    def compute_topic_coverage(
        self, texts: List[str]
    ) -> float:
        """Compute how many different topics are addressed across outputs.

        Parameters
        ----------
        texts : List[str]
            Multiple dialogue generation outputs.

        Returns
        -------
        float
            Topic coverage score in [0, 1].
        """
        if not texts:
            return 0.0

        all_topics_found: Set[str] = set()
        per_text_topics: List[Set[str]] = []

        for text in texts:
            text_lower = text.lower()
            text_tokens = set(_tokenize(text_lower))
            found_topics: Set[str] = set()

            for topic_name, keywords in _TOPIC_KEYWORDS.items():
                keyword_set = set(keywords)
                matches = len(text_tokens & keyword_set)
                if matches >= 2:
                    found_topics.add(topic_name)

            per_text_topics.append(found_topics)
            all_topics_found.update(found_topics)

        # Overall coverage
        total_possible = len(_TOPIC_KEYWORDS)
        coverage_ratio = len(all_topics_found) / total_possible if total_possible > 0 else 0.0

        # Per-text uniqueness — ideally each text covers different topics
        if len(per_text_topics) > 1:
            unique_per_text: List[int] = []
            for i, topics in enumerate(per_text_topics):
                others = set()
                for j, other_topics in enumerate(per_text_topics):
                    if j != i:
                        others.update(other_topics)
                unique_to_this = topics - others
                unique_per_text.append(len(unique_to_this))
            uniqueness_score = sum(unique_per_text) / max(
                sum(len(t) for t in per_text_topics), 1
            )
        else:
            uniqueness_score = 0.0

        return _clamp(0.6 * coverage_ratio * 2.0 + 0.4 * uniqueness_score * 3.0)

    def compute_style_diversity(
        self, texts: List[str]
    ) -> float:
        """Compute variety in writing styles across dialogue outputs.

        Measures variation between formal, informal, and emotional
        registers across the generated dialogues.

        Parameters
        ----------
        texts : List[str]
            Multiple dialogue generation outputs.

        Returns
        -------
        float
            Style diversity score in [0, 1].
        """
        if len(texts) < 2:
            return 0.0

        style_profiles: List[Dict[str, float]] = []

        for text in texts:
            text_lower = text.lower()
            total_words = max(len(text_lower.split()), 1)

            # Formality
            formal_count = _count_pattern_hits(text_lower, _FORMAL_MARKERS)
            casual_count = _count_pattern_hits(text_lower, _CASUAL_MARKERS)

            formality = formal_count / total_words
            casualness = casual_count / total_words

            # Emotionality
            emotion_count = 0
            for emotion_words in _EMOTIONAL_MARKERS.values():
                for word in emotion_words:
                    if word in text_lower:
                        emotion_count += 1
            emotionality = emotion_count / total_words

            # Question density
            question_count = text.count("?")
            question_density = question_count / total_words

            # Exclamation density
            exclamation_count = text.count("!")
            exclamation_density = exclamation_count / total_words

            # Average sentence length
            sentences = _sentences(text)
            avg_sent_len = (
                float(np.mean([len(s.split()) for s in sentences]))
                if sentences else 10.0
            )

            style_profiles.append({
                "formality": formality,
                "casualness": casualness,
                "emotionality": emotionality,
                "question_density": question_density,
                "exclamation_density": exclamation_density,
                "avg_sent_len": avg_sent_len / 30.0,  # normalize
            })

        # Compute variance in each style dimension
        dimension_variances: List[float] = []
        for key in style_profiles[0]:
            values = [p[key] for p in style_profiles]
            if len(values) > 1:
                variance = float(np.std(values))
                dimension_variances.append(min(1.0, variance * 10.0))

        # Style diversity is the average variance across dimensions
        if dimension_variances:
            return _clamp(float(np.mean(dimension_variances)))
        return 0.0

    def compute_length_distribution_diversity(
        self, texts: List[str]
    ) -> float:
        """Compute diversity in dialogue lengths (turns and words).

        Parameters
        ----------
        texts : List[str]
            Multiple dialogue generation outputs.

        Returns
        -------
        float
            Length distribution diversity in [0, 1].
        """
        if len(texts) < 2:
            return 0.0

        word_counts: List[int] = []
        turn_counts: List[int] = []

        for text in texts:
            word_counts.append(len(text.split()))
            turns = parse_dialogue_format(text)
            turn_counts.append(len(turns))

        # Word count diversity (normalized CV)
        mean_wc = float(np.mean(word_counts))
        if mean_wc > 0:
            wc_cv = float(np.std(word_counts)) / mean_wc
            wc_diversity = _clamp(wc_cv)
        else:
            wc_diversity = 0.0

        # Turn count diversity
        mean_tc = float(np.mean(turn_counts))
        if mean_tc > 0:
            tc_cv = float(np.std(turn_counts)) / mean_tc
            tc_diversity = _clamp(tc_cv)
        else:
            tc_diversity = 0.0

        # Entropy of binned lengths
        bins = [0, 50, 100, 200, 400, 800, float("inf")]
        bin_counts = [0] * (len(bins) - 1)
        for wc in word_counts:
            for b in range(len(bins) - 1):
                if bins[b] <= wc < bins[b + 1]:
                    bin_counts[b] += 1
                    break

        length_entropy = _normalized_entropy(bin_counts)

        return _clamp(
            0.35 * wc_diversity + 0.35 * tc_diversity + 0.30 * length_entropy
        )

    def compute_opening_diversity(
        self, texts: List[str]
    ) -> float:
        """Compute variety of conversation openers across outputs.

        Measures uniqueness in how dialogues begin — first turn text,
        opening dialogue act, and conversation setup.

        Parameters
        ----------
        texts : List[str]
            Multiple dialogue generation outputs.

        Returns
        -------
        float
            Opening diversity score in [0, 1].
        """
        if len(texts) < 2:
            return 0.0

        first_utterances: List[str] = []
        first_acts: List[str] = []
        first_words: List[str] = []

        for text in texts:
            turns = parse_dialogue_format(text)
            if turns:
                utterance = turns[0].utterance.lower().strip()
                first_utterances.append(utterance)

                acts = extract_dialogue_acts(turns[0].utterance)
                first_acts.append(acts[0].name if acts else "STATEMENT")

                words = utterance.split()
                if words:
                    first_words.append(words[0])
            else:
                lines = [l.strip().lower() for l in text.split("\n") if l.strip()]
                if lines:
                    first_utterances.append(lines[0])
                    first_words.append(lines[0].split()[0] if lines[0].split() else "")

        # Unique first utterances
        unique_openings = len(set(first_utterances))
        opening_ratio = unique_openings / len(first_utterances) if first_utterances else 0.0

        # Diversity of opening acts
        if first_acts:
            act_counts = Counter(first_acts)
            act_entropy = _normalized_entropy(list(act_counts.values()))
        else:
            act_entropy = 0.0

        # First word diversity
        if first_words:
            word_diversity = _type_token_ratio(first_words)
        else:
            word_diversity = 0.0

        # Pairwise edit distance of openings (normalized)
        pairwise_diffs: List[float] = []
        for i in range(len(first_utterances)):
            for j in range(i + 1, len(first_utterances)):
                max_len = max(len(first_utterances[i]), len(first_utterances[j]), 1)
                # Simple character-level difference ratio
                common = sum(
                    1 for a, b in zip(first_utterances[i], first_utterances[j])
                    if a == b
                )
                diff = 1.0 - common / max_len
                pairwise_diffs.append(diff)

        avg_diff = float(np.mean(pairwise_diffs)) if pairwise_diffs else 0.0

        return _clamp(
            0.30 * opening_ratio
            + 0.20 * act_entropy
            + 0.20 * word_diversity
            + 0.30 * avg_diff
        )

    def compute_strategy_diversity(
        self, texts: List[str]
    ) -> float:
        """Compute diversity in conversational strategies used.

        Measures the mix of questions, statements, acknowledgments, and
        other dialogue acts across generated dialogues.

        Parameters
        ----------
        texts : List[str]
            Multiple dialogue generation outputs.

        Returns
        -------
        float
            Strategy diversity score in [0, 1].
        """
        if not texts:
            return 0.0

        # Per-text strategy profiles
        strategy_profiles: List[Dict[str, float]] = []

        for text in texts:
            turns = parse_dialogue_format(text)
            if not turns:
                strategy_profiles.append({})
                continue

            act_counts: Counter = Counter()
            total_acts = 0
            for turn in turns:
                acts = extract_dialogue_acts(turn.utterance)
                for act in acts:
                    act_counts[act.name] += 1
                    total_acts += 1

            if total_acts > 0:
                profile = {
                    act_name: count / total_acts
                    for act_name, count in act_counts.items()
                }
            else:
                profile = {}
            strategy_profiles.append(profile)

        # Filter out empty profiles
        valid_profiles = [p for p in strategy_profiles if p]
        if len(valid_profiles) < 2:
            return 0.0

        # Compute pairwise distance between strategy profiles
        all_act_names = set()
        for p in valid_profiles:
            all_act_names.update(p.keys())

        pairwise_distances: List[float] = []
        for i in range(len(valid_profiles)):
            for j in range(i + 1, len(valid_profiles)):
                vec_i = np.array(
                    [valid_profiles[i].get(a, 0.0) for a in all_act_names]
                )
                vec_j = np.array(
                    [valid_profiles[j].get(a, 0.0) for a in all_act_names]
                )
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                if norm_i > 0 and norm_j > 0:
                    cos_sim = float(np.dot(vec_i, vec_j) / (norm_i * norm_j))
                    pairwise_distances.append(1.0 - cos_sim)
                else:
                    pairwise_distances.append(1.0)

        avg_distance = float(np.mean(pairwise_distances)) if pairwise_distances else 0.0

        # Also measure the overall act distribution entropy
        combined_acts: Counter = Counter()
        for p in valid_profiles:
            for act_name, prob in p.items():
                combined_acts[act_name] += 1

        overall_entropy = _normalized_entropy(list(combined_acts.values()))

        # Number of unique acts used across all dialogues
        unique_acts = len(combined_acts)
        total_possible_acts = len(DialogueAct)
        act_coverage = unique_acts / total_possible_acts

        return _clamp(
            0.40 * avg_distance
            + 0.30 * overall_entropy
            + 0.30 * act_coverage
        )


# ---------------------------------------------------------------------------
# DialogueTask
# ---------------------------------------------------------------------------


@GenerationTask.register("dialogue")
class DialogueTask(GenerationTask):
    """Full dialogue generation task implementation.

    Bundles prompt generation, formatting, constraint production,
    multi-dimensional evaluation for both quality and diversity,
    and post-processing for dialogue text.  Supports multiple dialogue
    types including open-domain, task-oriented, debate, interview,
    roleplay, and persona-based dialogues.
    """

    def __init__(
        self,
        config: Optional[DialogueConfig] = None,
        *,
        seed: int = 42,
    ) -> None:
        self.config = config or DialogueConfig(
            name="dialogue",
            domain=TaskDomain.DIALOGUE,
        )
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._prompt_generator = DialoguePromptGenerator(seed=seed)
        self._evaluator = DialogueEvaluator(config=self.config)
        self._diversity_metrics = DialogueDiversityMetrics()
        self._dataset: Optional[PromptDataset] = None

    # ------------------------------------------------------------------
    # Default config
    # ------------------------------------------------------------------

    @classmethod
    def get_default_config(cls) -> DialogueConfig:
        """Return a sensible default :class:`DialogueConfig`."""
        return DialogueConfig(
            name="dialogue",
            domain=TaskDomain.DIALOGUE,
            num_prompts=50,
            max_length=1024,
            min_length=20,
            temperature=0.9,
            evaluation_metrics=[
                "coherence",
                "engagement",
                "turn_taking",
                "informativeness",
                "naturalness",
                "contradiction_avoidance",
                "topic_diversity",
                "response_diversity",
            ],
        )

    # ------------------------------------------------------------------
    # Prompt loading
    # ------------------------------------------------------------------

    def load_prompts(self) -> PromptDataset:
        """Return a :class:`PromptDataset` with diverse dialogue prompts."""
        prompts = self._prompt_generator.generate_all_prompts()
        logger.info("Loaded %d dialogue prompts", len(prompts))
        return PromptDataset(
            prompts=prompts,
            name="dialogue_builtin",
            domain=TaskDomain.DIALOGUE,
        )

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_prompt(self, prompt: TaskPrompt) -> str:
        """Format *prompt* into the final string sent to the model.

        Applies dialogue-specific formatting including system instructions,
        context, and generation guidelines.
        """
        header = "=== Dialogue Generation Task ==="
        body = prompt.text if isinstance(prompt.text, str) else str(prompt)

        dialogue_type = prompt.metadata.get("dialogue_type", "open_domain")

        sections: List[str] = [header, ""]

        # Add type-specific instructions
        type_instructions = {
            "open_domain": (
                "Generate a natural, flowing conversation. Speakers should "
                "build on each other's comments and ask follow-up questions."
            ),
            "task_oriented": (
                "Generate a goal-directed dialogue. The conversation should "
                "progress toward completing the stated task with realistic "
                "complications and resolutions."
            ),
            "debate": (
                "Generate a structured debate. Each speaker should present "
                "well-reasoned arguments and directly address the opponent's "
                "points. Maintain a respectful tone."
            ),
            "interview": (
                "Generate an interview dialogue. The interviewer should ask "
                "probing, open-ended questions. The interviewee should give "
                "detailed, insightful responses."
            ),
            "roleplay": (
                "Generate an in-character dialogue. Each character should "
                "have a distinct voice that reflects their personality and "
                "background. Include action descriptions in brackets."
            ),
            "persona_based": (
                "Generate a dialogue where the character's speech reflects "
                "their described personality, occupation, and speech style. "
                "Stay consistent with the character profile."
            ),
        }

        instruction = type_instructions.get(dialogue_type, type_instructions["open_domain"])
        sections.append(f"Instructions: {instruction}")
        sections.append("")

        # Add format guidelines
        sections.append("Format: Use 'Speaker Name: utterance' format.")
        sections.append(f"Minimum turns: {self.config.min_turns}")
        sections.append(f"Maximum turns: {self.config.max_turns}")
        sections.append("")

        sections.append(body)
        sections.append("")

        # Add constraints
        constraints_text: List[str] = []
        if self.config.min_utterance_words > 0:
            constraints_text.append(
                f"Each utterance should be at least {self.config.min_utterance_words} words."
            )
        if self.config.max_utterance_words < 10000:
            constraints_text.append(
                f"Each utterance should be at most {self.config.max_utterance_words} words."
            )
        if self.config.require_turn_markers:
            constraints_text.append("Each turn must begin with the speaker's name followed by a colon.")

        if constraints_text:
            sections.append("Constraints:")
            for c in constraints_text:
                sections.append(f"  - {c}")

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def get_constraints(self) -> List[TaskConstraint]:
        """Return dialogue-specific constraints."""
        cfg = self.config
        constraints: List[TaskConstraint] = []

        # Length constraint
        constraints.append(TaskConstraint(
            constraint_type=ConstraintType.LENGTH,
            parameters={
                "min": cfg.min_length,
                "max": cfg.max_length,
                "unit": "words",
            },
            required=True,
            weight=1.0,
        ))

        # Format constraint — dialogue should have turn markers
        if cfg.require_turn_markers:
            constraints.append(TaskConstraint(
                constraint_type=ConstraintType.FORMAT,
                parameters={
                    "pattern": r'^[A-Za-z][\w\s\'-]*:',
                },
                required=False,
                weight=0.8,
            ))

        # Content constraint — dialogue should not be empty or repetitive
        constraints.append(TaskConstraint(
            constraint_type=ConstraintType.CONTENT,
            parameters={
                "max_repetition_ratio": 0.15,
                "min_unique_words": 10,
            },
            required=True,
            weight=0.9,
        ))

        # Style constraint — utterances should not be excessively long
        constraints.append(TaskConstraint(
            constraint_type=ConstraintType.STYLE,
            parameters={
                "max_sentence_length": cfg.max_utterance_words,
            },
            required=False,
            weight=0.5,
        ))

        return constraints

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, generations: List[str], prompts: List[TaskPrompt]
    ) -> Dict[str, Any]:
        """Score *generations* against their *prompts*.

        Returns a dictionary with per-generation scores and corpus-level
        aggregated metrics for both quality and diversity.
        """
        per_gen: List[Dict[str, float]] = []

        for text, prompt in zip(generations, prompts):
            persona = prompt.metadata.get("persona")
            context = prompt.context or ""

            quality_scores = self._evaluator.compute_dialogue_quality_score(
                text, context=context, persona=persona
            )
            per_gen.append(quality_scores)

        # Corpus-level quality aggregation
        all_keys: Set[str] = set()
        for s in per_gen:
            all_keys.update(s.keys())

        corpus_quality: Dict[str, float] = {}
        for k in sorted(all_keys):
            vals = [s[k] for s in per_gen if k in s and isinstance(s[k], float)]
            if vals:
                corpus_quality[f"{k}_mean"] = float(np.mean(vals))
                corpus_quality[f"{k}_std"] = float(np.std(vals))

        # Corpus-level diversity
        diversity_scores = self._evaluator.compute_dialogue_diversity_score(generations)

        # Combined result
        result: Dict[str, Any] = {
            "corpus_quality": corpus_quality,
            "corpus_diversity": diversity_scores,
            "per_generation": per_gen,
        }

        # Overall score combining quality and diversity
        quality_val = corpus_quality.get("overall_quality_mean", 0.5)
        diversity_val = diversity_scores.get("overall_diversity", 0.5)
        result["overall_score"] = (
            self.config.quality_weight * quality_val
            + self.config.diversity_weight * diversity_val
        )

        return result

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def format_output(
        self, generation: str, prompt: TaskPrompt
    ) -> Dict[str, Any]:
        """Format a single generation with its evaluation for output.

        Parameters
        ----------
        generation : str
            The generated dialogue text.
        prompt : TaskPrompt
            The prompt that produced the generation.

        Returns
        -------
        Dict[str, Any]
            Formatted output including parsed turns, evaluation scores,
            and metadata.
        """
        turns = parse_dialogue_format(generation)

        # Quality scores
        persona = prompt.metadata.get("persona")
        context = prompt.context or ""
        quality_scores = self._evaluator.compute_dialogue_quality_score(
            generation, context=context, persona=persona
        )

        # Dialogue analysis
        patterns = detect_conversation_patterns(turns)

        # Validation
        is_valid, reasons = self.validate_generation(generation, prompt)

        return {
            "prompt_id": prompt.prompt_id,
            "prompt_text": prompt.text[:200] + "..." if len(prompt.text) > 200 else prompt.text,
            "dialogue_type": prompt.metadata.get("dialogue_type", "unknown"),
            "generation": generation,
            "parsed_turns": [t.to_dict() for t in turns],
            "num_turns": len(turns),
            "num_speakers": len({t.speaker for t in turns}),
            "word_count": _word_count(generation),
            "quality_scores": quality_scores,
            "conversation_patterns": patterns,
            "is_valid": is_valid,
            "validation_issues": reasons,
        }

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def post_process(self, text: str) -> str:
        """Apply dialogue-specific post-processing to generated text.

        Cleans up formatting, normalises speaker labels, and removes
        artefacts while preserving dialogue structure.
        """
        # Strip leading/trailing whitespace
        text = text.strip()

        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)

        # Normalize common speaker label formats
        # Convert [Speaker] to Speaker:
        text = re.sub(
            r'^\[([A-Za-z][\w\s\'-]*?)\]\s*',
            r'\1: ',
            text,
            flags=re.MULTILINE,
        )

        # Remove trailing incomplete turns
        lines = text.split("\n")
        cleaned_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                # Keep lines that have content
                cleaned_lines.append(line)
            elif cleaned_lines:
                # Keep single blank lines for readability
                if cleaned_lines[-1].strip():
                    cleaned_lines.append("")

        # Remove trailing blank lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()

        return "\n".join(cleaned_lines)

    # ------------------------------------------------------------------
    # Metric names
    # ------------------------------------------------------------------

    def get_metric_names(self) -> List[str]:
        """Return the list of metric names this task evaluates."""
        base = list(self.config.evaluation_metrics)
        dialogue_metrics = [
            "coherence",
            "engagement",
            "persona_consistency",
            "topic_diversity",
            "response_diversity",
            "turn_taking",
            "informativeness",
            "naturalness",
            "contradiction_avoidance",
            "overall_quality",
            "overall_diversity",
            "response_variety",
            "topic_coverage",
            "style_diversity",
            "length_distribution",
            "opening_diversity",
            "strategy_diversity",
        ]
        all_metrics = list(dict.fromkeys(base + dialogue_metrics))
        return all_metrics

    # ------------------------------------------------------------------
    # Describe
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Return a human-readable description of this dialogue task."""
        return (
            f"Task: {self.config.name}\n"
            f"Domain: {self.config.domain.name}\n"
            f"Type: {self.config.dialogue_type.name}\n"
            f"Description: Dialogue generation with quality and diversity evaluation\n"
            f"Prompts: {self.config.num_prompts}\n"
            f"Turns: [{self.config.min_turns}, {self.config.max_turns}]\n"
            f"Speakers: up to {self.config.max_speakers}\n"
            f"Temperature: {self.config.temperature}\n"
            f"Constraints: {len(self.get_constraints())}\n"
            f"Metrics: {', '.join(self.config.evaluation_metrics)}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a machine-readable summary of this task configuration."""
        result: Dict[str, Any] = {
            "task_class": self.__class__.__name__,
            "config": self.config.to_dict(),
            "constraints": [c.to_dict() for c in self.get_constraints()],
            "metrics": self.get_metric_names(),
            "dialogue_types_supported": [dt.name for dt in DialogueType],
            "num_builtin_scenarios": (
                len(_OPEN_DOMAIN_SCENARIOS)
                + len(_TASK_ORIENTED_SCENARIOS)
                + len(_DEBATE_SCENARIOS)
                + len(_INTERVIEW_SCENARIOS)
                + len(_ROLEPLAY_SCENARIOS)
                + len(_ADDITIONAL_SCENARIOS)
            ),
        }
        if self._dataset is not None:
            result["dataset_size"] = len(self._dataset)
        return result


# ---------------------------------------------------------------------------
# DialogueDiversityAnalyzer
# ---------------------------------------------------------------------------


class DialogueDiversityAnalyzer:
    """Advanced diversity analysis across multiple dialogue generations.

    Provides fine-grained metrics for multi-turn diversity, persona
    consistency, emotional variety, pragmatic diversity, topic steering,
    and overall dialogue quality.  All public methods return scores
    normalised to [0, 1] unless otherwise noted.
    """

    # ------------------------------------------------------------------
    # Turn extraction helpers
    # ------------------------------------------------------------------

    _TURN_SEPARATOR_RE = re.compile(
        r"(?:^|\n)\s*(?:"
        r"(?:User|Assistant|Human|AI|Agent|Bot|Speaker\s*\w*|A|B)\s*:"
        r"|>>>"
        r"|\[(?:User|Assistant|Human|AI|Agent|Bot|Speaker\s*\w*)\]"
        r")\s*",
        re.IGNORECASE,
    )

    @staticmethod
    def _extract_turns(dialogue: str) -> List[str]:
        """Parse a dialogue string into individual turns.

        Splits on common speaker label patterns such as ``User:``,
        ``Assistant:``, ``Speaker 1:``, or ``>>>``.  Falls back to
        splitting on double-newlines if no labels are found.

        Parameters
        ----------
        dialogue : str
            Raw dialogue text.

        Returns
        -------
        List[str]
            Non-empty turns extracted from the dialogue.
        """
        parts = DialogueDiversityAnalyzer._TURN_SEPARATOR_RE.split(dialogue)
        turns = [p.strip() for p in parts if p and p.strip()]
        if len(turns) <= 1:
            # Fallback: split on blank lines
            turns = [p.strip() for p in dialogue.split("\n\n") if p.strip()]
        return turns

    # ------------------------------------------------------------------
    # Speech act classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_speech_act(utterance: str) -> str:
        """Classify an utterance into its dominant speech act category.

        Uses the existing pattern banks defined at module level to
        determine the primary dialogue act, returning its name as a
        lowercase string.

        Parameters
        ----------
        utterance : str
            A single dialogue utterance.

        Returns
        -------
        str
            Lowercase speech-act label (e.g. ``"question"``, ``"greeting"``).
        """
        acts = extract_dialogue_acts(utterance)
        if acts:
            return acts[0].name.lower()
        return "statement"

    # ------------------------------------------------------------------
    # Emotion detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_emotion(text: str) -> Dict[str, float]:
        """Detect emotion intensities in *text* using keyword matching.

        Scans for markers defined in ``_EMOTIONAL_MARKERS`` and returns
        a normalised distribution over emotion categories.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        Dict[str, float]
            Mapping from emotion name to intensity in [0, 1].  Values
            sum to 1.0 when at least one marker is found; all zeros
            otherwise.
        """
        text_lower = text.lower()
        scores: Dict[str, float] = {}
        for emotion, keywords in _EMOTIONAL_MARKERS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            scores[emotion] = float(count)

        total = sum(scores.values())
        if total > 0.0:
            scores = {k: v / total for k, v in scores.items()}
        return scores

    # ------------------------------------------------------------------
    # Topic shift detection
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_topic_shifts(turns: List[str]) -> List[float]:
        """Compute per-transition topic shift scores for a list of turns.

        Uses the module-level ``compute_topic_shift_score`` to measure
        pairwise topic divergence between consecutive turns.

        Parameters
        ----------
        turns : List[str]
            Ordered turns of a dialogue.

        Returns
        -------
        List[float]
            List of shift scores (length ``len(turns) - 1``).  Each
            value is in [0, 1]; higher means larger topic change.
        """
        if len(turns) < 2:
            return []
        return [
            compute_topic_shift_score(turns[i], turns[i + 1])
            for i in range(len(turns) - 1)
        ]

    # ------------------------------------------------------------------
    # Engagement scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _engagement_score(turns: List[str]) -> float:
        """Estimate how *engaging* a dialogue is.

        Engagement is approximated from:
        * question density (more questions → more engaging),
        * back-and-forth length balance between speakers,
        * presence of acknowledgment / backchannel tokens.

        Parameters
        ----------
        turns : List[str]
            Turns of the dialogue.

        Returns
        -------
        float
            Engagement score in [0, 1].
        """
        if not turns:
            return 0.0

        n = len(turns)

        # Question ratio
        question_count = sum(
            1 for t in turns
            if _count_pattern_hits(t, _QUESTION_PATTERNS) > 0
        )
        question_ratio = question_count / n

        # Length balance — compare even- vs odd-indexed turn lengths
        even_lens = [len(_tokenize(turns[i])) for i in range(0, n, 2)]
        odd_lens = [len(_tokenize(turns[i])) for i in range(1, n, 2)]
        avg_even = float(np.mean(even_lens)) if even_lens else 1.0
        avg_odd = float(np.mean(odd_lens)) if odd_lens else 1.0
        balance = 1.0 - abs(avg_even - avg_odd) / max(avg_even, avg_odd, 1.0)

        # Backchannel / acknowledgment ratio
        ack_count = sum(
            1 for t in turns
            if (
                _count_pattern_hits(t.lower(), _ACKNOWLEDGMENT_PATTERNS) > 0
                or _count_pattern_hits(t.lower(), _BACKCHANNEL_PATTERNS) > 0
            )
        )
        ack_ratio = min(ack_count / max(n, 1), 1.0)

        # Weighted combination
        score = 0.40 * question_ratio + 0.35 * balance + 0.25 * ack_ratio
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Informativeness scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _informativeness_score(turns: List[str]) -> float:
        """Estimate information density across dialogue turns.

        Informativeness is gauged by:
        * average unique-token ratio per turn,
        * named-entity–like capitalised word density,
        * proportion of content words (tokens longer than 3 chars).

        Parameters
        ----------
        turns : List[str]
            Turns of the dialogue.

        Returns
        -------
        float
            Informativeness score in [0, 1].
        """
        if not turns:
            return 0.0

        unique_ratios: List[float] = []
        content_ratios: List[float] = []
        entity_densities: List[float] = []

        for turn in turns:
            tokens = _tokenize(turn)
            if not tokens:
                continue

            n_tok = len(tokens)
            unique_ratios.append(len(set(tokens)) / n_tok)

            # Content words: length > 3 as a rough proxy
            content_count = sum(1 for t in tokens if len(t) > 3)
            content_ratios.append(content_count / n_tok)

            # Capitalised words (crude named-entity proxy)
            words = turn.split()
            cap_count = sum(
                1 for w in words[1:]  # skip first word of sentence
                if w and w[0].isupper() and not w.isupper()
            )
            entity_densities.append(cap_count / max(len(words), 1))

        if not unique_ratios:
            return 0.0

        avg_unique = float(np.mean(unique_ratios))
        avg_content = float(np.mean(content_ratios))
        avg_entity = float(np.mean(entity_densities))

        score = 0.40 * avg_unique + 0.35 * avg_content + 0.25 * min(avg_entity * 5.0, 1.0)
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public: multi-turn diversity
    # ------------------------------------------------------------------

    def measure_multi_turn_diversity(
        self, dialogues: List[str]
    ) -> Dict[str, float]:
        """Measure diversity *across* multiple multi-turn dialogues.

        Computes turn-level n-gram novelty, structural variety (turn
        counts), and pairwise cosine distance of bag-of-words vectors.

        Parameters
        ----------
        dialogues : List[str]
            Each element is a full multi-turn dialogue.

        Returns
        -------
        Dict[str, float]
            Dictionary with keys ``"ngram_novelty"``,
            ``"structural_variety"``, ``"pairwise_distance"``, and
            ``"overall"``, each in [0, 1].
        """
        if len(dialogues) < 2:
            return {
                "ngram_novelty": 0.0,
                "structural_variety": 0.0,
                "pairwise_distance": 0.0,
                "overall": 0.0,
            }

        all_turn_lists = [self._extract_turns(d) for d in dialogues]

        # --- n-gram novelty across dialogues ---
        per_dialogue_trigrams: List[Set[Tuple[str, ...]]] = []
        for turns in all_turn_lists:
            combined_tokens = []
            for t in turns:
                combined_tokens.extend(_tokenize(t))
            per_dialogue_trigrams.append(set(_ngrams(combined_tokens, 3)))

        total_trigrams: Set[Tuple[str, ...]] = set()
        for s in per_dialogue_trigrams:
            total_trigrams.update(s)

        if total_trigrams:
            # Average fraction of *new* trigrams each dialogue adds
            running: Set[Tuple[str, ...]] = set()
            novel_fracs: List[float] = []
            for tset in per_dialogue_trigrams:
                if running:
                    new = tset - running
                    novel_fracs.append(len(new) / max(len(tset), 1))
                running.update(tset)
            ngram_novelty = float(np.mean(novel_fracs)) if novel_fracs else 0.0
        else:
            ngram_novelty = 0.0

        # --- structural variety (distribution of turn counts) ---
        turn_counts = np.array([len(tl) for tl in all_turn_lists], dtype=float)
        if turn_counts.std() > 0:
            cv = float(turn_counts.std() / turn_counts.mean())
            structural_variety = float(np.clip(cv, 0.0, 1.0))
        else:
            structural_variety = 0.0

        # --- pairwise cosine distance of BoW vectors ---
        vocab: Dict[str, int] = {}
        for turns in all_turn_lists:
            for t in turns:
                for tok in _tokenize(t):
                    if tok not in vocab:
                        vocab[tok] = len(vocab)

        if vocab:
            vectors = np.zeros((len(dialogues), len(vocab)), dtype=float)
            for idx, turns in enumerate(all_turn_lists):
                for t in turns:
                    for tok in _tokenize(t):
                        vectors[idx, vocab[tok]] += 1.0

            dists: List[float] = []
            for i, j in combinations(range(len(dialogues)), 2):
                norm_i = np.linalg.norm(vectors[i])
                norm_j = np.linalg.norm(vectors[j])
                if norm_i > 0 and norm_j > 0:
                    dists.append(float(cosine_distance(vectors[i], vectors[j])))
            pairwise_distance = float(np.mean(dists)) if dists else 0.0
        else:
            pairwise_distance = 0.0

        overall = (
            0.40 * ngram_novelty
            + 0.25 * structural_variety
            + 0.35 * pairwise_distance
        )

        return {
            "ngram_novelty": round(ngram_novelty, 4),
            "structural_variety": round(structural_variety, 4),
            "pairwise_distance": round(pairwise_distance, 4),
            "overall": round(float(np.clip(overall, 0.0, 1.0)), 4),
        }

    # ------------------------------------------------------------------
    # Public: persona consistency
    # ------------------------------------------------------------------

    def measure_persona_consistency(
        self,
        dialogues: List[str],
        personas: List[str],
    ) -> Dict[str, float]:
        """Measure persona-diverse yet internally consistent responses.

        For each ``(dialogue, persona)`` pair the method checks how well
        the dialogue adheres to its persona (keyword overlap) and how
        *different* the dialogues are from each other.

        Parameters
        ----------
        dialogues : List[str]
            Generated dialogues, one per persona.
        personas : List[str]
            Persona descriptions, aligned with *dialogues*.

        Returns
        -------
        Dict[str, float]
            ``"avg_consistency"`` — mean persona adherence across
            dialogues; ``"inter_persona_diversity"`` — pairwise
            divergence between persona-specific dialogues;
            ``"combined"`` — harmonic mean of the two.
        """
        if not dialogues or not personas:
            return {
                "avg_consistency": 0.0,
                "inter_persona_diversity": 0.0,
                "combined": 0.0,
            }

        n = min(len(dialogues), len(personas))

        # --- persona adherence ---
        consistencies: List[float] = []
        for idx in range(n):
            persona_tokens = set(_tokenize(personas[idx]))
            dialogue_tokens = set(_tokenize(dialogues[idx]))
            if persona_tokens:
                overlap = len(persona_tokens & dialogue_tokens) / len(persona_tokens)
            else:
                overlap = 0.0
            consistencies.append(overlap)

        avg_consistency = float(np.mean(consistencies))

        # --- inter-persona diversity (pairwise Jaccard distance) ---
        token_sets = [set(_tokenize(d)) for d in dialogues[:n]]
        pair_dists: List[float] = []
        for i, j in combinations(range(n), 2):
            jacc = _jaccard_similarity(token_sets[i], token_sets[j])
            pair_dists.append(1.0 - jacc)

        inter_diversity = float(np.mean(pair_dists)) if pair_dists else 0.0

        # Harmonic mean of consistency and diversity
        if avg_consistency + inter_diversity > 0:
            combined = (
                2.0 * avg_consistency * inter_diversity
                / (avg_consistency + inter_diversity)
            )
        else:
            combined = 0.0

        return {
            "avg_consistency": round(avg_consistency, 4),
            "inter_persona_diversity": round(inter_diversity, 4),
            "combined": round(combined, 4),
        }

    # ------------------------------------------------------------------
    # Public: emotion diversity
    # ------------------------------------------------------------------

    def measure_emotion_diversity(
        self, dialogues: List[str]
    ) -> Dict[str, float]:
        """Measure sentiment and emotion variety across dialogues.

        Aggregates per-dialogue emotion distributions and computes the
        entropy of the overall emotion profile as well as pairwise
        distance between individual distributions.

        Parameters
        ----------
        dialogues : List[str]
            List of dialogue texts.

        Returns
        -------
        Dict[str, float]
            ``"emotion_entropy"`` — normalised Shannon entropy of the
            aggregated emotion distribution; ``"pairwise_divergence"``
            — mean Jensen–Shannon divergence between dialogue pairs;
            ``"dominant_emotion_variety"`` — fraction of distinct
            dominant emotions; ``"overall"`` — weighted combination.
        """
        if not dialogues:
            return {
                "emotion_entropy": 0.0,
                "pairwise_divergence": 0.0,
                "dominant_emotion_variety": 0.0,
                "overall": 0.0,
            }

        emotion_names = sorted(_EMOTIONAL_MARKERS.keys())
        distributions: List[np.ndarray] = []

        for dialogue in dialogues:
            em = self._detect_emotion(dialogue)
            vec = np.array([em.get(e, 0.0) for e in emotion_names], dtype=float)
            distributions.append(vec)

        dist_matrix = np.array(distributions)  # (n_dialogues, n_emotions)

        # Aggregated distribution
        agg = dist_matrix.sum(axis=0)
        agg_total = agg.sum()
        if agg_total > 0:
            agg_normed = agg / agg_total
        else:
            agg_normed = np.ones(len(emotion_names)) / len(emotion_names)

        max_entropy = math.log(len(emotion_names)) if len(emotion_names) > 1 else 1.0
        emotion_entropy = float(scipy_entropy(agg_normed, base=math.e) / max_entropy)

        # Pairwise Jensen-Shannon divergence
        js_divs: List[float] = []
        for i, j in combinations(range(len(distributions)), 2):
            p = distributions[i]
            q = distributions[j]
            p_sum = p.sum()
            q_sum = q.sum()
            if p_sum > 0 and q_sum > 0:
                p_n = p / p_sum
                q_n = q / q_sum
                m = 0.5 * (p_n + q_n)
                jsd = 0.5 * (scipy_entropy(p_n, m, base=math.e)
                             + scipy_entropy(q_n, m, base=math.e))
                js_divs.append(float(jsd))

        pairwise_divergence = float(np.mean(js_divs)) if js_divs else 0.0
        # Normalise JSD (max is ln(2))
        max_jsd = math.log(2.0)
        pairwise_divergence = min(pairwise_divergence / max_jsd, 1.0) if max_jsd > 0 else 0.0

        # Dominant emotion variety
        dominants: Set[str] = set()
        for vec in distributions:
            if vec.sum() > 0:
                dominants.add(emotion_names[int(np.argmax(vec))])
        dominant_variety = len(dominants) / max(len(emotion_names), 1)

        overall = (
            0.35 * emotion_entropy
            + 0.35 * pairwise_divergence
            + 0.30 * dominant_variety
        )

        return {
            "emotion_entropy": round(emotion_entropy, 4),
            "pairwise_divergence": round(pairwise_divergence, 4),
            "dominant_emotion_variety": round(dominant_variety, 4),
            "overall": round(float(np.clip(overall, 0.0, 1.0)), 4),
        }

    # ------------------------------------------------------------------
    # Public: pragmatic diversity
    # ------------------------------------------------------------------

    def measure_pragmatic_diversity(
        self, dialogues: List[str]
    ) -> Dict[str, float]:
        """Measure diversity of speech acts across dialogues.

        Classifies every utterance in each dialogue and computes the
        entropy and coverage of speech-act types.

        Parameters
        ----------
        dialogues : List[str]
            List of dialogue texts.

        Returns
        -------
        Dict[str, float]
            ``"speech_act_entropy"`` — normalised entropy over speech
            act distribution; ``"speech_act_coverage"`` — fraction of
            known speech-act types observed; ``"per_dialogue_variety"``
            — average number of distinct speech acts per dialogue
            (normalised); ``"overall"`` — weighted combination.
        """
        all_act_names = [a.name.lower() for a in DialogueAct]
        n_act_types = len(all_act_names)

        if not dialogues:
            return {
                "speech_act_entropy": 0.0,
                "speech_act_coverage": 0.0,
                "per_dialogue_variety": 0.0,
                "overall": 0.0,
            }

        global_counts: Counter = Counter()
        per_dialogue_unique: List[float] = []

        for dialogue in dialogues:
            turns = self._extract_turns(dialogue)
            dialogue_acts: Set[str] = set()
            for turn in turns:
                act = self._classify_speech_act(turn)
                global_counts[act] += 1
                dialogue_acts.add(act)
            per_dialogue_unique.append(len(dialogue_acts) / max(n_act_types, 1))

        # Entropy of global speech-act distribution
        total = sum(global_counts.values())
        if total > 0:
            probs = np.array([global_counts.get(a, 0) / total for a in all_act_names], dtype=float)
        else:
            probs = np.zeros(n_act_types, dtype=float)

        max_ent = math.log(n_act_types) if n_act_types > 1 else 1.0
        speech_act_entropy = float(scipy_entropy(probs, base=math.e) / max_ent) if max_ent > 0 else 0.0

        # Coverage
        observed = sum(1 for a in all_act_names if global_counts.get(a, 0) > 0)
        speech_act_coverage = observed / max(n_act_types, 1)

        per_dialogue_variety = float(np.mean(per_dialogue_unique)) if per_dialogue_unique else 0.0

        overall = (
            0.35 * speech_act_entropy
            + 0.30 * speech_act_coverage
            + 0.35 * per_dialogue_variety
        )

        return {
            "speech_act_entropy": round(speech_act_entropy, 4),
            "speech_act_coverage": round(speech_act_coverage, 4),
            "per_dialogue_variety": round(per_dialogue_variety, 4),
            "overall": round(float(np.clip(overall, 0.0, 1.0)), 4),
        }

    # ------------------------------------------------------------------
    # Public: topic steering diversity
    # ------------------------------------------------------------------

    def measure_topic_steering_diversity(
        self, dialogues: List[str]
    ) -> Dict[str, float]:
        """Measure how differently topics are introduced and shifted.

        Analyses per-dialogue topic-shift profiles and compares them
        across all supplied dialogues to gauge steering diversity.

        Parameters
        ----------
        dialogues : List[str]
            List of dialogue texts.

        Returns
        -------
        Dict[str, float]
            ``"avg_shift_magnitude"`` — mean topic-shift score across
            all transitions; ``"shift_variance"`` — variance in shift
            magnitudes (higher means more diverse steering);
            ``"topic_coverage"`` — fraction of known topic categories
            mentioned across dialogues; ``"overall"`` — weighted
            combination.
        """
        if not dialogues:
            return {
                "avg_shift_magnitude": 0.0,
                "shift_variance": 0.0,
                "topic_coverage": 0.0,
                "overall": 0.0,
            }

        all_shifts: List[float] = []
        topics_seen: Set[str] = set()

        for dialogue in dialogues:
            turns = self._extract_turns(dialogue)
            shifts = self._compute_topic_shifts(turns)
            all_shifts.extend(shifts)

            # Detect topics present
            full_text_lower = dialogue.lower()
            for topic, keywords in _TOPIC_KEYWORDS.items():
                if any(kw in full_text_lower for kw in keywords):
                    topics_seen.add(topic)

        avg_shift = float(np.mean(all_shifts)) if all_shifts else 0.0
        shift_var = float(np.var(all_shifts)) if all_shifts else 0.0
        # Normalise variance — empirical cap at 0.25 (max var for [0,1] values)
        shift_variance_norm = min(shift_var / 0.25, 1.0)

        n_known_topics = len(_TOPIC_KEYWORDS)
        topic_coverage = len(topics_seen) / max(n_known_topics, 1)

        overall = (
            0.30 * avg_shift
            + 0.35 * shift_variance_norm
            + 0.35 * topic_coverage
        )

        return {
            "avg_shift_magnitude": round(avg_shift, 4),
            "shift_variance": round(shift_variance_norm, 4),
            "topic_coverage": round(topic_coverage, 4),
            "overall": round(float(np.clip(overall, 0.0, 1.0)), 4),
        }

    # ------------------------------------------------------------------
    # Public: dialogue quality
    # ------------------------------------------------------------------

    def compute_dialogue_quality(
        self, dialogue: str
    ) -> Dict[str, float]:
        """Compute quality scores for a single dialogue.

        Returns relevance (intra-turn coherence), informativeness
        (information density), and engagement estimates.

        Parameters
        ----------
        dialogue : str
            Full dialogue text.

        Returns
        -------
        Dict[str, float]
            ``"relevance"``, ``"informativeness"``, ``"engagement"``,
            and ``"overall"`` scores, each in [0, 1].
        """
        turns = self._extract_turns(dialogue)
        if not turns:
            return {
                "relevance": 0.0,
                "informativeness": 0.0,
                "engagement": 0.0,
                "overall": 0.0,
            }

        # --- relevance (coherence between consecutive turns) ---
        if len(turns) >= 2:
            coherence_scores: List[float] = []
            for i in range(len(turns) - 1):
                tokens_a = set(_tokenize(turns[i]))
                tokens_b = set(_tokenize(turns[i + 1]))
                if tokens_a and tokens_b:
                    overlap = len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))
                    coherence_scores.append(overlap)
            relevance = float(np.mean(coherence_scores)) if coherence_scores else 0.5
        else:
            relevance = 0.5

        informativeness = self._informativeness_score(turns)
        engagement = self._engagement_score(turns)

        overall = 0.35 * relevance + 0.35 * informativeness + 0.30 * engagement

        return {
            "relevance": round(float(np.clip(relevance, 0.0, 1.0)), 4),
            "informativeness": round(informativeness, 4),
            "engagement": round(engagement, 4),
            "overall": round(float(np.clip(overall, 0.0, 1.0)), 4),
        }
