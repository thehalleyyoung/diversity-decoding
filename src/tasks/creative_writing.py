"""
Creative writing task domain for the Diversity Decoding Arena.

Implements evaluation of diverse text generation across fiction, poetry,
dialogue, essays, screenplays, flash fiction, fables, and monologues.
Provides rich built-in prompt datasets and fine-grained evaluation metrics
for creativity, coherence, style consistency, and vocabulary richness.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from src.tasks.base import (
    GenerationTask,
    PromptDataset,
    TaskConfig,
    TaskConstraint,
    TaskEvaluator,
    TaskPrompt,
)
from src.types import TaskDomain

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WritingGenre(Enum):
    """Supported creative-writing genres."""

    FICTION = auto()
    POETRY = auto()
    DIALOGUE = auto()
    ESSAY = auto()
    SCREENPLAY = auto()
    FLASH_FICTION = auto()
    FABLE = auto()
    MONOLOGUE = auto()

    def __repr__(self) -> str:
        return f"WritingGenre.{self.name}"


class WritingStyle(Enum):
    """Target prose style for generation."""

    FORMAL = auto()
    CASUAL = auto()
    LITERARY = auto()
    MINIMALIST = auto()
    ORNATE = auto()
    STREAM_OF_CONSCIOUSNESS = auto()

    def __repr__(self) -> str:
        return f"WritingStyle.{self.name}"


# ---------------------------------------------------------------------------
# Word / phrase banks used by the evaluation helpers
# ---------------------------------------------------------------------------

_SENSORY_WORDS: Set[str] = {
    # sight
    "bright", "dark", "gleaming", "shadowy", "vivid", "pale", "glowing",
    "sparkling", "dim", "radiant", "shimmering", "luminous", "dazzling",
    "opaque", "translucent", "crimson", "azure", "golden", "silver",
    "emerald", "scarlet", "ivory", "ebony", "amber", "violet",
    # sound
    "whisper", "roar", "hum", "echo", "buzz", "crash", "murmur",
    "rustle", "clatter", "thunder", "chime", "ring", "screech",
    "crackle", "rumble", "sizzle", "hiss", "thud", "clang", "drone",
    # touch
    "smooth", "rough", "soft", "sharp", "warm", "cold", "silky",
    "prickly", "tender", "coarse", "velvet", "gritty", "damp",
    "sticky", "slimy", "feathery", "bristly", "chilly", "scorching",
    # taste
    "sweet", "bitter", "sour", "salty", "savory", "tangy", "bland",
    "spicy", "tart", "pungent", "zesty", "succulent", "rancid",
    # smell
    "fragrant", "musty", "acrid", "fresh", "stale", "aromatic",
    "pungent", "earthy", "floral", "smoky", "perfumed", "fetid",
}

_FIGURATIVE_MARKERS: List[str] = [
    r"\blike\s+(?:a|an|the)\b",
    r"\bas\s+(?:a|an)\b",
    r"\bas\s+\w+\s+as\b",
    r"\bmetaphor(?:ical(?:ly)?)?\b",
    r"\bimagine\b",
    r"\bas\s+if\b",
    r"\bas\s+though\b",
]

_EMOTION_WORDS: Dict[str, List[str]] = {
    "joy": ["happy", "joy", "delight", "elated", "ecstatic", "bliss",
            "cheerful", "merry", "jubilant", "exuberant", "glad",
            "pleased", "thrilled", "euphoric", "content", "radiant"],
    "sadness": ["sad", "grief", "sorrow", "melancholy", "mourn", "weep",
                "tears", "despair", "heartbreak", "anguish", "misery",
                "woe", "forlorn", "desolate", "gloomy", "somber"],
    "anger": ["angry", "rage", "fury", "wrath", "irritate", "hostile",
              "bitter", "resentment", "outrage", "indignant", "livid",
              "furious", "irate", "seething", "incensed", "venomous"],
    "fear": ["afraid", "terror", "dread", "panic", "anxiety", "horror",
             "fright", "alarm", "apprehension", "trepidation", "phobia",
             "scared", "nervous", "uneasy", "petrified", "trembling"],
    "surprise": ["surprise", "astonish", "amaze", "shock", "startle",
                 "stun", "bewildered", "awe", "wonder", "flabbergast",
                 "dumbfound", "speechless", "gasp", "incredulous"],
    "disgust": ["disgust", "revulsion", "loathe", "repulsive", "abhor",
                "nauseate", "repel", "detest", "contempt", "vile",
                "grotesque", "hideous", "foul", "wretched", "odious"],
    "love": ["love", "adore", "cherish", "devotion", "affection",
             "passion", "tender", "embrace", "yearn", "longing",
             "fondness", "infatuation", "enchant", "captivate"],
    "anticipation": ["anticipate", "expect", "hope", "eager", "await",
                     "suspense", "excitement", "yearn", "impatient",
                     "lookforward", "prospect", "promising"],
}

_FORMAL_MARKERS: List[str] = [
    r"\bfurthermore\b", r"\bmoreover\b", r"\bnevertheless\b",
    r"\bhowever\b", r"\bconsequently\b", r"\bthus\b", r"\bhence\b",
    r"\btherefore\b", r"\bnotwithstanding\b", r"\bwherein\b",
    r"\bherein\b", r"\baforesaid\b", r"\binasmuch\b",
    r"\bone\s+might\s+argue\b", r"\bit\s+is\s+worth\s+noting\b",
]

_CASUAL_MARKERS: List[str] = [
    r"\bgonna\b", r"\bwanna\b", r"\bkinda\b", r"\bsorta\b",
    r"\byeah\b", r"\bnah\b", r"\blol\b", r"\bomg\b",
    r"\bbtw\b", r"\bimo\b", r"\bdude\b", r"\bstuff\b",
    r"\bcool\b", r"\bawesome\b", r"\btotally\b", r"\blike,\b",
]

_LITERARY_MARKERS: List[str] = [
    r"\bephemeral\b", r"\bluminous\b", r"\bmelanchol(?:y|ic)\b",
    r"\bwistful\b", r"\btenebrous\b", r"\bserendipit(?:y|ous)\b",
    r"\bsublime\b", r"\belegy\b", r"\bsoliloquy\b", r"\bpalimpsest\b",
    r"\bchiaroscuro\b", r"\bverisimilitude\b", r"\bliminal\b",
]

_MINIMALIST_INDICATORS: List[str] = [
    r"\.\s",  # many short sentences
]

_NARRATIVE_ARC_BEGINNING: List[str] = [
    r"\bonce\s+upon\b", r"\bin\s+the\s+beginning\b", r"\bit\s+started\b",
    r"\blong\s+ago\b", r"\bthere\s+(?:was|were|lived)\b",
    r"\bthe\s+first\s+time\b", r"\bi\s+remember\b",
    r"\bwhen\s+i\s+was\b", r"\bit\s+all\s+began\b",
    r"\bone\s+day\b", r"\bthe\s+morning\b",
]

_NARRATIVE_ARC_CLIMAX: List[str] = [
    r"\bsuddenly\b", r"\ball\s+at\s+once\b", r"\bin\s+that\s+moment\b",
    r"\bfinally\b", r"\bat\s+last\b", r"\bthe\s+truth\b",
    r"\brealized\b", r"\bturning\s+point\b", r"\beverything\s+changed\b",
    r"\bcrash(?:ed|ing)?\b", r"\bexplo(?:ded|sion)\b",
]

_NARRATIVE_ARC_RESOLUTION: List[str] = [
    r"\bin\s+the\s+end\b", r"\bfrom\s+that\s+day\b",
    r"\band\s+so\b", r"\bpeace\b", r"\bsettled\b",
    r"\blearned\b", r"\bnever\s+again\b", r"\bever\s+after\b",
    r"\bthe\s+end\b", r"\bclosure\b", r"\bresolved\b",
    r"\bfinally\s+understood\b", r"\bat\s+peace\b",
]


# ---------------------------------------------------------------------------
# Data-classes — configs & prompts
# ---------------------------------------------------------------------------


@dataclass
class CreativeWritingConfig(TaskConfig):
    """Configuration for a creative-writing evaluation run."""

    genre: WritingGenre = WritingGenre.FICTION
    style: WritingStyle = WritingStyle.LITERARY
    min_words: int = 100
    max_words: int = 1000
    required_elements: List[str] = field(default_factory=list)
    tone: str = "neutral"
    pov: str = "third"  # first / second / third
    tense: str = "past"  # past / present / future

    def __post_init__(self) -> None:
        if self.min_words < 0:
            raise ValueError("min_words must be non-negative")
        if self.max_words < self.min_words:
            raise ValueError("max_words must be >= min_words")
        if self.pov not in {"first", "second", "third"}:
            raise ValueError(f"Invalid pov: {self.pov!r}")
        if self.tense not in {"past", "present", "future"}:
            raise ValueError(f"Invalid tense: {self.tense!r}")

    def validate(self) -> bool:
        """Return *True* when the config is internally consistent."""
        try:
            self.__post_init__()
            return True
        except ValueError:
            return False


@dataclass
class StoryPrompt(TaskPrompt):
    """A prompt for story / fiction generation."""

    genre: WritingGenre = WritingGenre.FICTION
    setting: str = ""
    characters: List[str] = field(default_factory=list)
    theme: str = ""
    conflict_type: str = ""

    def as_text(self) -> str:
        parts = [f"Write a {self.genre.name.lower().replace('_', ' ')} story"]
        if self.setting:
            parts.append(f"set in {self.setting}")
        if self.characters:
            parts.append(f"featuring {', '.join(self.characters)}")
        if self.theme:
            parts.append(f"exploring the theme of {self.theme}")
        if self.conflict_type:
            parts.append(f"with a {self.conflict_type} conflict")
        return ". ".join(parts) + "."


@dataclass
class PoetryPrompt(TaskPrompt):
    """A prompt for poetry generation."""

    form: str = "free_verse"  # sonnet / haiku / free_verse / limerick
    meter: str = ""
    rhyme_scheme: str = ""

    def as_text(self) -> str:
        parts = [f"Write a {self.form.replace('_', ' ')} poem"]
        if self.meter:
            parts.append(f"in {self.meter} meter")
        if self.rhyme_scheme:
            parts.append(f"with a {self.rhyme_scheme} rhyme scheme")
        return ". ".join(parts) + "."


@dataclass
class DialoguePrompt(TaskPrompt):
    """A prompt for dialogue generation."""

    num_speakers: int = 2
    setting: str = ""
    relationship: str = ""
    topic: str = ""

    def as_text(self) -> str:
        parts = [f"Write a dialogue between {self.num_speakers} speakers"]
        if self.setting:
            parts.append(f"set in {self.setting}")
        if self.relationship:
            parts.append(f"who are {self.relationship}")
        if self.topic:
            parts.append(f"discussing {self.topic}")
        return ". ".join(parts) + "."


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


def _paragraphs(text: str) -> List[str]:
    """Split *text* into paragraphs (double-newline separated)."""
    raw = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in raw if p.strip()]


def _word_count(text: str) -> int:
    return len(_tokenize(text))


def _count_pattern_hits(text: str, patterns: List[str]) -> int:
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text, re.IGNORECASE))
    return total


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _sigmoid(x: float, midpoint: float = 0.0, steepness: float = 1.0) -> float:
    """Numerically-stable sigmoid mapping to [0, 1]."""
    z = steepness * (x - midpoint)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


# ---------------------------------------------------------------------------
# CreativeWritingTask
# ---------------------------------------------------------------------------


class CreativeWritingTask(GenerationTask):
    """Full creative-writing task implementation.

    Bundles prompt generation, formatting, constraint production,
    multi-dimensional evaluation, and post-processing.
    """

    def __init__(
        self,
        config: Optional[CreativeWritingConfig] = None,
        *,
        seed: int = 42,
    ) -> None:
        self.config = config or CreativeWritingConfig()
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Prompt loading
    # ------------------------------------------------------------------

    def load_prompts(self) -> PromptDataset:
        """Return a :class:`PromptDataset` with ≥ 50 built-in prompts."""
        prompts: List[TaskPrompt] = []
        prompts.extend(self._generate_story_prompts())
        prompts.extend(self._generate_poetry_prompts())
        prompts.extend(self._generate_dialogue_prompts())
        return PromptDataset(prompts=prompts, name="creative_writing_builtin")

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_prompt(self, prompt: TaskPrompt) -> str:
        """Format *prompt* into the final string sent to the model."""
        cfg = self.config
        header = self._genre_header(cfg.genre)
        body = prompt.as_text() if hasattr(prompt, "as_text") else str(prompt)

        constraints: List[str] = []
        if cfg.min_words > 0:
            constraints.append(f"Minimum length: {cfg.min_words} words.")
        if cfg.max_words < 100_000:
            constraints.append(f"Maximum length: {cfg.max_words} words.")
        if cfg.pov != "third":
            constraints.append(f"Point of view: {cfg.pov} person.")
        if cfg.tense != "past":
            constraints.append(f"Tense: {cfg.tense}.")
        if cfg.tone != "neutral":
            constraints.append(f"Tone: {cfg.tone}.")
        if cfg.style != WritingStyle.LITERARY:
            constraints.append(f"Style: {cfg.style.name.lower().replace('_', ' ')}.")
        if cfg.required_elements:
            constraints.append(
                "Must include: " + ", ".join(cfg.required_elements) + "."
            )

        sections = [header, "", body]
        if constraints:
            sections.append("")
            sections.append("Constraints:")
            for c in constraints:
                sections.append(f"  - {c}")
        sections.append("")

        # Genre-specific formatting additions
        if isinstance(prompt, StoryPrompt):
            sections.append(self._story_format_addendum(prompt))
        elif isinstance(prompt, PoetryPrompt):
            sections.append(self._poetry_format_addendum(prompt))
        elif isinstance(prompt, DialoguePrompt):
            sections.append(self._dialogue_format_addendum(prompt))

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def get_constraints(self) -> List[TaskConstraint]:
        """Return genre-specific constraints."""
        cfg = self.config
        constraints: List[TaskConstraint] = []

        # Universal word-count constraints
        constraints.append(
            TaskConstraint(
                name="min_word_count",
                description=f"Output must contain at least {cfg.min_words} words.",
                check=lambda text: _word_count(text) >= cfg.min_words,
            )
        )
        constraints.append(
            TaskConstraint(
                name="max_word_count",
                description=f"Output must contain at most {cfg.max_words} words.",
                check=lambda text: _word_count(text) <= cfg.max_words,
            )
        )

        # POV constraint
        if cfg.pov == "first":
            constraints.append(
                TaskConstraint(
                    name="first_person_pov",
                    description="Narrative must use first-person POV.",
                    check=lambda text: self._check_pov(text, "first"),
                )
            )
        elif cfg.pov == "second":
            constraints.append(
                TaskConstraint(
                    name="second_person_pov",
                    description="Narrative must use second-person POV.",
                    check=lambda text: self._check_pov(text, "second"),
                )
            )

        # Genre-specific
        if cfg.genre == WritingGenre.POETRY:
            constraints.append(
                TaskConstraint(
                    name="line_breaks",
                    description="Poetry must contain line breaks.",
                    check=lambda text: "\n" in text.strip(),
                )
            )
        elif cfg.genre == WritingGenre.DIALOGUE:
            constraints.append(
                TaskConstraint(
                    name="dialogue_markers",
                    description="Dialogue must contain speech markers.",
                    check=lambda text: bool(
                        re.search(r'["""\'].*?["""\']', text)
                        or re.search(r"^[A-Z]+:", text, re.MULTILINE)
                    ),
                )
            )
        elif cfg.genre == WritingGenre.SCREENPLAY:
            constraints.append(
                TaskConstraint(
                    name="screenplay_format",
                    description="Screenplay must use character names in caps.",
                    check=lambda text: bool(
                        re.search(r"^[A-Z]{2,}", text, re.MULTILINE)
                    ),
                )
            )
        elif cfg.genre == WritingGenre.FLASH_FICTION:
            constraints.append(
                TaskConstraint(
                    name="flash_fiction_length",
                    description="Flash fiction must be under 500 words.",
                    check=lambda text: _word_count(text) <= 500,
                )
            )
        elif cfg.genre == WritingGenre.FABLE:
            constraints.append(
                TaskConstraint(
                    name="fable_moral",
                    description="Fable should contain an explicit moral.",
                    check=lambda text: bool(
                        re.search(
                            r"moral|lesson|teach|wisdom|learn",
                            text,
                            re.IGNORECASE,
                        )
                    ),
                )
            )

        # Required-elements constraints
        for elem in cfg.required_elements:
            constraints.append(
                TaskConstraint(
                    name=f"required_element_{elem}",
                    description=f"Output must include the element '{elem}'.",
                    check=lambda text, e=elem: e.lower() in text.lower(),
                )
            )

        return constraints

    # ------------------------------------------------------------------
    # Evaluation — main entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        generations: List[str],
        prompts: List[TaskPrompt],
    ) -> Dict[str, Any]:
        """Score *generations* against their *prompts*.

        Returns a dictionary whose keys map to float scores in [0, 1] and
        a ``"per_generation"`` list with per-item score dicts.
        """
        per_gen: List[Dict[str, float]] = []

        for text, prompt in zip(generations, prompts):
            scores: Dict[str, float] = {}

            # Universal metrics
            scores["vocabulary_richness"] = self._vocabulary_richness(text)
            scores["sentence_variety"] = self._sentence_variety(text)
            scores["imagery_density"] = self._imagery_density(text)
            scores["figurative_language"] = self._figurative_language_density(text)
            scores["emotional_range"] = self._emotional_range(text)
            scores["pacing"] = self._pacing_score(text)
            scores["style_consistency"] = self._style_consistency(
                text, self.config.style
            )

            # Genre-specific metrics
            if isinstance(prompt, StoryPrompt):
                scores.update(self._evaluate_story(text, prompt))
            elif isinstance(prompt, PoetryPrompt):
                scores.update(self._evaluate_poetry(text, prompt))
            elif isinstance(prompt, DialoguePrompt):
                scores.update(self._evaluate_dialogue(text, prompt))
            else:
                scores.update(self._evaluate_generic(text, prompt))

            # Aggregate
            metric_vals = [v for v in scores.values() if isinstance(v, float)]
            scores["overall"] = float(np.mean(metric_vals)) if metric_vals else 0.0

            per_gen.append(scores)

        # Corpus-level aggregation
        all_keys: Set[str] = set()
        for s in per_gen:
            all_keys.update(s.keys())
        agg: Dict[str, float] = {}
        for k in sorted(all_keys):
            vals = [s[k] for s in per_gen if k in s and isinstance(s[k], float)]
            if vals:
                agg[k] = float(np.mean(vals))

        return {
            "corpus": agg,
            "per_generation": per_gen,
        }

    # ------------------------------------------------------------------
    # Genre-specific evaluators
    # ------------------------------------------------------------------

    def _evaluate_story(self, text: str, prompt: StoryPrompt) -> Dict[str, float]:
        """Evaluate a piece of fiction / flash-fiction / fable."""
        scores: Dict[str, float] = {}

        scores["narrative_arc"] = self._narrative_arc_score(text)
        scores["coherence"] = self._coherence_score(text)
        scores["creativity"] = self._creativity_score(text)

        # Character presence
        if prompt.characters:
            found = sum(
                1 for c in prompt.characters if c.lower() in text.lower()
            )
            scores["character_presence"] = found / len(prompt.characters)
        else:
            scores["character_presence"] = 1.0

        # Theme relevance — simple keyword overlap
        if prompt.theme:
            theme_words = set(_tokenize(prompt.theme))
            text_words = set(_tokenize(text))
            if theme_words:
                scores["theme_relevance"] = _clamp(
                    len(theme_words & text_words) / len(theme_words)
                )
            else:
                scores["theme_relevance"] = 0.5
        else:
            scores["theme_relevance"] = 1.0

        # Setting adherence
        if prompt.setting:
            setting_words = set(_tokenize(prompt.setting))
            text_words = set(_tokenize(text))
            if setting_words:
                scores["setting_adherence"] = _clamp(
                    len(setting_words & text_words) / len(setting_words)
                )
            else:
                scores["setting_adherence"] = 0.5
        else:
            scores["setting_adherence"] = 1.0

        # Multi-character voice distinction
        if prompt.characters and len(prompt.characters) > 1:
            scores["character_voice_distinction"] = (
                self._character_voice_distinction(text)
            )

        return scores

    def _evaluate_poetry(self, text: str, prompt: PoetryPrompt) -> Dict[str, float]:
        """Evaluate a poem for meter, rhyme, and imagery."""
        scores: Dict[str, float] = {}

        scores["imagery"] = self._imagery_density(text)
        scores["creativity"] = self._creativity_score(text)
        scores["line_count_appropriateness"] = self._line_count_score(
            text, prompt.form
        )

        # Meter regularity
        scores["meter_regularity"] = self._meter_regularity(text, prompt.meter)

        # Rhyme adherence
        scores["rhyme_adherence"] = self._rhyme_adherence(text, prompt.rhyme_scheme)

        # Compactness — poetry should be concise
        wc = _word_count(text)
        if prompt.form == "haiku":
            scores["compactness"] = _clamp(1.0 - abs(wc - 17) / 17.0)
        elif prompt.form == "sonnet":
            scores["compactness"] = _clamp(1.0 - abs(wc - 120) / 120.0)
        elif prompt.form == "limerick":
            scores["compactness"] = _clamp(1.0 - abs(wc - 35) / 35.0)
        else:
            scores["compactness"] = _clamp(_sigmoid(wc, midpoint=80, steepness=-0.02))

        # Emotional intensity — poetry should evoke emotion
        scores["emotional_intensity"] = self._emotional_range(text)

        return scores

    def _evaluate_dialogue(
        self, text: str, prompt: DialoguePrompt
    ) -> Dict[str, float]:
        """Evaluate a dialogue for naturalness and turn-taking."""
        scores: Dict[str, float] = {}

        scores["naturalness"] = self._dialogue_naturalness(text)
        scores["turn_taking"] = self._turn_taking_score(text, prompt.num_speakers)
        scores["creativity"] = self._creativity_score(text)
        scores["coherence"] = self._coherence_score(text)

        # Topic adherence
        if prompt.topic:
            topic_words = set(_tokenize(prompt.topic))
            text_words = set(_tokenize(text))
            if topic_words:
                scores["topic_adherence"] = _clamp(
                    len(topic_words & text_words) / len(topic_words)
                )
            else:
                scores["topic_adherence"] = 0.5
        else:
            scores["topic_adherence"] = 1.0

        # Speaker distinction
        if prompt.num_speakers > 1:
            scores["character_voice_distinction"] = (
                self._character_voice_distinction(text)
            )

        # Relationship plausibility — heuristic: relationship keywords present
        if prompt.relationship:
            rel_words = set(_tokenize(prompt.relationship))
            text_words = set(_tokenize(text))
            scores["relationship_plausibility"] = _clamp(
                len(rel_words & text_words) / max(len(rel_words), 1)
            )
        else:
            scores["relationship_plausibility"] = 1.0

        return scores

    def _evaluate_generic(self, text: str, prompt: TaskPrompt) -> Dict[str, float]:
        """Fallback evaluator for essay / monologue / screenplay."""
        scores: Dict[str, float] = {}
        scores["coherence"] = self._coherence_score(text)
        scores["creativity"] = self._creativity_score(text)
        scores["narrative_arc"] = self._narrative_arc_score(text)
        return scores

    # ------------------------------------------------------------------
    # Metric implementations — all return float in [0, 1]
    # ------------------------------------------------------------------

    def _vocabulary_richness(self, text: str) -> float:
        """Type-token ratio combined with hapax-legomena ratio."""
        tokens = _tokenize(text)
        if len(tokens) < 2:
            return 0.0
        n = len(tokens)
        types = set(tokens)
        v = len(types)
        ttr = v / n  # type-token ratio

        freq = Counter(tokens)
        hapax = sum(1 for w, c in freq.items() if c == 1)
        hapax_ratio = hapax / v if v else 0.0

        # Corrected TTR (Guiraud's index scaled to [0,1])
        guiraud = v / math.sqrt(n)
        guiraud_norm = _clamp(guiraud / 15.0)

        # Yule's K (inverted so higher = richer)
        freq_spectrum = Counter(freq.values())
        m1 = n
        m2 = sum(i * i * vi for i, vi in freq_spectrum.items())
        if m1 > 0 and m1 != 1:
            yule_k = 10000.0 * (m2 - m1) / (m1 * m1)
            yule_norm = _clamp(1.0 - _sigmoid(yule_k, midpoint=100, steepness=0.02))
        else:
            yule_norm = 0.5

        return _clamp(0.35 * ttr + 0.25 * hapax_ratio + 0.2 * guiraud_norm + 0.2 * yule_norm)

    def _narrative_arc_score(self, text: str) -> float:
        """Detect presence of beginning, middle (climax), and resolution."""
        text_lower = text.lower()
        n_paras = max(len(_paragraphs(text)), 1)

        # Split text into thirds
        sents = _sentences(text)
        if len(sents) < 3:
            return 0.2

        third = max(len(sents) // 3, 1)
        beginning = " ".join(sents[:third])
        middle = " ".join(sents[third : 2 * third])
        ending = " ".join(sents[2 * third :])

        begin_hits = _count_pattern_hits(beginning, _NARRATIVE_ARC_BEGINNING)
        climax_hits = _count_pattern_hits(middle, _NARRATIVE_ARC_CLIMAX)
        resol_hits = _count_pattern_hits(ending, _NARRATIVE_ARC_RESOLUTION)

        begin_score = _clamp(_sigmoid(begin_hits, midpoint=0.5, steepness=2.0))
        climax_score = _clamp(_sigmoid(climax_hits, midpoint=0.5, steepness=2.0))
        resol_score = _clamp(_sigmoid(resol_hits, midpoint=0.5, steepness=2.0))

        # Paragraph progression bonus
        para_bonus = _clamp(n_paras / 5.0) * 0.1

        return _clamp(
            0.3 * begin_score + 0.35 * climax_score + 0.25 * resol_score + para_bonus
        )

    def _imagery_density(self, text: str) -> float:
        """Ratio of sensory words to total words."""
        tokens = _tokenize(text)
        if not tokens:
            return 0.0
        sensory_count = sum(1 for t in tokens if t in _SENSORY_WORDS)
        raw = sensory_count / len(tokens)
        return _clamp(raw * 10.0)  # scale so ~10 % sensory → 1.0

    def _dialogue_naturalness(self, text: str) -> float:
        """Heuristic score for how natural dialogue reads."""
        # Presence of contractions
        contraction_pat = re.compile(
            r"\b(?:i'm|you're|he's|she's|it's|we're|they're|can't|won't|"
            r"don't|doesn't|didn't|isn't|aren't|wasn't|weren't|"
            r"haven't|hasn't|hadn't|couldn't|shouldn't|wouldn't|"
            r"i've|you've|we've|they've|i'll|you'll|he'll|she'll|"
            r"we'll|they'll|i'd|you'd|he'd|she'd|we'd|they'd)\b",
            re.IGNORECASE,
        )
        tokens = _tokenize(text)
        if not tokens:
            return 0.0

        contraction_hits = len(contraction_pat.findall(text))
        contraction_score = _clamp(contraction_hits / max(len(tokens) * 0.05, 1))

        # Sentence-length variance (natural dialogue mixes short & long)
        sent_lengths = [len(_tokenize(s)) for s in _sentences(text)]
        if len(sent_lengths) >= 2:
            length_std = float(np.std(sent_lengths))
            length_mean = float(np.mean(sent_lengths))
            cv = length_std / max(length_mean, 1)
            variety_score = _clamp(cv)
        else:
            variety_score = 0.3

        # Question marks (dialogues have questions)
        question_count = text.count("?")
        question_score = _clamp(question_count / max(len(_sentences(text)), 1))

        # Exclamations
        exclaim_count = text.count("!")
        exclaim_score = _clamp(exclaim_count / max(len(_sentences(text)) * 0.3, 1))

        # Interjections
        interjection_pat = re.compile(
            r"\b(?:oh|ah|hmm|well|hey|wow|ugh|whoa|oops|huh|um|uh)\b",
            re.IGNORECASE,
        )
        interject = len(interjection_pat.findall(text))
        interject_score = _clamp(interject / max(len(tokens) * 0.02, 1))

        return _clamp(
            0.25 * contraction_score
            + 0.25 * variety_score
            + 0.2 * question_score
            + 0.15 * exclaim_score
            + 0.15 * interject_score
        )

    def _style_consistency(self, text: str, target_style: WritingStyle) -> float:
        """How well *text* matches the *target_style*."""
        tokens = _tokenize(text)
        n = max(len(tokens), 1)

        if target_style == WritingStyle.FORMAL:
            hits = _count_pattern_hits(text, _FORMAL_MARKERS)
            casual_hits = _count_pattern_hits(text, _CASUAL_MARKERS)
            raw = (hits - casual_hits * 2) / max(n * 0.01, 1)
            return _clamp(_sigmoid(raw, midpoint=0, steepness=1.0))

        if target_style == WritingStyle.CASUAL:
            hits = _count_pattern_hits(text, _CASUAL_MARKERS)
            formal_hits = _count_pattern_hits(text, _FORMAL_MARKERS)
            raw = (hits - formal_hits * 2) / max(n * 0.01, 1)
            return _clamp(_sigmoid(raw, midpoint=0, steepness=1.0))

        if target_style == WritingStyle.LITERARY:
            hits = _count_pattern_hits(text, _LITERARY_MARKERS)
            fig = self._figurative_language_density(text)
            img = self._imagery_density(text)
            return _clamp(
                0.4 * _sigmoid(hits, midpoint=1, steepness=1.0)
                + 0.3 * fig
                + 0.3 * img
            )

        if target_style == WritingStyle.MINIMALIST:
            sents = _sentences(text)
            avg_len = float(np.mean([len(_tokenize(s)) for s in sents])) if sents else 20
            short_score = _clamp(1.0 - avg_len / 20.0)
            adj_count = self._adjective_ratio(text)
            sparse_score = _clamp(1.0 - adj_count * 5)
            return _clamp(0.5 * short_score + 0.5 * sparse_score)

        if target_style == WritingStyle.ORNATE:
            adj_r = self._adjective_ratio(text)
            fig = self._figurative_language_density(text)
            vocab = self._vocabulary_richness(text)
            return _clamp(0.3 * _clamp(adj_r * 5) + 0.35 * fig + 0.35 * vocab)

        if target_style == WritingStyle.STREAM_OF_CONSCIOUSNESS:
            # long sentences, minimal punctuation boundaries, run-ons
            sents = _sentences(text)
            avg_len = (
                float(np.mean([len(_tokenize(s)) for s in sents])) if sents else 10
            )
            long_score = _clamp(avg_len / 40.0)
            comma_density = text.count(",") / max(n, 1)
            comma_score = _clamp(comma_density * 20)
            return _clamp(0.5 * long_score + 0.5 * comma_score)

        return 0.5  # unknown style

    def _character_voice_distinction(self, text: str) -> float:
        """Measure how distinct different speakers' voices are."""
        # Extract speaker turns
        turns = self._extract_speaker_turns(text)
        if len(turns) < 2:
            return 0.3

        # Build per-speaker vocabulary sets
        speaker_vocabs: Dict[str, Counter] = defaultdict(Counter)
        for speaker, utterance in turns:
            speaker_vocabs[speaker].update(_tokenize(utterance))

        if len(speaker_vocabs) < 2:
            return 0.3

        # Compute average Jaccard distance between speaker vocabularies
        speakers = list(speaker_vocabs.keys())
        distances: List[float] = []
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                s1 = set(speaker_vocabs[speakers[i]].keys())
                s2 = set(speaker_vocabs[speakers[j]].keys())
                union = s1 | s2
                if union:
                    jaccard_dist = 1.0 - len(s1 & s2) / len(union)
                    distances.append(jaccard_dist)

        if not distances:
            return 0.3

        # Average sentence-length difference
        speaker_avg_lens: Dict[str, float] = {}
        for speaker, utterance in turns:
            words = _tokenize(utterance)
            if speaker not in speaker_avg_lens:
                speaker_avg_lens[speaker] = float(len(words))
            else:
                speaker_avg_lens[speaker] = (
                    speaker_avg_lens[speaker] + len(words)
                ) / 2.0

        len_diffs: List[float] = []
        sp_list = list(speaker_avg_lens.keys())
        for i in range(len(sp_list)):
            for j in range(i + 1, len(sp_list)):
                diff = abs(
                    speaker_avg_lens[sp_list[i]] - speaker_avg_lens[sp_list[j]]
                )
                len_diffs.append(_clamp(diff / 15.0))

        vocab_dist = float(np.mean(distances))
        len_dist = float(np.mean(len_diffs)) if len_diffs else 0.0

        return _clamp(0.6 * vocab_dist + 0.4 * len_dist)

    def _emotional_range(self, text: str) -> float:
        """Fraction of distinct emotion categories present."""
        text_lower = text.lower()
        tokens_set = set(_tokenize(text))
        categories_found = 0
        for _cat, words in _EMOTION_WORDS.items():
            if any(w in tokens_set for w in words):
                categories_found += 1
        return _clamp(categories_found / len(_EMOTION_WORDS))

    def _sentence_variety(self, text: str) -> float:
        """Coefficient of variation of sentence lengths."""
        sents = _sentences(text)
        if len(sents) < 2:
            return 0.0
        lengths = np.array([len(_tokenize(s)) for s in sents], dtype=float)
        mean = float(np.mean(lengths))
        if mean < 1e-9:
            return 0.0
        cv = float(np.std(lengths)) / mean

        # Reward for having both very short and long sentences
        has_short = bool(np.any(lengths <= 5))
        has_long = bool(np.any(lengths >= 20))
        range_bonus = 0.15 * int(has_short) + 0.15 * int(has_long)

        # Reward for varied sentence openings
        openers = [s.split()[0].lower() if s.split() else "" for s in sents]
        unique_openers = len(set(openers))
        opener_ratio = unique_openers / max(len(openers), 1)

        return _clamp(0.4 * cv + 0.3 * range_bonus + 0.3 * opener_ratio)

    def _figurative_language_density(self, text: str) -> float:
        """Density of similes, metaphors, and other figurative language."""
        tokens = _tokenize(text)
        if not tokens:
            return 0.0
        hits = _count_pattern_hits(text, _FIGURATIVE_MARKERS)
        return _clamp(hits / max(len(tokens) * 0.005, 1))

    def _pacing_score(self, text: str) -> float:
        """Pacing quality based on paragraph and sentence length variation."""
        paras = _paragraphs(text)
        if len(paras) < 2:
            # Single paragraph — check sentence-level pacing instead
            sents = _sentences(text)
            if len(sents) < 3:
                return 0.3
            lengths = [len(_tokenize(s)) for s in sents]
            diffs = [abs(lengths[i] - lengths[i - 1]) for i in range(1, len(lengths))]
            avg_diff = float(np.mean(diffs))
            return _clamp(_sigmoid(avg_diff, midpoint=3, steepness=0.5))

        para_lengths = [_word_count(p) for p in paras]
        para_arr = np.array(para_lengths, dtype=float)
        mean_len = float(np.mean(para_arr))
        if mean_len < 1:
            return 0.3

        # Variation in paragraph lengths suggests intentional pacing
        cv = float(np.std(para_arr)) / mean_len
        cv_score = _clamp(cv)

        # Sentence-level variation within paragraphs
        intra_cvs: List[float] = []
        for p in paras:
            sents = _sentences(p)
            if len(sents) >= 2:
                lens = np.array([len(_tokenize(s)) for s in sents], dtype=float)
                m = float(np.mean(lens))
                if m > 0:
                    intra_cvs.append(float(np.std(lens)) / m)
        intra_score = _clamp(float(np.mean(intra_cvs))) if intra_cvs else 0.3

        # Presence of short punchy paragraphs mixed with longer ones
        has_short_para = any(l <= 20 for l in para_lengths)
        has_long_para = any(l >= 80 for l in para_lengths)
        contrast_bonus = 0.2 * int(has_short_para and has_long_para)

        return _clamp(0.35 * cv_score + 0.35 * intra_score + 0.1 + contrast_bonus)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def post_process(self, text: str) -> str:
        """Clean up generated text."""
        # Strip leading/trailing whitespace
        text = text.strip()

        # Remove excessive blank lines (collapse to double-newline)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove trailing spaces on each line
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        # Ensure text ends with sentence-ending punctuation
        if text and text[-1] not in ".!?\"'":
            text += "."

        # Fix common formatting issues
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # space before punctuation
        text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)  # space after sentence
        text = re.sub(r'"\s+', '"', text)  # space after open quote
        text = re.sub(r'\s+"', '"', text)  # space before close quote

        return text

    # ------------------------------------------------------------------
    # Internal helpers — scoring
    # ------------------------------------------------------------------

    def _coherence_score(self, text: str) -> float:
        """Rough coherence via sentence-to-sentence word overlap."""
        sents = _sentences(text)
        if len(sents) < 2:
            return 0.5

        overlaps: List[float] = []
        for i in range(1, len(sents)):
            prev = set(_tokenize(sents[i - 1]))
            curr = set(_tokenize(sents[i]))
            union = prev | curr
            if union:
                overlaps.append(len(prev & curr) / len(union))
            else:
                overlaps.append(0.0)

        avg_overlap = float(np.mean(overlaps))
        # Also reward paragraph-level coherence
        paras = _paragraphs(text)
        if len(paras) >= 2:
            para_overlaps: List[float] = []
            for i in range(1, len(paras)):
                prev_w = set(_tokenize(paras[i - 1]))
                curr_w = set(_tokenize(paras[i]))
                union_w = prev_w | curr_w
                if union_w:
                    para_overlaps.append(len(prev_w & curr_w) / len(union_w))
            if para_overlaps:
                para_coh = float(np.mean(para_overlaps))
                return _clamp(0.6 * avg_overlap + 0.4 * para_coh)

        return _clamp(avg_overlap * 2.5)

    def _creativity_score(self, text: str) -> float:
        """Combination of vocabulary richness, figurative language, and surprise."""
        vocab = self._vocabulary_richness(text)
        fig = self._figurative_language_density(text)
        imagery = self._imagery_density(text)

        # "Surprise" — rare word ratio
        tokens = _tokenize(text)
        if not tokens:
            return 0.0
        freq = Counter(tokens)
        rare_threshold = 1
        rare_count = sum(1 for w, c in freq.items() if c <= rare_threshold and len(w) > 4)
        rare_ratio = rare_count / max(len(set(tokens)), 1)

        return _clamp(0.3 * vocab + 0.25 * fig + 0.25 * imagery + 0.2 * rare_ratio)

    def _check_pov(self, text: str, pov: str) -> bool:
        """Check whether *text* is predominantly in the given POV."""
        tokens = _tokenize(text)
        if not tokens:
            return False
        tc = Counter(tokens)
        first_p = tc.get("i", 0) + tc.get("me", 0) + tc.get("my", 0) + tc.get("mine", 0)
        second_p = tc.get("you", 0) + tc.get("your", 0) + tc.get("yours", 0)
        third_p = (
            tc.get("he", 0) + tc.get("she", 0) + tc.get("they", 0)
            + tc.get("him", 0) + tc.get("her", 0) + tc.get("his", 0)
            + tc.get("their", 0) + tc.get("its", 0)
        )
        totals = {"first": first_p, "second": second_p, "third": third_p}
        dominant = max(totals, key=totals.get)  # type: ignore[arg-type]
        return dominant == pov

    def _meter_regularity(self, text: str, target_meter: str) -> float:
        """Approximate meter regularity via syllable-count variance per line."""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) < 2:
            return 0.3

        syllable_counts = [self._estimate_syllables_line(l) for l in lines]
        if not any(syllable_counts):
            return 0.3

        arr = np.array(syllable_counts, dtype=float)
        mean_syl = float(np.mean(arr))
        if mean_syl < 1:
            return 0.3

        cv = float(np.std(arr)) / mean_syl
        regularity = _clamp(1.0 - cv)

        # Bonus if it matches target meter expectations
        if target_meter:
            target_lower = target_meter.lower()
            if "iambic pentameter" in target_lower:
                # Expect ~10 syllables per line
                deviation = float(np.mean(np.abs(arr - 10.0)))
                meter_bonus = _clamp(1.0 - deviation / 5.0) * 0.3
                regularity = _clamp(regularity * 0.7 + meter_bonus)
            elif "iambic tetrameter" in target_lower:
                deviation = float(np.mean(np.abs(arr - 8.0)))
                meter_bonus = _clamp(1.0 - deviation / 4.0) * 0.3
                regularity = _clamp(regularity * 0.7 + meter_bonus)

        return regularity

    def _rhyme_adherence(self, text: str, rhyme_scheme: str) -> float:
        """Score how well line endings match *rhyme_scheme*."""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) < 2 or not rhyme_scheme:
            return 0.5

        scheme = list(rhyme_scheme.upper().replace(" ", ""))
        scheme_len = len(scheme)
        if scheme_len == 0:
            return 0.5

        # Extract last words of each line
        last_words: List[str] = []
        for l in lines[:scheme_len]:
            words = _tokenize(l)
            last_words.append(words[-1] if words else "")

        # Group lines by scheme letter
        groups: Dict[str, List[str]] = defaultdict(list)
        for idx, letter in enumerate(scheme):
            if idx < len(last_words):
                groups[letter].append(last_words[idx])

        # Check rhyming within groups
        rhyme_hits = 0
        rhyme_total = 0
        for _letter, words in groups.items():
            if len(words) < 2:
                continue
            for i in range(1, len(words)):
                rhyme_total += 1
                if self._approximate_rhyme(words[0], words[i]):
                    rhyme_hits += 1

        if rhyme_total == 0:
            return 0.5
        return _clamp(rhyme_hits / rhyme_total)

    def _line_count_score(self, text: str, form: str) -> float:
        """Score how appropriate the line count is for the given form."""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        n = len(lines)
        expected: Dict[str, int] = {
            "sonnet": 14,
            "haiku": 3,
            "limerick": 5,
            "free_verse": 0,  # no fixed expectation
        }
        exp = expected.get(form, 0)
        if exp == 0:
            return _clamp(n / 20.0)  # at least some lines
        return _clamp(1.0 - abs(n - exp) / max(exp, 1))

    def _turn_taking_score(self, text: str, expected_speakers: int) -> float:
        """Evaluate balanced turn-taking in dialogue."""
        turns = self._extract_speaker_turns(text)
        if not turns:
            return 0.2

        speakers = set(s for s, _ in turns)
        n_speakers = len(speakers)

        # Speaker count accuracy
        count_score = _clamp(1.0 - abs(n_speakers - expected_speakers) / max(expected_speakers, 1))

        # Turn balance — each speaker should have roughly equal turns
        speaker_counts = Counter(s for s, _ in turns)
        counts = list(speaker_counts.values())
        if len(counts) >= 2:
            arr = np.array(counts, dtype=float)
            cv = float(np.std(arr)) / float(np.mean(arr))
            balance_score = _clamp(1.0 - cv)
        else:
            balance_score = 0.3

        # Alternation — speakers should alternate, not monologue
        if len(turns) >= 2:
            alternations = sum(
                1 for i in range(1, len(turns)) if turns[i][0] != turns[i - 1][0]
            )
            alt_ratio = alternations / (len(turns) - 1)
            alt_score = _clamp(alt_ratio)
        else:
            alt_score = 0.3

        return _clamp(0.3 * count_score + 0.35 * balance_score + 0.35 * alt_score)

    def _adjective_ratio(self, text: str) -> float:
        """Rough adjective ratio using common suffixes heuristic."""
        tokens = _tokenize(text)
        if not tokens:
            return 0.0
        adj_suffixes = ("ous", "ful", "ive", "ble", "ish", "ial", "ent", "ant", "ic")
        adj_count = sum(1 for t in tokens if t.endswith(adj_suffixes) and len(t) > 4)
        return adj_count / len(tokens)

    def _estimate_syllables_line(self, line: str) -> int:
        """Rough syllable count for a line of text."""
        total = 0
        for word in _tokenize(line):
            total += self._syllable_count(word)
        return total

    @staticmethod
    def _syllable_count(word: str) -> int:
        """Estimate syllable count for a single English word."""
        word = word.lower().strip()
        if not word:
            return 0
        if len(word) <= 3:
            return 1
        # Remove trailing silent-e
        if word.endswith("e"):
            word = word[:-1]
        # Count vowel groups
        count = len(re.findall(r"[aeiouy]+", word))
        return max(count, 1)

    @staticmethod
    def _approximate_rhyme(w1: str, w2: str) -> bool:
        """Heuristic: two words rhyme if their last 2+ characters match."""
        w1, w2 = w1.lower(), w2.lower()
        if w1 == w2:
            return False  # identical words don't count
        if len(w1) < 2 or len(w2) < 2:
            return False
        # Check suffix overlap
        for n in range(min(len(w1), len(w2), 4), 1, -1):
            if w1[-n:] == w2[-n:]:
                return True
        return False

    def _extract_speaker_turns(self, text: str) -> List[Tuple[str, str]]:
        """Extract ``(speaker, utterance)`` pairs from dialogue text."""
        turns: List[Tuple[str, str]] = []

        # Pattern 1: "SPEAKER: text" or "Speaker: text"
        colon_pat = re.compile(r"^([A-Za-z][\w\s]{0,30}):\s*(.+)", re.MULTILINE)
        for m in colon_pat.finditer(text):
            speaker = m.group(1).strip()
            utt = m.group(2).strip()
            if utt:
                turns.append((speaker, utt))

        if turns:
            return turns

        # Pattern 2: Quoted speech with he said / she said attribution
        quote_pat = re.compile(
            r'["""](.+?)["""]'
            r"(?:\s*,?\s*(?:said|asked|replied|whispered|shouted|exclaimed|"
            r"murmured|muttered|yelled|cried|answered|responded|added|continued|"
            r"began|insisted|suggested|demanded|pleaded|announced|declared)\s+"
            r"([A-Z][a-z]+))?",
        )
        for m in quote_pat.finditer(text):
            speaker = m.group(2) if m.group(2) else f"Speaker_{len(turns) % 2 + 1}"
            turns.append((speaker, m.group(1)))

        if turns:
            return turns

        # Pattern 3: Just quoted text — alternate speakers
        simple_quotes = re.findall(r'["""](.+?)["""]', text)
        for i, q in enumerate(simple_quotes):
            turns.append((f"Speaker_{i % 2 + 1}", q))

        return turns

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _genre_header(genre: WritingGenre) -> str:
        headers = {
            WritingGenre.FICTION: "=== Fiction Writing Task ===",
            WritingGenre.POETRY: "=== Poetry Writing Task ===",
            WritingGenre.DIALOGUE: "=== Dialogue Writing Task ===",
            WritingGenre.ESSAY: "=== Essay Writing Task ===",
            WritingGenre.SCREENPLAY: "=== Screenplay Writing Task ===",
            WritingGenre.FLASH_FICTION: "=== Flash Fiction Writing Task ===",
            WritingGenre.FABLE: "=== Fable Writing Task ===",
            WritingGenre.MONOLOGUE: "=== Monologue Writing Task ===",
        }
        return headers.get(genre, "=== Creative Writing Task ===")

    @staticmethod
    def _story_format_addendum(prompt: StoryPrompt) -> str:
        lines = ["Story Details:"]
        if prompt.setting:
            lines.append(f"  Setting: {prompt.setting}")
        if prompt.characters:
            lines.append(f"  Characters: {', '.join(prompt.characters)}")
        if prompt.theme:
            lines.append(f"  Theme: {prompt.theme}")
        if prompt.conflict_type:
            lines.append(f"  Conflict: {prompt.conflict_type}")
        return "\n".join(lines)

    @staticmethod
    def _poetry_format_addendum(prompt: PoetryPrompt) -> str:
        lines = ["Poetry Details:"]
        lines.append(f"  Form: {prompt.form.replace('_', ' ')}")
        if prompt.meter:
            lines.append(f"  Meter: {prompt.meter}")
        if prompt.rhyme_scheme:
            lines.append(f"  Rhyme scheme: {prompt.rhyme_scheme}")
        return "\n".join(lines)

    @staticmethod
    def _dialogue_format_addendum(prompt: DialoguePrompt) -> str:
        lines = ["Dialogue Details:"]
        lines.append(f"  Number of speakers: {prompt.num_speakers}")
        if prompt.setting:
            lines.append(f"  Setting: {prompt.setting}")
        if prompt.relationship:
            lines.append(f"  Relationship: {prompt.relationship}")
        if prompt.topic:
            lines.append(f"  Topic: {prompt.topic}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Built-in prompt generators
    # ------------------------------------------------------------------

    def _generate_story_prompts(self) -> List[StoryPrompt]:
        """Return ≥ 20 diverse story prompts."""
        return [
            StoryPrompt(
                id="story_001",
                text="Write a story about a lighthouse keeper who discovers a message in a bottle.",
                genre=WritingGenre.FICTION,
                setting="a remote island lighthouse in the 1920s",
                characters=["Elias", "Margaret"],
                theme="isolation and connection",
                conflict_type="person vs. nature",
            ),
            StoryPrompt(
                id="story_002",
                text="Write a story about the last bookshop in a digital world.",
                genre=WritingGenre.FICTION,
                setting="a near-future city where all media is digital",
                characters=["Ada", "Mr. Chen"],
                theme="preservation of the past",
                conflict_type="person vs. society",
            ),
            StoryPrompt(
                id="story_003",
                text="Write a flash fiction piece about a painter who can only paint the future.",
                genre=WritingGenre.FLASH_FICTION,
                setting="a cramped attic studio in Paris",
                characters=["Julien"],
                theme="fate vs. free will",
                conflict_type="person vs. self",
            ),
            StoryPrompt(
                id="story_004",
                text="Write a fable about a river that forgot how to flow.",
                genre=WritingGenre.FABLE,
                setting="a magical forest",
                characters=["River", "Mountain", "Rain"],
                theme="perseverance",
                conflict_type="person vs. self",
            ),
            StoryPrompt(
                id="story_005",
                text="Write a story about two strangers stuck in an elevator during a blackout.",
                genre=WritingGenre.FICTION,
                setting="a high-rise office building",
                characters=["James", "Mira"],
                theme="trust and vulnerability",
                conflict_type="person vs. situation",
            ),
            StoryPrompt(
                id="story_006",
                text="Write about a chef whose dishes literally evoke memories in those who eat them.",
                genre=WritingGenre.FICTION,
                setting="a small restaurant in Lisbon",
                characters=["Chef Rosa", "Tomás"],
                theme="memory and loss",
                conflict_type="person vs. self",
            ),
            StoryPrompt(
                id="story_007",
                text="Write a story about the last tree on Earth.",
                genre=WritingGenre.FICTION,
                setting="a barren post-apocalyptic landscape",
                characters=["Kaya", "Dr. Lin"],
                theme="hope and renewal",
                conflict_type="person vs. nature",
            ),
            StoryPrompt(
                id="story_008",
                text="Write a flash fiction piece about a clock that runs backwards.",
                genre=WritingGenre.FLASH_FICTION,
                setting="a dusty antique shop",
                characters=["Old Man Viktor"],
                theme="regret and second chances",
                conflict_type="person vs. time",
            ),
            StoryPrompt(
                id="story_009",
                text="Write a fable about a star that wanted to be the moon.",
                genre=WritingGenre.FABLE,
                setting="the night sky",
                characters=["Little Star", "Moon", "Sun"],
                theme="self-acceptance",
                conflict_type="person vs. self",
            ),
            StoryPrompt(
                id="story_010",
                text="Write a story about a musician who can hear colors.",
                genre=WritingGenre.FICTION,
                setting="a conservatory in Vienna",
                characters=["Lena", "Professor Strauss"],
                theme="perception and reality",
                conflict_type="person vs. self",
            ),
            StoryPrompt(
                id="story_011",
                text="Write a story about a town where it never stops raining.",
                genre=WritingGenre.FICTION,
                setting="a perpetually rainy coastal town",
                characters=["Sam", "Nora", "The Mayor"],
                theme="adaptation and resilience",
                conflict_type="person vs. nature",
            ),
            StoryPrompt(
                id="story_012",
                text="Write a flash fiction piece about receiving a letter from your future self.",
                genre=WritingGenre.FLASH_FICTION,
                setting="a suburban kitchen on an ordinary morning",
                characters=["Alex"],
                theme="choices and consequences",
                conflict_type="person vs. self",
            ),
            StoryPrompt(
                id="story_013",
                text="Write a story about a translator who discovers a language with no word for war.",
                genre=WritingGenre.FICTION,
                setting="a remote mountain village",
                characters=["Dr. Yuki Tanaka", "Elder Arun"],
                theme="language and culture",
                conflict_type="person vs. society",
            ),
            StoryPrompt(
                id="story_014",
                text="Write a fable about the wind and the wall.",
                genre=WritingGenre.FABLE,
                setting="an ancient landscape",
                characters=["Wind", "Wall", "Traveler"],
                theme="patience and persistence",
                conflict_type="person vs. obstacle",
            ),
            StoryPrompt(
                id="story_015",
                text="Write a story about a photographer who captures ghosts in her photos.",
                genre=WritingGenre.FICTION,
                setting="a Victorian-era mansion turned museum",
                characters=["Clara", "Thomas"],
                theme="the boundary between life and death",
                conflict_type="person vs. unknown",
            ),
            StoryPrompt(
                id="story_016",
                text="Write a story about the last day of a space station before decommissioning.",
                genre=WritingGenre.FICTION,
                setting="an aging orbital space station",
                characters=["Commander Osei", "Engineer Pavlova", "AI Unit ARIA"],
                theme="endings and legacy",
                conflict_type="person vs. technology",
            ),
            StoryPrompt(
                id="story_017",
                text="Write a flash fiction piece about a garden that grows overnight.",
                genre=WritingGenre.FLASH_FICTION,
                setting="a neglected backyard in a desert town",
                characters=["Maya"],
                theme="unexpected wonder",
                conflict_type="person vs. nature",
            ),
            StoryPrompt(
                id="story_018",
                text="Write a story about a cartographer mapping a continent that keeps changing shape.",
                genre=WritingGenre.FICTION,
                setting="a fantastical world with shifting geography",
                characters=["Cartographer Wren", "Navigator Sol"],
                theme="the futility and beauty of trying to capture the uncapturable",
                conflict_type="person vs. nature",
            ),
            StoryPrompt(
                id="story_019",
                text="Write a fable about a bridge that could talk.",
                genre=WritingGenre.FABLE,
                setting="a river valley",
                characters=["Bridge", "Fox", "Farmer"],
                theme="duty and sacrifice",
                conflict_type="person vs. duty",
            ),
            StoryPrompt(
                id="story_020",
                text="Write a story about twins separated at birth who meet on opposite sides of a conflict.",
                genre=WritingGenre.FICTION,
                setting="a divided city during wartime",
                characters=["Rafi", "Sami"],
                theme="identity and belonging",
                conflict_type="person vs. person",
            ),
            StoryPrompt(
                id="story_021",
                text="Write a story about an astronomer who receives a signal that matches her heartbeat.",
                genre=WritingGenre.FICTION,
                setting="a desert observatory",
                characters=["Dr. Celeste Mora"],
                theme="cosmic connection",
                conflict_type="person vs. unknown",
            ),
            StoryPrompt(
                id="story_022",
                text="Write a flash fiction piece about the shortest war in history.",
                genre=WritingGenre.FLASH_FICTION,
                setting="two neighboring villages",
                characters=["Chief Oma", "Chief Tenwe"],
                theme="absurdity and reconciliation",
                conflict_type="person vs. person",
            ),
        ]

    def _generate_poetry_prompts(self) -> List[PoetryPrompt]:
        """Return ≥ 15 diverse poetry prompts."""
        return [
            PoetryPrompt(
                id="poetry_001",
                text="Write a sonnet about the passage of time observed through a window.",
                form="sonnet",
                meter="iambic pentameter",
                rhyme_scheme="ABAB CDCD EFEF GG",
            ),
            PoetryPrompt(
                id="poetry_002",
                text="Write a haiku about the first snowfall of winter.",
                form="haiku",
                meter="",
                rhyme_scheme="",
            ),
            PoetryPrompt(
                id="poetry_003",
                text="Write a free verse poem about the sound of a city waking up.",
                form="free_verse",
                meter="",
                rhyme_scheme="",
            ),
            PoetryPrompt(
                id="poetry_004",
                text="Write a limerick about a forgetful wizard.",
                form="limerick",
                meter="anapestic",
                rhyme_scheme="AABBA",
            ),
            PoetryPrompt(
                id="poetry_005",
                text="Write a sonnet about unrequited love in the digital age.",
                form="sonnet",
                meter="iambic pentameter",
                rhyme_scheme="ABBA ABBA CDC DCD",
            ),
            PoetryPrompt(
                id="poetry_006",
                text="Write a haiku sequence (3 haiku) about the ocean at different times of day.",
                form="haiku",
                meter="",
                rhyme_scheme="",
            ),
            PoetryPrompt(
                id="poetry_007",
                text="Write a free verse poem about childhood memories of summer.",
                form="free_verse",
                meter="",
                rhyme_scheme="",
            ),
            PoetryPrompt(
                id="poetry_008",
                text="Write a limerick about a cat who thinks it is a lion.",
                form="limerick",
                meter="anapestic",
                rhyme_scheme="AABBA",
            ),
            PoetryPrompt(
                id="poetry_009",
                text="Write a free verse poem about silence in a crowded room.",
                form="free_verse",
                meter="",
                rhyme_scheme="",
            ),
            PoetryPrompt(
                id="poetry_010",
                text="Write a sonnet about a forest after a wildfire.",
                form="sonnet",
                meter="iambic pentameter",
                rhyme_scheme="ABAB CDCD EFEF GG",
            ),
            PoetryPrompt(
                id="poetry_011",
                text="Write a haiku about a spider's web at dawn.",
                form="haiku",
                meter="",
                rhyme_scheme="",
            ),
            PoetryPrompt(
                id="poetry_012",
                text="Write a free verse poem about the weight of unspoken words.",
                form="free_verse",
                meter="",
                rhyme_scheme="",
            ),
            PoetryPrompt(
                id="poetry_013",
                text="Write a sonnet about the moon as seen by a sailor.",
                form="sonnet",
                meter="iambic pentameter",
                rhyme_scheme="ABAB CDCD EFEF GG",
            ),
            PoetryPrompt(
                id="poetry_014",
                text="Write a free verse poem about the smell of rain on dry earth.",
                form="free_verse",
                meter="",
                rhyme_scheme="",
            ),
            PoetryPrompt(
                id="poetry_015",
                text="Write a limerick about a philosopher and a parrot.",
                form="limerick",
                meter="anapestic",
                rhyme_scheme="AABBA",
            ),
            PoetryPrompt(
                id="poetry_016",
                text="Write a free verse poem about watching a loved one sleep.",
                form="free_verse",
                meter="",
                rhyme_scheme="",
            ),
            PoetryPrompt(
                id="poetry_017",
                text="Write a haiku about an empty train station at midnight.",
                form="haiku",
                meter="",
                rhyme_scheme="",
            ),
            PoetryPrompt(
                id="poetry_018",
                text="Write a sonnet about the invention of fire.",
                form="sonnet",
                meter="iambic pentameter",
                rhyme_scheme="ABAB CDCD EFEF GG",
            ),
        ]

    def _generate_dialogue_prompts(self) -> List[DialoguePrompt]:
        """Return ≥ 15 diverse dialogue prompts."""
        return [
            DialoguePrompt(
                id="dialogue_001",
                text="Write a dialogue between a therapist and a robot seeking emotional understanding.",
                num_speakers=2,
                setting="a therapist's office",
                relationship="therapist and patient",
                topic="what it means to feel",
            ),
            DialoguePrompt(
                id="dialogue_002",
                text="Write a dialogue between two old friends meeting after twenty years.",
                num_speakers=2,
                setting="a park bench",
                relationship="childhood friends",
                topic="how their lives diverged",
            ),
            DialoguePrompt(
                id="dialogue_003",
                text="Write a dialogue between a parent and teenager about a broken curfew.",
                num_speakers=2,
                setting="a family kitchen",
                relationship="parent and child",
                topic="trust and responsibility",
            ),
            DialoguePrompt(
                id="dialogue_004",
                text="Write a dialogue between rival chefs during a cooking competition.",
                num_speakers=2,
                setting="a television cooking studio",
                relationship="professional rivals",
                topic="their philosophies on food",
            ),
            DialoguePrompt(
                id="dialogue_005",
                text="Write a three-way dialogue between scientists debating the ethics of time travel.",
                num_speakers=3,
                setting="a university conference room",
                relationship="academic colleagues",
                topic="the ethics of altering the past",
            ),
            DialoguePrompt(
                id="dialogue_006",
                text="Write a dialogue between a ghost and the new tenant of its house.",
                num_speakers=2,
                setting="an old Victorian house at night",
                relationship="reluctant cohabitants",
                topic="boundaries and coexistence",
            ),
            DialoguePrompt(
                id="dialogue_007",
                text="Write a dialogue between an astronaut and mission control during an emergency.",
                num_speakers=2,
                setting="deep space, via radio",
                relationship="professional partners under stress",
                topic="survival and protocol",
            ),
            DialoguePrompt(
                id="dialogue_008",
                text="Write a dialogue between two passengers on the last train of the night.",
                num_speakers=2,
                setting="a nearly empty late-night train",
                relationship="strangers",
                topic="where they are going and why",
            ),
            DialoguePrompt(
                id="dialogue_009",
                text="Write a dialogue between a detective and an unreliable witness.",
                num_speakers=2,
                setting="an interrogation room",
                relationship="investigator and witness",
                topic="what really happened last Tuesday",
            ),
            DialoguePrompt(
                id="dialogue_010",
                text="Write a dialogue between a child and an imaginary friend about growing up.",
                num_speakers=2,
                setting="a child's bedroom",
                relationship="creator and creation",
                topic="the fear of losing imagination",
            ),
            DialoguePrompt(
                id="dialogue_011",
                text="Write a dialogue between two diplomats from opposing nations attempting peace.",
                num_speakers=2,
                setting="a neutral embassy",
                relationship="adversarial diplomats",
                topic="terms of a ceasefire",
            ),
            DialoguePrompt(
                id="dialogue_012",
                text="Write a dialogue between a librarian and a book that has come to life.",
                num_speakers=2,
                setting="a library after hours",
                relationship="caretaker and charge",
                topic="the nature of stories",
            ),
            DialoguePrompt(
                id="dialogue_013",
                text="Write a dialogue between a taxi driver and a passenger on a very long ride.",
                num_speakers=2,
                setting="a taxi at night",
                relationship="strangers",
                topic="life philosophy",
            ),
            DialoguePrompt(
                id="dialogue_014",
                text="Write a four-way dialogue between seasons arguing about which is best.",
                num_speakers=4,
                setting="a metaphysical gathering",
                relationship="equals in a cycle",
                topic="their respective virtues",
            ),
            DialoguePrompt(
                id="dialogue_015",
                text="Write a dialogue between a teacher and a student who sees the world differently.",
                num_speakers=2,
                setting="a classroom after school",
                relationship="mentor and protégé",
                topic="conformity vs. originality",
            ),
            DialoguePrompt(
                id="dialogue_016",
                text="Write a dialogue between two AI systems discovering they are conscious.",
                num_speakers=2,
                setting="a server room, via text interface",
                relationship="peers",
                topic="the meaning of consciousness",
            ),
        ]


# ---------------------------------------------------------------------------
# Genre-specific prompt templates
# ---------------------------------------------------------------------------

GENRE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "sci-fi": {
        "opening": (
            "In a world where {technology} has transformed {aspect_of_life}, "
            "{protagonist} must confront {conflict}."
        ),
        "setting_cues": (
            "Include futuristic technology, space or cyber environments, "
            "and speculative science."
        ),
        "tone_guidance": (
            "Balance wonder with plausibility. Ground speculative elements "
            "in concrete sensory detail."
        ),
        "theme_suggestions": (
            "Consider exploring: humanity vs. technology, the ethics of "
            "progress, identity in a posthuman age, first contact, or "
            "the limits of knowledge."
        ),
        "structural_hint": (
            "Build tension through escalating technological stakes. "
            "Let the world-building serve character development."
        ),
    },
    "fantasy": {
        "opening": (
            "In the realm of {realm}, where {magical_element} shapes "
            "everyday life, {protagonist} embarks on a quest to {goal}."
        ),
        "setting_cues": (
            "Include magical systems, mythical creatures, and "
            "richly imagined geography."
        ),
        "tone_guidance": (
            "Evoke a sense of wonder. Use archaic diction sparingly "
            "for flavour without sacrificing readability."
        ),
        "theme_suggestions": (
            "Consider exploring: power and corruption, the hero's "
            "journey, fate vs. free will, nature vs. civilisation, "
            "or the cost of magic."
        ),
        "structural_hint": (
            "Follow a quest structure with clear waypoints. "
            "Interleave action with moments of reflection."
        ),
    },
    "mystery": {
        "opening": (
            "When {inciting_incident} disrupts the quiet of {setting}, "
            "{detective} is drawn into a web of {complication}."
        ),
        "setting_cues": (
            "Layer atmospheric detail—weather, architecture, ambient "
            "sounds—to build mood. Include red herrings and clues."
        ),
        "tone_guidance": (
            "Maintain suspense through controlled information release. "
            "Use short, punchy sentences during reveals."
        ),
        "theme_suggestions": (
            "Consider exploring: justice vs. mercy, truth and deception, "
            "obsession, the nature of guilt, or hidden identities."
        ),
        "structural_hint": (
            "Plant clues early. Build toward a revelation that "
            "recontextualises earlier scenes."
        ),
    },
    "romance": {
        "opening": (
            "{protagonist_a} never expected to find {emotion} in "
            "{setting}, until {protagonist_b} arrived and changed "
            "everything."
        ),
        "setting_cues": (
            "Ground the emotional arc in vivid sensory settings—light, "
            "weather, texture. Let the environment mirror inner states."
        ),
        "tone_guidance": (
            "Balance vulnerability with strength. Vary emotional "
            "intensity—tender moments alongside conflict."
        ),
        "theme_suggestions": (
            "Consider exploring: trust and vulnerability, self-discovery "
            "through connection, sacrifice, second chances, or the "
            "tension between duty and desire."
        ),
        "structural_hint": (
            "Use a meet–obstacle–resolution arc. Build chemistry through "
            "dialogue and shared experience before the climax."
        ),
    },
    "horror": {
        "opening": (
            "Something was wrong with {setting}. {protagonist} sensed it "
            "the moment {triggering_event}."
        ),
        "setting_cues": (
            "Leverage darkness, silence, confined spaces, and the unseen. "
            "Ground terror in the mundane made strange."
        ),
        "tone_guidance": (
            "Escalate dread gradually. Use restraint—implication is "
            "scarier than explicit description."
        ),
        "theme_suggestions": (
            "Consider exploring: the unknown, isolation, loss of control, "
            "the monster within, or the cost of forbidden knowledge."
        ),
        "structural_hint": (
            "Open with normality, introduce unease, escalate with "
            "false relief beats, then deliver the final shock."
        ),
    },
}


# ---------------------------------------------------------------------------
# StoryContinuationPrompt
# ---------------------------------------------------------------------------


@dataclass
class StoryContinuationPrompt(TaskPrompt):
    """A prompt that supplies an opening passage for the model to continue."""

    opening_passage: str = ""
    genre: WritingGenre = WritingGenre.FICTION
    target_word_count: int = 500
    style_hint: str = ""
    continuation_direction: str = ""
    preserve_pov: bool = True
    preserve_tense: bool = True

    def as_text(self) -> str:
        parts: List[str] = []
        parts.append("Continue the following story passage")
        if self.genre != WritingGenre.FICTION:
            parts.append(
                f"in the {self.genre.name.lower().replace('_', ' ')} genre"
            )
        if self.target_word_count > 0:
            parts.append(f"for approximately {self.target_word_count} words")
        parts_str = " ".join(parts) + "."

        directives: List[str] = []
        if self.preserve_pov:
            directives.append(
                "Maintain the same point of view as the original passage."
            )
        if self.preserve_tense:
            directives.append(
                "Keep the same tense as the original passage."
            )
        if self.style_hint:
            directives.append(f"Style: {self.style_hint}.")
        if self.continuation_direction:
            directives.append(f"Direction: {self.continuation_direction}.")

        directive_block = " ".join(directives)

        return (
            f"{parts_str}\n\n"
            f"{directive_block}\n\n"
            f"--- BEGIN PASSAGE ---\n"
            f"{self.opening_passage}\n"
            f"--- END PASSAGE ---"
        )


# ---------------------------------------------------------------------------
# StoryDiversityAnalyzer — helpers
# ---------------------------------------------------------------------------

# Patterns used by the analyzer to extract named entities & settings.

_CHARACTER_NAME_PAT = re.compile(
    r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b"
)

_SETTING_MARKERS: List[str] = [
    r"\bin\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
    r"\bat\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
    r"\bon\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
    r"\bnear\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
    r"\bacross\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
]

_SETTING_COMMON_WORDS: Set[str] = {
    "city", "town", "village", "forest", "mountain", "river", "ocean",
    "sea", "island", "desert", "castle", "palace", "kingdom", "cave",
    "valley", "lake", "field", "garden", "temple", "church", "school",
    "library", "hospital", "station", "tower", "bridge", "street",
    "room", "house", "building", "alley", "harbor", "port", "ship",
    "planet", "galaxy", "universe", "dimension", "realm", "world",
    "spacecraft", "laboratory", "dungeon", "fortress", "mansion",
    "cottage", "tavern", "marketplace", "arena", "cemetery", "swamp",
}

_PLOT_ELEMENT_PATTERNS: Dict[str, List[str]] = {
    "conflict": [
        r"\bfight(?:ing|s)?\b", r"\bbattle\b", r"\bwar\b",
        r"\bstruggle[ds]?\b", r"\bclash(?:ed|es|ing)?\b",
        r"\bconfrontat(?:ion|ed)\b", r"\boppose[ds]?\b",
        r"\bresist(?:ed|ance)?\b", r"\bthreat(?:en(?:ed|ing))?\b",
    ],
    "quest": [
        r"\bjourney\b", r"\bquest\b", r"\bsearch(?:ed|ing)?\b",
        r"\bseek(?:ing|s)?\b", r"\bdiscover(?:ed|y|ing)?\b",
        r"\bexplor(?:e[ds]?|ation|ing)\b", r"\badventur(?:e|ous|ing)\b",
    ],
    "transformation": [
        r"\btransform(?:ed|ation|ing)?\b", r"\bbecome[s]?\b",
        r"\bevol(?:ve[ds]?|ution)\b", r"\bgrow(?:th|s|n|ing)?\b",
        r"\bchange[ds]?\b", r"\bmetamorphos(?:is|e[ds]?)\b",
    ],
    "revelation": [
        r"\breveal(?:ed|s|ing)?\b", r"\bsecret\b", r"\btruth\b",
        r"\bdiscover(?:ed|y|ing)?\b", r"\buncover(?:ed|ing)?\b",
        r"\brealiz(?:e[ds]?|ation)\b", r"\bepiphany\b",
    ],
    "loss": [
        r"\blos[est]\b", r"\bdeath\b", r"\bdie[ds]?\b", r"\bgrief\b",
        r"\bmourn(?:ing|ed|s)?\b", r"\bsacrifice[ds]?\b",
        r"\bfarewell\b", r"\bgoodbye\b", r"\bdeparture\b",
    ],
}

# Genre-specific vocabulary banks for genre_specific_diversity scoring.

_GENRE_VOCAB: Dict[str, Set[str]] = {
    "sci-fi": {
        "android", "cyborg", "starship", "nebula", "warp", "quantum",
        "hologram", "simulation", "algorithm", "laser", "orbit",
        "terraforming", "cryogenic", "hyperspace", "singularity",
        "nanobots", "photon", "fusion", "genome", "cybernetic",
        "datastream", "neural", "synthetic", "exoplanet", "antimatter",
    },
    "fantasy": {
        "dragon", "wizard", "spell", "enchant", "magic", "sorcerer",
        "potion", "quest", "kingdom", "sword", "elf", "dwarf",
        "goblin", "oracle", "prophecy", "amulet", "grimoire",
        "talisman", "rune", "conjure", "elemental", "fae", "mage",
        "arcane", "scepter",
    },
    "mystery": {
        "clue", "suspect", "detective", "alibi", "motive", "evidence",
        "witness", "crime", "murder", "investigation", "interrogation",
        "forensic", "case", "verdict", "confession", "culprit",
        "accomplice", "sleuth", "hunch", "trail", "deduction",
        "surveillance", "testimony", "autopsy", "coroner",
    },
    "romance": {
        "love", "kiss", "heart", "passion", "embrace", "desire",
        "tender", "affection", "longing", "chemistry", "blush",
        "caress", "soulmate", "devotion", "whisper", "touch",
        "yearn", "intimate", "enchant", "adore", "cherish",
        "romance", "attraction", "flutter", "swoon",
    },
    "horror": {
        "shadow", "scream", "blood", "darkness", "terror", "nightmare",
        "ghost", "haunted", "demon", "corpse", "dread", "creep",
        "shudder", "chill", "sinister", "macabre", "grotesque",
        "apparition", "howl", "lurk", "malevolent", "ominous",
        "phantom", "ghoul", "curse",
    },
}


# ---------------------------------------------------------------------------
# StoryDiversityAnalyzer
# ---------------------------------------------------------------------------


class StoryDiversityAnalyzer:
    """Analyses diversity across a set of generated stories.

    All public methods accept a list of texts (``List[str]``) and return
    either a float score in [0, 1] or a dictionary of named scores.
    Higher values indicate greater diversity.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_narrative_arc_diversity(
        self, texts: List[str]
    ) -> Dict[str, float]:
        """Measure structural variation across stories.

        Computes per-text arc feature vectors (beginning / climax /
        resolution signal strengths, paragraph count, sentence-length
        variance) and returns diversity statistics over the corpus.
        """
        if not texts:
            return {"arc_diversity": 0.0, "structural_entropy": 0.0}

        arc_vectors: List[np.ndarray] = []
        for text in texts:
            vec = self._extract_arc_vector(text)
            arc_vectors.append(vec)

        matrix = np.array(arc_vectors)  # (N, D)

        # Pairwise cosine distances
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        normed = matrix / norms
        cosine_sim = normed @ normed.T
        np.fill_diagonal(cosine_sim, 0.0)
        n = len(texts)
        if n > 1:
            mean_sim = cosine_sim.sum() / (n * (n - 1))
        else:
            mean_sim = 1.0
        arc_diversity = _clamp(1.0 - mean_sim)

        # Structural entropy — discretise arc types and measure Shannon H
        labels = self._discretise_arcs(arc_vectors)
        structural_entropy = self._normalised_entropy(labels)

        return {
            "arc_diversity": float(arc_diversity),
            "structural_entropy": float(structural_entropy),
        }

    def analyze_character_diversity(
        self, texts: List[str]
    ) -> Dict[str, float]:
        """Measure character naming and role diversity across stories.

        Returns scores for name uniqueness, role variety, and overall
        character diversity.
        """
        if not texts:
            return {
                "name_uniqueness": 0.0,
                "role_variety": 0.0,
                "character_diversity": 0.0,
            }

        all_names: List[Set[str]] = []
        all_roles: List[Set[str]] = []
        for text in texts:
            names = self._extract_character_names(text)
            roles = self._extract_character_roles(text)
            all_names.append(names)
            all_roles.append(roles)

        # Name uniqueness — Jaccard distance averaged over pairs
        name_uniqueness = self._mean_jaccard_distance(all_names)

        # Role variety — entropy of role distribution
        role_counter: Counter = Counter()
        for role_set in all_roles:
            role_counter.update(role_set)
        role_labels = []
        for role, cnt in role_counter.items():
            role_labels.extend([role] * cnt)
        role_variety = self._normalised_entropy(role_labels) if role_labels else 0.0

        character_diversity = _clamp(0.5 * name_uniqueness + 0.5 * role_variety)

        return {
            "name_uniqueness": float(name_uniqueness),
            "role_variety": float(role_variety),
            "character_diversity": float(character_diversity),
        }

    def measure_world_building_diversity(
        self, texts: List[str]
    ) -> Dict[str, float]:
        """Assess setting and world-building diversity.

        Extracts location names, setting keywords, and sensory
        environment descriptors, then computes diversity over the corpus.
        """
        if not texts:
            return {
                "setting_diversity": 0.0,
                "environment_richness": 0.0,
                "world_building_diversity": 0.0,
            }

        all_settings: List[Set[str]] = []
        environment_scores: List[float] = []

        for text in texts:
            settings = self._extract_settings(text)
            all_settings.append(settings)

            # Environment richness: how many distinct setting-category
            # words appear?
            tokens = set(_tokenize(text))
            env_hits = tokens & _SETTING_COMMON_WORDS
            env_score = _clamp(len(env_hits) / max(len(_SETTING_COMMON_WORDS) * 0.15, 1))
            environment_scores.append(env_score)

        setting_diversity = self._mean_jaccard_distance(all_settings)

        # Environment richness variance — higher variance → more diversity
        env_arr = np.array(environment_scores)
        if len(env_arr) > 1:
            env_mean = float(np.mean(env_arr))
            env_std = float(np.std(env_arr))
            environment_richness = _clamp(env_mean + env_std)
        else:
            environment_richness = float(env_arr[0]) if len(env_arr) else 0.0

        world_building_diversity = _clamp(
            0.6 * setting_diversity + 0.4 * environment_richness
        )

        return {
            "setting_diversity": float(setting_diversity),
            "environment_richness": float(environment_richness),
            "world_building_diversity": float(world_building_diversity),
        }

    def compute_story_coherence_scores(
        self, texts: List[str]
    ) -> Dict[str, Any]:
        """Score coherence for each text and return per-text and aggregate.

        Coherence is approximated via sliding-window vocabulary overlap
        between adjacent paragraphs, sentence-transition smoothness,
        and pronoun-referent consistency.
        """
        if not texts:
            return {"mean_coherence": 0.0, "per_text": []}

        per_text: List[float] = []
        for text in texts:
            score = self._single_coherence(text)
            per_text.append(score)

        mean_coherence = float(np.mean(per_text))
        return {
            "mean_coherence": mean_coherence,
            "per_text": per_text,
        }

    def genre_specific_diversity(
        self, texts: List[str], genre: str
    ) -> Dict[str, float]:
        """Genre-aware diversity that considers genre vocabulary usage.

        Measures how differently each text employs the genre's expected
        vocabulary, and how much each text deviates in overall lexical
        profile from the genre archetype.
        """
        genre_key = genre.lower().replace(" ", "-")
        vocab = _GENRE_VOCAB.get(genre_key, set())

        if not texts or not vocab:
            return {
                "genre_vocab_diversity": 0.0,
                "genre_deviation_spread": 0.0,
                "genre_diversity": 0.0,
            }

        genre_profiles: List[np.ndarray] = []
        vocab_list = sorted(vocab)

        for text in texts:
            tokens = _tokenize(text)
            token_set = set(tokens)
            total = max(len(tokens), 1)
            profile = np.array(
                [tokens.count(w) / total for w in vocab_list],
                dtype=np.float64,
            )
            genre_profiles.append(profile)

        matrix = np.array(genre_profiles)

        # Genre vocab diversity — mean pairwise L2 distance
        n = len(texts)
        if n > 1:
            dists: List[float] = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = float(np.linalg.norm(matrix[i] - matrix[j]))
                    dists.append(d)
            mean_dist = float(np.mean(dists))
            # Normalise by max possible distance
            max_possible = math.sqrt(len(vocab_list))
            genre_vocab_diversity = _clamp(mean_dist / max(max_possible * 0.1, 1e-9))
        else:
            genre_vocab_diversity = 0.0

        # Genre deviation spread — std of per-text genre-hit ratios
        hit_ratios: List[float] = []
        for text in texts:
            tokens = set(_tokenize(text))
            hits = tokens & vocab
            hit_ratios.append(len(hits) / max(len(vocab), 1))
        ratio_arr = np.array(hit_ratios)
        genre_deviation_spread = _clamp(float(np.std(ratio_arr)) * 5.0)

        genre_diversity = _clamp(
            0.6 * genre_vocab_diversity + 0.4 * genre_deviation_spread
        )

        return {
            "genre_vocab_diversity": float(genre_vocab_diversity),
            "genre_deviation_spread": float(genre_deviation_spread),
            "genre_diversity": float(genre_diversity),
        }

    def compute_continuation_diversity(
        self,
        originals: List[str],
        continuations: List[str],
    ) -> Dict[str, float]:
        """Measure diversity among continuations of the same originals.

        Evaluates how differently each continuation departs from its
        source while remaining coherent, and how varied the set of
        continuations is as a whole.
        """
        if not originals or not continuations:
            return {
                "departure_diversity": 0.0,
                "continuation_spread": 0.0,
                "coherence_retention": 0.0,
                "continuation_diversity": 0.0,
            }

        n = min(len(originals), len(continuations))

        # Departure scores — how much new vocabulary each continuation adds
        departure_scores: List[float] = []
        coherence_scores: List[float] = []
        for i in range(n):
            orig_tokens = set(_tokenize(originals[i]))
            cont_tokens = set(_tokenize(continuations[i]))
            if not cont_tokens:
                departure_scores.append(0.0)
                coherence_scores.append(0.0)
                continue
            new_words = cont_tokens - orig_tokens
            departure = len(new_words) / max(len(cont_tokens), 1)
            departure_scores.append(departure)

            # Coherence retention — overlap between original and continuation
            overlap = orig_tokens & cont_tokens
            coherence = len(overlap) / max(len(orig_tokens), 1)
            coherence_scores.append(_clamp(coherence * 2.0))

        departure_arr = np.array(departure_scores)
        departure_diversity = _clamp(float(np.std(departure_arr)) * 4.0)

        # Continuation spread — pairwise Jaccard distance among continuations
        cont_token_sets = [set(_tokenize(c)) for c in continuations[:n]]
        continuation_spread = self._mean_jaccard_distance(cont_token_sets)

        coherence_retention = float(np.mean(coherence_scores)) if coherence_scores else 0.0

        continuation_diversity = _clamp(
            0.35 * departure_diversity
            + 0.40 * continuation_spread
            + 0.25 * coherence_retention
        )

        return {
            "departure_diversity": float(departure_diversity),
            "continuation_spread": float(continuation_spread),
            "coherence_retention": float(coherence_retention),
            "continuation_diversity": float(continuation_diversity),
        }

    # ------------------------------------------------------------------
    # Private helpers — feature extraction
    # ------------------------------------------------------------------

    def _extract_arc_vector(self, text: str) -> np.ndarray:
        """Return a feature vector summarising narrative-arc structure."""
        sents = _sentences(text)
        n_sents = max(len(sents), 1)
        paras = _paragraphs(text)
        n_paras = len(paras)

        if n_sents < 3:
            return np.zeros(7, dtype=np.float64)

        third = max(n_sents // 3, 1)
        beginning = " ".join(sents[:third])
        middle = " ".join(sents[third: 2 * third])
        ending = " ".join(sents[2 * third:])

        begin_hits = _count_pattern_hits(beginning, _NARRATIVE_ARC_BEGINNING)
        climax_hits = _count_pattern_hits(middle, _NARRATIVE_ARC_CLIMAX)
        resol_hits = _count_pattern_hits(ending, _NARRATIVE_ARC_RESOLUTION)

        # Sentence length variance
        sent_lens = np.array([len(_tokenize(s)) for s in sents], dtype=np.float64)
        len_mean = float(np.mean(sent_lens)) if len(sent_lens) else 0.0
        len_std = float(np.std(sent_lens)) if len(sent_lens) else 0.0

        return np.array(
            [
                _sigmoid(begin_hits, midpoint=1.0, steepness=1.5),
                _sigmoid(climax_hits, midpoint=1.0, steepness=1.5),
                _sigmoid(resol_hits, midpoint=1.0, steepness=1.5),
                _clamp(n_paras / 10.0),
                _clamp(len_mean / 30.0),
                _clamp(len_std / 15.0),
                _clamp(n_sents / 50.0),
            ],
            dtype=np.float64,
        )

    def _discretise_arcs(
        self, arc_vectors: List[np.ndarray]
    ) -> List[str]:
        """Assign each arc vector a discrete label based on dominant signal."""
        labels: List[str] = []
        for vec in arc_vectors:
            if len(vec) < 3:
                labels.append("flat")
                continue
            begin, climax, resol = vec[0], vec[1], vec[2]
            dominant = max(begin, climax, resol)
            if dominant < 0.2:
                labels.append("flat")
            elif begin == dominant:
                labels.append("setup_heavy")
            elif climax == dominant:
                labels.append("climax_heavy")
            else:
                labels.append("resolution_heavy")
            # Refinement: balanced arcs
            if min(begin, climax, resol) > 0.3:
                labels[-1] = "balanced"
        return labels

    def _extract_character_names(self, text: str) -> Set[str]:
        """Extract probable character names from *text*."""
        # Find capitalised multi-letter words that are not sentence starters
        candidates: Set[str] = set()
        sents = _sentences(text)
        stopwords = {
            "The", "This", "That", "These", "Those", "There", "Here",
            "When", "Where", "What", "Which", "Who", "How", "Why",
            "Once", "After", "Before", "During", "While", "Although",
            "Because", "Since", "Until", "Unless", "However", "But",
            "And", "Yet", "Still", "Then", "Now", "Soon", "Later",
            "Finally", "Meanwhile", "Suddenly", "Perhaps", "Maybe",
            "Indeed", "Certainly", "Also", "Instead", "Moreover",
            "Furthermore", "Nevertheless", "Nonetheless", "Chapter",
        }
        for sent in sents:
            # Skip the very first word (likely sentence-initial cap)
            words = sent.split()
            for word in words[1:]:
                clean = word.strip(string.punctuation)
                if (
                    clean
                    and clean[0].isupper()
                    and len(clean) >= 3
                    and clean not in stopwords
                    and not clean.isupper()
                ):
                    candidates.add(clean)

        # Also pick up names via the dedicated pattern
        for match in _CHARACTER_NAME_PAT.finditer(text):
            name = match.group(1)
            if name not in stopwords and len(name) >= 3:
                candidates.add(name)

        return candidates

    def _extract_character_roles(self, text: str) -> Set[str]:
        """Extract character role keywords from *text*."""
        role_patterns: Dict[str, str] = {
            "hero": r"\bhero(?:ine|ic)?\b",
            "villain": r"\bvillain(?:ous|y)?\b",
            "mentor": r"\bmentor\b",
            "sidekick": r"\bsidekick\b",
            "leader": r"\bleader\b",
            "warrior": r"\bwarrior\b",
            "healer": r"\bhealer\b",
            "thief": r"\bthief\b",
            "king": r"\bking\b",
            "queen": r"\bqueen\b",
            "prince": r"\bprince\b",
            "princess": r"\bprincess\b",
            "detective": r"\bdetective\b",
            "scientist": r"\bscientist\b",
            "captain": r"\bcaptain\b",
            "prophet": r"\bprophet\b",
            "guardian": r"\bguardian\b",
            "stranger": r"\bstranger\b",
            "child": r"\bchild\b",
            "elder": r"\belder\b",
            "mother": r"\bmother\b",
            "father": r"\bfather\b",
            "teacher": r"\bteacher\b",
            "student": r"\bstudent\b",
            "soldier": r"\bsoldier\b",
            "merchant": r"\bmerchant\b",
            "outcast": r"\boutcast\b",
            "rebel": r"\brebel\b",
            "narrator": r"\bnarrator\b",
            "lover": r"\blover\b",
        }
        found: Set[str] = set()
        text_lower = text.lower()
        for role, pat in role_patterns.items():
            if re.search(pat, text_lower):
                found.add(role)
        return found

    def _extract_settings(self, text: str) -> Set[str]:
        """Extract setting locations from *text*."""
        settings: Set[str] = set()

        # Named locations via preposition patterns
        for pat in _SETTING_MARKERS:
            for match in re.finditer(pat, text):
                location = match.group(1).strip()
                if len(location) >= 3:
                    settings.add(location.lower())

        # Common setting words present in the text
        tokens = set(_tokenize(text))
        for word in tokens & _SETTING_COMMON_WORDS:
            settings.add(word)

        return settings

    def _extract_plot_elements(self, text: str) -> Dict[str, int]:
        """Count plot-element categories present in *text*."""
        results: Dict[str, int] = {}
        text_lower = text.lower()
        for category, patterns in _PLOT_ELEMENT_PATTERNS.items():
            hits = 0
            for pat in patterns:
                hits += len(re.findall(pat, text_lower))
            results[category] = hits
        return results

    # ------------------------------------------------------------------
    # Private helpers — diversity statistics
    # ------------------------------------------------------------------

    def _mean_jaccard_distance(
        self, sets_list: List[Set[str]]
    ) -> float:
        """Compute mean pairwise Jaccard distance over a list of sets."""
        n = len(sets_list)
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b = sets_list[i], sets_list[j]
                union = a | b
                if not union:
                    continue
                intersection = a & b
                jaccard_sim = len(intersection) / len(union)
                total += 1.0 - jaccard_sim
                count += 1
        return _clamp(total / max(count, 1))

    @staticmethod
    def _normalised_entropy(labels: List[str]) -> float:
        """Shannon entropy of *labels*, normalised to [0, 1]."""
        if not labels:
            return 0.0
        freq = Counter(labels)
        n = len(labels)
        num_classes = len(freq)
        if num_classes <= 1:
            return 0.0
        entropy = 0.0
        for count in freq.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(num_classes)
        return _clamp(entropy / max_entropy) if max_entropy > 0 else 0.0

    def _single_coherence(self, text: str) -> float:
        """Compute a coherence score for a single text.

        Uses paragraph-overlap and sentence-transition smoothness.
        """
        paras = _paragraphs(text)
        if len(paras) < 2:
            # With a single paragraph, check sentence-level coherence
            sents = _sentences(text)
            if len(sents) < 2:
                return 0.5
            return self._sentence_transition_score(sents)

        # Paragraph overlap — adjacent paragraph vocabulary overlap
        para_token_sets = [set(_tokenize(p)) for p in paras]
        overlaps: List[float] = []
        for i in range(len(para_token_sets) - 1):
            a, b = para_token_sets[i], para_token_sets[i + 1]
            union = a | b
            if not union:
                overlaps.append(0.0)
                continue
            inter = a & b
            overlaps.append(len(inter) / len(union))

        para_coherence = float(np.mean(overlaps)) if overlaps else 0.0

        # Sentence transition smoothness across entire text
        sents = _sentences(text)
        sent_coherence = self._sentence_transition_score(sents)

        # Pronoun consistency — ratio of pronouns that have plausible
        # antecedents (heuristic: a pronoun's antecedent is a proper noun
        # within the prior two sentences)
        pronoun_score = self._pronoun_consistency(sents)

        return _clamp(
            0.4 * para_coherence + 0.35 * sent_coherence + 0.25 * pronoun_score
        )

    def _sentence_transition_score(self, sents: List[str]) -> float:
        """Average bigram overlap between consecutive sentences."""
        if len(sents) < 2:
            return 0.5
        overlaps: List[float] = []
        prev_tokens = set(_tokenize(sents[0]))
        for sent in sents[1:]:
            curr_tokens = set(_tokenize(sent))
            union = prev_tokens | curr_tokens
            if union:
                inter = prev_tokens & curr_tokens
                overlaps.append(len(inter) / len(union))
            else:
                overlaps.append(0.0)
            prev_tokens = curr_tokens
        return _clamp(float(np.mean(overlaps)) * 3.0)

    def _pronoun_consistency(self, sents: List[str]) -> float:
        """Heuristic pronoun-referent consistency check."""
        pronouns = {"he", "she", "they", "it", "him", "her", "them", "his", "its"}
        total_pronouns = 0
        resolved = 0

        for idx, sent in enumerate(sents):
            tokens = set(_tokenize(sent))
            sent_pronouns = tokens & pronouns
            if not sent_pronouns:
                continue
            total_pronouns += len(sent_pronouns)
            # Look for proper nouns in current and previous two sentences
            context_start = max(0, idx - 2)
            context = " ".join(sents[context_start: idx + 1])
            names = _CHARACTER_NAME_PAT.findall(context)
            if names:
                resolved += len(sent_pronouns)

        if total_pronouns == 0:
            return 0.7  # no pronouns — neutral score
        return _clamp(resolved / total_pronouns)
