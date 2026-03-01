"""
Base classes for task domains in the Diversity Decoding Arena.

Provides foundational abstractions for defining generation tasks, prompts,
constraints, datasets, and evaluation logic. Concrete task implementations
(creative writing, code generation, etc.) inherit from :class:`GenerationTask`
and supply domain-specific prompts, formatting, and evaluation metrics.

Architecture
------------
::

    TaskConfig          -- static configuration for a task run
    TaskPrompt          -- single prompt instance with metadata
    TaskConstraint      -- enforceable constraint on generated text
    PromptDataset       -- iterable collection of TaskPrompts
    TaskEvaluator       -- computes quality/diversity scores
    GenerationTask      -- abstract base that ties everything together

All dataclasses are JSON-serialisable via their ``to_dict`` / ``from_dict``
helpers so that experiment configurations can be checkpointed and reproduced.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
import uuid
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from itertools import combinations
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np

from src.types import TaskDomain

# ---------------------------------------------------------------------------
# Module-level task registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY: Dict[str, Type["GenerationTask"]] = {}


def get_registered_tasks() -> Dict[str, Type["GenerationTask"]]:
    """Return a shallow copy of the global task registry."""
    return dict(_TASK_REGISTRY)


# ---------------------------------------------------------------------------
# Constraint types
# ---------------------------------------------------------------------------


class ConstraintType(Enum):
    """Categories of constraints that can be applied to generated text."""

    LENGTH = auto()
    FORMAT = auto()
    CONTENT = auto()
    STYLE = auto()
    KEYWORD = auto()

    def __repr__(self) -> str:
        return f"ConstraintType.{self.name}"


# ---------------------------------------------------------------------------
# TaskConstraint
# ---------------------------------------------------------------------------


@dataclass
class TaskConstraint:
    """A single enforceable constraint on generated text.

    Parameters
    ----------
    constraint_type : ConstraintType
        The category of constraint.
    parameters : dict
        Constraint-specific parameters.  For example a ``LENGTH`` constraint
        might have ``{"min": 50, "max": 200}`` while a ``KEYWORD`` constraint
        might have ``{"keywords": ["apple", "banana"], "mode": "any"}``.
    required : bool
        If ``True`` the constraint *must* be satisfied for the generation to
        be considered valid.
    weight : float
        Relative importance used when computing a weighted constraint score.
    """

    constraint_type: ConstraintType
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: bool = True
    weight: float = 1.0

    # -----------------------------------------------------------------
    # Core API
    # -----------------------------------------------------------------

    def check(self, text: str) -> bool:
        """Return ``True`` if *text* satisfies this constraint.

        Dispatches to a private handler based on :pyattr:`constraint_type`.
        Unknown constraint types always return ``True`` (open-world assumption).
        """
        handler = {
            ConstraintType.LENGTH: self._check_length,
            ConstraintType.FORMAT: self._check_format,
            ConstraintType.CONTENT: self._check_content,
            ConstraintType.STYLE: self._check_style,
            ConstraintType.KEYWORD: self._check_keyword,
        }.get(self.constraint_type)
        if handler is None:
            return True
        return handler(text)

    # -----------------------------------------------------------------
    # Private constraint checkers
    # -----------------------------------------------------------------

    def _check_length(self, text: str) -> bool:
        """Verify length constraints (character or word count)."""
        unit = self.parameters.get("unit", "words")
        if unit == "characters":
            length = len(text)
        elif unit == "sentences":
            length = len(re.split(r"[.!?]+", text.strip()))
        else:
            length = len(text.split())

        min_len = self.parameters.get("min", 0)
        max_len = self.parameters.get("max", float("inf"))
        return min_len <= length <= max_len

    def _check_format(self, text: str) -> bool:
        """Verify format constraints (regex, structure, etc.)."""
        pattern = self.parameters.get("pattern")
        if pattern:
            if not re.search(pattern, text, re.DOTALL):
                return False

        required_sections = self.parameters.get("required_sections", [])
        for section in required_sections:
            if section.lower() not in text.lower():
                return False

        starts_with = self.parameters.get("starts_with")
        if starts_with and not text.strip().startswith(starts_with):
            return False

        ends_with = self.parameters.get("ends_with")
        if ends_with and not text.strip().endswith(ends_with):
            return False

        return True

    def _check_content(self, text: str) -> bool:
        """Verify content constraints (topics, banned words, etc.)."""
        banned_words = self.parameters.get("banned_words", [])
        text_lower = text.lower()
        for word in banned_words:
            if word.lower() in text_lower:
                return False

        required_topics = self.parameters.get("required_topics", [])
        for topic in required_topics:
            if topic.lower() not in text_lower:
                return False

        min_unique_words = self.parameters.get("min_unique_words", 0)
        if min_unique_words > 0:
            unique = len(set(text.lower().split()))
            if unique < min_unique_words:
                return False

        max_repetition_ratio = self.parameters.get("max_repetition_ratio", 1.0)
        words = text.lower().split()
        if words:
            counts = Counter(words)
            most_common_count = counts.most_common(1)[0][1]
            ratio = most_common_count / len(words)
            if ratio > max_repetition_ratio:
                return False

        return True

    def _check_style(self, text: str) -> bool:
        """Verify style constraints (tone, reading level, etc.)."""
        max_sentence_length = self.parameters.get("max_sentence_length")
        if max_sentence_length is not None:
            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
            for sentence in sentences:
                if len(sentence.split()) > max_sentence_length:
                    return False

        min_sentence_length = self.parameters.get("min_sentence_length")
        if min_sentence_length is not None:
            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
            for sentence in sentences:
                if len(sentence.split()) < min_sentence_length:
                    return False

        prohibited_patterns = self.parameters.get("prohibited_patterns", [])
        for pat in prohibited_patterns:
            if re.search(pat, text):
                return False

        required_punctuation = self.parameters.get("required_punctuation", [])
        for punct in required_punctuation:
            if punct not in text:
                return False

        persona = self.parameters.get("persona")
        if persona == "formal":
            informal_markers = ["lol", "omg", "gonna", "wanna", "idk", "btw"]
            text_lower = text.lower()
            for marker in informal_markers:
                if re.search(rf"\b{marker}\b", text_lower):
                    return False

        return True

    def _check_keyword(self, text: str) -> bool:
        """Verify keyword constraints (inclusion/exclusion)."""
        keywords = self.parameters.get("keywords", [])
        mode = self.parameters.get("mode", "all")
        case_sensitive = self.parameters.get("case_sensitive", False)

        if not keywords:
            return True

        check_text = text if case_sensitive else text.lower()
        check_keywords = keywords if case_sensitive else [k.lower() for k in keywords]

        if mode == "all":
            return all(kw in check_text for kw in check_keywords)
        elif mode == "any":
            return any(kw in check_text for kw in check_keywords)
        elif mode == "none":
            return not any(kw in check_text for kw in check_keywords)
        elif mode == "exact_count":
            expected = self.parameters.get("count", len(keywords))
            found = sum(1 for kw in check_keywords if kw in check_text)
            return found == expected
        return True

    # -----------------------------------------------------------------
    # Serialisation helpers
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-friendly dictionary."""
        return {
            "constraint_type": self.constraint_type.name,
            "parameters": self.parameters,
            "required": self.required,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConstraint":
        """Reconstruct from a dictionary produced by :meth:`to_dict`."""
        return cls(
            constraint_type=ConstraintType[data["constraint_type"]],
            parameters=data.get("parameters", {}),
            required=data.get("required", True),
            weight=data.get("weight", 1.0),
        )

    def describe(self) -> str:
        """Human-readable one-line description of this constraint."""
        parts = [f"{self.constraint_type.name}"]
        if self.parameters:
            param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
            parts.append(f"({param_str})")
        if not self.required:
            parts.append("[optional]")
        if self.weight != 1.0:
            parts.append(f"weight={self.weight:.2f}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------


@dataclass
class TaskConfig:
    """Static configuration for a task run.

    Parameters
    ----------
    name : str
        Human-readable task name (e.g. ``"creative-writing-short"``).
    domain : TaskDomain
        The broad generation domain.
    num_prompts : int
        Number of prompts to use from the dataset.
    max_length : int
        Maximum generation length in tokens.
    min_length : int
        Minimum generation length in tokens.
    temperature : float
        Default sampling temperature.
    constraints : List[TaskConstraint]
        Global constraints applied to every generation.
    evaluation_metrics : List[str]
        Names of metrics to compute during evaluation.
    prompt_template : str
        A Python format-string applied to each prompt before generation.
    seed : int
        Random seed for reproducibility.
    """

    name: str = "default"
    domain: TaskDomain = TaskDomain.OPEN_ENDED_GENERATION
    num_prompts: int = 100
    max_length: int = 512
    min_length: int = 10
    temperature: float = 1.0
    constraints: List[TaskConstraint] = field(default_factory=list)
    evaluation_metrics: List[str] = field(
        default_factory=lambda: ["fluency", "relevance", "diversity"]
    )
    prompt_template: str = "{text}"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "domain": self.domain.name,
            "num_prompts": self.num_prompts,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "temperature": self.temperature,
            "constraints": [c.to_dict() for c in self.constraints],
            "evaluation_metrics": list(self.evaluation_metrics),
            "prompt_template": self.prompt_template,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        """Reconstruct from a dictionary."""
        constraints = [
            TaskConstraint.from_dict(c) for c in data.get("constraints", [])
        ]
        return cls(
            name=data.get("name", "default"),
            domain=TaskDomain[data["domain"]],
            num_prompts=data.get("num_prompts", 100),
            max_length=data.get("max_length", 512),
            min_length=data.get("min_length", 10),
            temperature=data.get("temperature", 1.0),
            constraints=constraints,
            evaluation_metrics=data.get(
                "evaluation_metrics", ["fluency", "relevance", "diversity"]
            ),
            prompt_template=data.get("prompt_template", "{text}"),
            seed=data.get("seed", 42),
        )

    def validate(self) -> List[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: List[str] = []
        if self.max_length <= 0:
            errors.append("max_length must be positive")
        if self.min_length < 0:
            errors.append("min_length must be non-negative")
        if self.min_length > self.max_length:
            errors.append("min_length must not exceed max_length")
        if self.temperature <= 0:
            errors.append("temperature must be positive")
        if self.num_prompts <= 0:
            errors.append("num_prompts must be positive")
        return errors


# ---------------------------------------------------------------------------
# TaskPrompt
# ---------------------------------------------------------------------------


@dataclass
class TaskPrompt:
    """A single prompt instance with metadata and optional reference outputs.

    Parameters
    ----------
    prompt_id : str
        Unique identifier.
    text : str
        The raw prompt text.
    context : str
        Additional context provided alongside the prompt.
    metadata : Dict[str, Any]
        Arbitrary key/value metadata (source corpus, difficulty, etc.).
    reference_outputs : List[str]
        Gold / reference outputs for evaluation.
    domain : TaskDomain
        Domain this prompt belongs to.
    constraints : List[TaskConstraint]
        Prompt-level constraints (in addition to any global constraints).
    max_gen_length : int
        Maximum generation length specific to this prompt.
    """

    prompt_id: str = ""
    text: str = ""
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    reference_outputs: List[str] = field(default_factory=list)
    domain: TaskDomain = TaskDomain.OPEN_ENDED_GENERATION
    constraints: List[TaskConstraint] = field(default_factory=list)
    max_gen_length: int = 512

    def __post_init__(self) -> None:
        if not self.prompt_id:
            self.prompt_id = self._generate_id()

    def _generate_id(self) -> str:
        """Derive a deterministic ID from the prompt text."""
        content = self.text + self.context
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    @property
    def word_count(self) -> int:
        """Number of whitespace-delimited words in the prompt text."""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """Number of characters in the prompt text."""
        return len(self.text)

    def has_reference(self) -> bool:
        """Return ``True`` if at least one reference output exists."""
        return len(self.reference_outputs) > 0

    def all_constraints(self, global_constraints: Optional[List[TaskConstraint]] = None) -> List[TaskConstraint]:
        """Merge prompt-level and global constraints."""
        base = list(global_constraints) if global_constraints else []
        return base + list(self.constraints)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "prompt_id": self.prompt_id,
            "text": self.text,
            "context": self.context,
            "metadata": self.metadata,
            "reference_outputs": self.reference_outputs,
            "domain": self.domain.name,
            "constraints": [c.to_dict() for c in self.constraints],
            "max_gen_length": self.max_gen_length,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskPrompt":
        """Reconstruct from a dictionary."""
        constraints = [
            TaskConstraint.from_dict(c) for c in data.get("constraints", [])
        ]
        return cls(
            prompt_id=data.get("prompt_id", ""),
            text=data.get("text", ""),
            context=data.get("context", ""),
            metadata=data.get("metadata", {}),
            reference_outputs=data.get("reference_outputs", []),
            domain=TaskDomain[data["domain"]] if "domain" in data else TaskDomain.OPEN_ENDED_GENERATION,
            constraints=constraints,
            max_gen_length=data.get("max_gen_length", 512),
        )


# ---------------------------------------------------------------------------
# PromptDataset
# ---------------------------------------------------------------------------


class PromptDataset:
    """An iterable, sliceable collection of :class:`TaskPrompt` objects.

    Supports sampling, filtering, splitting, and serialisation.

    Parameters
    ----------
    prompts : Sequence[TaskPrompt]
        The prompts to wrap.
    name : str
        Human-readable name for this dataset.
    domain : TaskDomain
        The domain all prompts belong to.
    """

    def __init__(
        self,
        prompts: Sequence[TaskPrompt],
        name: str = "unnamed",
        domain: TaskDomain = TaskDomain.OPEN_ENDED_GENERATION,
    ) -> None:
        self._prompts: List[TaskPrompt] = list(prompts)
        self.name = name
        self.domain = domain

    # -----------------------------------------------------------------
    # Container protocol
    # -----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._prompts)

    def __getitem__(self, index: Union[int, slice]) -> Union[TaskPrompt, "PromptDataset"]:
        if isinstance(index, slice):
            return PromptDataset(self._prompts[index], name=self.name, domain=self.domain)
        return self._prompts[index]

    def __iter__(self) -> Iterator[TaskPrompt]:
        return iter(self._prompts)

    def __repr__(self) -> str:
        return f"PromptDataset(name={self.name!r}, size={len(self)}, domain={self.domain.name})"

    def __add__(self, other: "PromptDataset") -> "PromptDataset":
        return PromptDataset(
            self._prompts + other._prompts,
            name=f"{self.name}+{other.name}",
            domain=self.domain,
        )

    # -----------------------------------------------------------------
    # Sampling & filtering
    # -----------------------------------------------------------------

    def sample(self, n: int, seed: Optional[int] = None) -> List[TaskPrompt]:
        """Return *n* prompts sampled uniformly without replacement.

        Parameters
        ----------
        n : int
            Number of prompts to sample.  Clamped to dataset size.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        List[TaskPrompt]
        """
        rng = np.random.RandomState(seed)
        n = min(n, len(self._prompts))
        indices = rng.choice(len(self._prompts), size=n, replace=False)
        return [self._prompts[i] for i in indices]

    def sample_dataset(self, n: int, seed: Optional[int] = None) -> "PromptDataset":
        """Like :meth:`sample` but returns a new :class:`PromptDataset`."""
        sampled = self.sample(n, seed=seed)
        return PromptDataset(sampled, name=f"{self.name}_sample_{n}", domain=self.domain)

    def filter_by(self, predicate: Callable[[TaskPrompt], bool]) -> "PromptDataset":
        """Return a new dataset containing only prompts that satisfy *predicate*.

        Parameters
        ----------
        predicate : Callable[[TaskPrompt], bool]
            A function that returns ``True`` for prompts to keep.

        Returns
        -------
        PromptDataset
        """
        filtered = [p for p in self._prompts if predicate(p)]
        return PromptDataset(filtered, name=f"{self.name}_filtered", domain=self.domain)

    def filter_by_domain(self, domain: TaskDomain) -> "PromptDataset":
        """Convenience filter for a specific domain."""
        return self.filter_by(lambda p: p.domain == domain)

    def filter_by_length(self, min_words: int = 0, max_words: int = int(1e9)) -> "PromptDataset":
        """Keep only prompts whose word count is within the given range."""
        return self.filter_by(lambda p: min_words <= p.word_count <= max_words)

    # -----------------------------------------------------------------
    # Splitting
    # -----------------------------------------------------------------

    def split(
        self,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: Optional[int] = None,
    ) -> Tuple["PromptDataset", "PromptDataset", "PromptDataset"]:
        """Split into train / validation / test datasets.

        Parameters
        ----------
        train_frac, val_frac, test_frac : float
            Must sum to 1.0 (tolerance 1e-6).
        seed : int, optional
            Random seed for shuffling before splitting.

        Returns
        -------
        Tuple[PromptDataset, PromptDataset, PromptDataset]
        """
        total = train_frac + val_frac + test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Fractions must sum to 1.0, got {total:.6f}"
            )

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self._prompts))

        n = len(self._prompts)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        def _subset(idx: np.ndarray, suffix: str) -> "PromptDataset":
            return PromptDataset(
                [self._prompts[i] for i in idx],
                name=f"{self.name}_{suffix}",
                domain=self.domain,
            )

        return _subset(train_idx, "train"), _subset(val_idx, "val"), _subset(test_idx, "test")

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise the entire dataset to a JSON string."""
        payload = {
            "name": self.name,
            "domain": self.domain.name,
            "prompts": [p.to_dict() for p in self._prompts],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "PromptDataset":
        """Reconstruct a dataset from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON produced by :meth:`to_json`.

        Returns
        -------
        PromptDataset
        """
        data = json.loads(json_str)
        prompts = [TaskPrompt.from_dict(p) for p in data.get("prompts", [])]
        domain = TaskDomain[data["domain"]] if "domain" in data else TaskDomain.OPEN_ENDED_GENERATION
        return cls(prompts=prompts, name=data.get("name", "unnamed"), domain=domain)

    def to_file(self, path: str) -> None:
        """Write the dataset to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def from_file(cls, path: str) -> "PromptDataset":
        """Load a dataset from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_json(fh.read())

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        """Compute descriptive statistics over the prompt texts.

        Returns
        -------
        dict
            Keys include ``count``, ``length_*`` (word-level stats),
            ``char_length_*``, ``vocab_size``, ``vocab_richness``,
            ``has_reference_frac``, and ``domain_distribution``.
        """
        if not self._prompts:
            return {"count": 0}

        word_lengths = np.array([p.word_count for p in self._prompts], dtype=float)
        char_lengths = np.array([p.char_count for p in self._prompts], dtype=float)

        # Vocabulary statistics across all prompts
        all_words: List[str] = []
        for p in self._prompts:
            all_words.extend(p.text.lower().split())
        vocab = set(all_words)
        word_counts = Counter(all_words)
        total_words = len(all_words)

        # Domain distribution
        domain_counts: Dict[str, int] = defaultdict(int)
        for p in self._prompts:
            domain_counts[p.domain.name] += 1

        # Reference output coverage
        ref_count = sum(1 for p in self._prompts if p.has_reference())

        return {
            "count": len(self._prompts),
            # Word-level length statistics
            "length_mean": float(np.mean(word_lengths)),
            "length_std": float(np.std(word_lengths)),
            "length_min": int(np.min(word_lengths)),
            "length_max": int(np.max(word_lengths)),
            "length_median": float(np.median(word_lengths)),
            "length_p25": float(np.percentile(word_lengths, 25)),
            "length_p75": float(np.percentile(word_lengths, 75)),
            # Character-level length statistics
            "char_length_mean": float(np.mean(char_lengths)),
            "char_length_std": float(np.std(char_lengths)),
            "char_length_min": int(np.min(char_lengths)),
            "char_length_max": int(np.max(char_lengths)),
            # Vocabulary statistics
            "vocab_size": len(vocab),
            "total_words": total_words,
            "vocab_richness": len(vocab) / total_words if total_words > 0 else 0.0,
            "top_10_words": word_counts.most_common(10),
            "hapax_legomena": sum(1 for w, c in word_counts.items() if c == 1),
            # Coverage
            "has_reference_frac": ref_count / len(self._prompts),
            "domain_distribution": dict(domain_counts),
        }


# ---------------------------------------------------------------------------
# TaskEvaluator
# ---------------------------------------------------------------------------


class TaskEvaluator:
    """Evaluates generated text against prompts and reference outputs.

    Computes fluency, relevance, constraint satisfaction, and set-level
    diversity scores.

    Parameters
    ----------
    metrics_config : Dict[str, Any]
        Configuration for each metric.  Keys are metric names, values are
        dicts of metric-specific parameters.
    """

    def __init__(self, metrics_config: Optional[Dict[str, Any]] = None) -> None:
        self.metrics_config = metrics_config or {}
        self._metric_cache: Dict[str, float] = {}

    # -----------------------------------------------------------------
    # Single-generation evaluation
    # -----------------------------------------------------------------

    def evaluate_single(
        self,
        generation: str,
        prompt: TaskPrompt,
        reference: Optional[str] = None,
    ) -> Dict[str, float]:
        """Score a single generation against its prompt and optional reference.

        Parameters
        ----------
        generation : str
            The generated text.
        prompt : TaskPrompt
            The prompt that produced the generation.
        reference : str, optional
            A reference / gold output for comparison.

        Returns
        -------
        Dict[str, float]
            Metric name → score mapping.
        """
        scores: Dict[str, float] = {}

        scores["fluency"] = self.fluency_score(generation)
        scores["relevance"] = self.relevance_score(generation, prompt)
        scores["length_ratio"] = self._length_ratio(generation, prompt)

        if reference is not None:
            scores["reference_overlap"] = self._reference_overlap(generation, reference)
            scores["bleu_1gram"] = self._bleu_ngram(generation, reference, n=1)
            scores["bleu_2gram"] = self._bleu_ngram(generation, reference, n=2)

        constraint_results = self.check_constraints(generation, prompt.constraints)
        if constraint_results:
            scores["constraint_satisfaction"] = sum(constraint_results) / len(constraint_results)

        scores["lexical_diversity"] = self._type_token_ratio(generation)

        return scores

    # -----------------------------------------------------------------
    # Set-level evaluation
    # -----------------------------------------------------------------

    def evaluate_set(
        self,
        generations: List[str],
        prompts: List[TaskPrompt],
        references: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, float]:
        """Evaluate a set of generations and aggregate scores.

        Parameters
        ----------
        generations : List[str]
            Generated texts, one per prompt.
        prompts : List[TaskPrompt]
            Corresponding prompts.
        references : List[str], optional
            Corresponding reference outputs.

        Returns
        -------
        Dict[str, float]
            Aggregated metric scores including per-metric means and
            set-level diversity.
        """
        if len(generations) != len(prompts):
            raise ValueError(
                f"Length mismatch: {len(generations)} generations vs {len(prompts)} prompts"
            )

        refs = references or [None] * len(generations)
        all_scores: List[Dict[str, float]] = []

        for gen, prompt, ref in zip(generations, prompts, refs):
            all_scores.append(self.evaluate_single(gen, prompt, ref))

        # Aggregate per-metric
        aggregated: Dict[str, float] = {}
        if all_scores:
            metric_names = set()
            for s in all_scores:
                metric_names.update(s.keys())

            for metric in metric_names:
                values = [s[metric] for s in all_scores if metric in s]
                if values:
                    aggregated[f"{metric}_mean"] = float(np.mean(values))
                    aggregated[f"{metric}_std"] = float(np.std(values))
                    aggregated[f"{metric}_min"] = float(np.min(values))
                    aggregated[f"{metric}_max"] = float(np.max(values))

        # Set-level diversity
        aggregated["set_diversity"] = self.diversity_within_set(generations)
        aggregated["num_generations"] = float(len(generations))

        return aggregated

    # -----------------------------------------------------------------
    # Constraint checking
    # -----------------------------------------------------------------

    def check_constraints(
        self, text: str, constraints: List[TaskConstraint]
    ) -> List[bool]:
        """Check each constraint against *text*.

        Parameters
        ----------
        text : str
            The text to check.
        constraints : List[TaskConstraint]
            Constraints to evaluate.

        Returns
        -------
        List[bool]
            One boolean per constraint, in the same order.
        """
        return [c.check(text) for c in constraints]

    def weighted_constraint_score(
        self, text: str, constraints: List[TaskConstraint]
    ) -> float:
        """Return a weighted average constraint satisfaction score in [0, 1]."""
        if not constraints:
            return 1.0
        results = self.check_constraints(text, constraints)
        total_weight = sum(c.weight for c in constraints)
        if total_weight == 0:
            return 1.0
        weighted_sum = sum(
            c.weight * float(r) for c, r in zip(constraints, results)
        )
        return weighted_sum / total_weight

    # -----------------------------------------------------------------
    # Fluency scoring
    # -----------------------------------------------------------------

    def fluency_score(self, text: str) -> float:
        """Estimate the fluency of *text* using surface-level heuristics.

        The score is in [0, 1] where 1.0 indicates highly fluent text.  This
        is a lightweight proxy; production systems should plug in a trained
        model.

        Heuristics used:
        - Fraction of words that are dictionary-plausible (all-alpha or common
          punctuation patterns).
        - Sentence structure: at least one sentence-ending punctuation per ~20
          words.
        - Repetition penalty: excessive n-gram repetition lowers the score.
        - Capitalisation: first word of sentences should be capitalised.
        """
        if not text or not text.strip():
            return 0.0

        scores: List[float] = []

        # --- word plausibility ---
        words = text.split()
        if words:
            plausible = sum(
                1 for w in words
                if re.match(r"^[A-Za-z'-]+[.,;:!?]?$", w)
            )
            scores.append(plausible / len(words))

        # --- sentence structure ---
        sentence_enders = len(re.findall(r"[.!?]", text))
        expected_enders = max(1, len(words) / 20)
        structure_score = min(1.0, sentence_enders / expected_enders)
        scores.append(structure_score)

        # --- repetition penalty ---
        if len(words) >= 4:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
            bigram_counts = Counter(bigrams)
            if bigrams:
                max_rep = bigram_counts.most_common(1)[0][1]
                rep_ratio = max_rep / len(bigrams)
                scores.append(max(0.0, 1.0 - rep_ratio * 2))
        else:
            scores.append(1.0)

        # --- capitalisation ---
        sentences = re.split(r"[.!?]\s+", text.strip())
        if sentences:
            cap_count = sum(1 for s in sentences if s and s[0].isupper())
            scores.append(cap_count / len(sentences))

        return float(np.mean(scores)) if scores else 0.0

    # -----------------------------------------------------------------
    # Relevance scoring
    # -----------------------------------------------------------------

    def relevance_score(self, text: str, prompt: TaskPrompt) -> float:
        """Estimate how relevant *text* is to *prompt* using word overlap.

        Uses a combination of unigram overlap and keyword coverage relative
        to the prompt.  Returns a score in [0, 1].
        """
        if not text.strip() or not prompt.text.strip():
            return 0.0

        gen_words = set(text.lower().split())
        prompt_words = set(prompt.text.lower().split())

        # Remove very common words (stop-word approximation)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "shall", "should", "may", "might", "must", "can",
            "could", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "and", "but", "or", "nor", "not", "so", "yet", "both",
            "either", "neither", "each", "every", "all", "any", "few",
            "more", "most", "other", "some", "such", "no", "only", "own",
            "same", "than", "too", "very", "just", "because", "if", "when",
            "where", "how", "what", "which", "who", "whom", "this", "that",
            "these", "those", "i", "me", "my", "we", "our", "you", "your",
            "he", "him", "his", "she", "her", "it", "its", "they", "them",
            "their",
        }
        prompt_content = prompt_words - stop_words
        gen_content = gen_words - stop_words

        if not prompt_content:
            return 0.5  # no content words in prompt — neutral

        overlap = prompt_content & gen_content
        recall = len(overlap) / len(prompt_content)
        precision = len(overlap) / len(gen_content) if gen_content else 0.0

        if recall + precision == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)

        # Also consider context if available
        context_bonus = 0.0
        if prompt.context:
            ctx_words = set(prompt.context.lower().split()) - stop_words
            if ctx_words:
                ctx_overlap = ctx_words & gen_content
                context_bonus = 0.1 * len(ctx_overlap) / len(ctx_words)

        return min(1.0, f1 + context_bonus)

    # -----------------------------------------------------------------
    # Diversity scoring
    # -----------------------------------------------------------------

    def diversity_within_set(self, generations: List[str]) -> float:
        """Measure lexical diversity across a set of generations.

        Computes the mean pairwise Jaccard distance between the word sets of
        each pair of generations.  Returns a score in [0, 1] where 1.0 means
        completely disjoint vocabularies.

        Parameters
        ----------
        generations : List[str]
            The set of generated texts.

        Returns
        -------
        float
        """
        if len(generations) < 2:
            return 0.0

        word_sets = [set(g.lower().split()) for g in generations]

        distances: List[float] = []
        for i, j in combinations(range(len(word_sets)), 2):
            si, sj = word_sets[i], word_sets[j]
            union = si | sj
            if not union:
                distances.append(0.0)
                continue
            intersection = si & sj
            jaccard_dist = 1.0 - len(intersection) / len(union)
            distances.append(jaccard_dist)

        return float(np.mean(distances))

    def ngram_diversity(self, generations: List[str], n: int = 2) -> float:
        """Distinct-n metric: fraction of unique n-grams across all generations."""
        all_ngrams: List[Tuple[str, ...]] = []
        for gen in generations:
            words = gen.lower().split()
            for i in range(len(words) - n + 1):
                all_ngrams.append(tuple(words[i : i + n]))
        if not all_ngrams:
            return 0.0
        return len(set(all_ngrams)) / len(all_ngrams)

    def self_bleu(self, generations: List[str], n: int = 4) -> float:
        """Compute average Self-BLEU (lower is more diverse).

        For each generation, treats all others as references and computes
        a simplified BLEU.  Returns the average across generations.
        """
        if len(generations) < 2:
            return 0.0

        tokenised = [g.lower().split() for g in generations]
        bleu_scores: List[float] = []

        for i, hypothesis in enumerate(tokenised):
            refs = [tokenised[j] for j in range(len(tokenised)) if j != i]
            score = self._simple_bleu(hypothesis, refs, max_n=n)
            bleu_scores.append(score)

        return float(np.mean(bleu_scores))

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _type_token_ratio(text: str) -> float:
        """Type-token ratio: unique words / total words."""
        words = text.lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    @staticmethod
    def _length_ratio(generation: str, prompt: TaskPrompt) -> float:
        """Ratio of generation length to the maximum allowed length."""
        gen_len = len(generation.split())
        max_len = prompt.max_gen_length if prompt.max_gen_length > 0 else 512
        return min(1.0, gen_len / max_len)

    @staticmethod
    def _reference_overlap(generation: str, reference: str) -> float:
        """Unigram overlap (F1) between generation and reference."""
        gen_words = set(generation.lower().split())
        ref_words = set(reference.lower().split())
        if not gen_words or not ref_words:
            return 0.0
        intersection = gen_words & ref_words
        precision = len(intersection) / len(gen_words)
        recall = len(intersection) / len(ref_words)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _bleu_ngram(generation: str, reference: str, n: int = 1) -> float:
        """Simplified n-gram precision (single reference)."""
        gen_tokens = generation.lower().split()
        ref_tokens = reference.lower().split()
        if len(gen_tokens) < n or len(ref_tokens) < n:
            return 0.0

        gen_ngrams = Counter(
            tuple(gen_tokens[i : i + n]) for i in range(len(gen_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        )

        clipped = {ng: min(gen_ngrams[ng], ref_ngrams.get(ng, 0)) for ng in gen_ngrams}
        numerator = sum(clipped.values())
        denominator = sum(gen_ngrams.values())
        return numerator / denominator if denominator > 0 else 0.0

    @staticmethod
    def _simple_bleu(
        hypothesis: List[str],
        references: List[List[str]],
        max_n: int = 4,
    ) -> float:
        """Simplified multi-reference BLEU for Self-BLEU computation."""
        if not hypothesis:
            return 0.0

        precisions: List[float] = []
        for n in range(1, max_n + 1):
            if len(hypothesis) < n:
                precisions.append(0.0)
                continue

            hyp_ngrams = Counter(
                tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) - n + 1)
            )
            max_ref_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
            for ref in references:
                ref_ngrams = Counter(
                    tuple(ref[i : i + n]) for i in range(len(ref) - n + 1)
                )
                for ng, count in ref_ngrams.items():
                    max_ref_counts[ng] = max(max_ref_counts[ng], count)

            clipped = sum(
                min(hyp_ngrams[ng], max_ref_counts.get(ng, 0))
                for ng in hyp_ngrams
            )
            total = sum(hyp_ngrams.values())
            precisions.append(clipped / total if total > 0 else 0.0)

        # Geometric mean of precisions with smoothing
        if all(p == 0 for p in precisions):
            return 0.0

        log_avg = sum(
            math.log(max(p, 1e-10)) for p in precisions
        ) / len(precisions)

        # Brevity penalty (use shortest reference)
        ref_lens = [len(r) for r in references]
        closest_len = min(ref_lens, key=lambda r: abs(r - len(hypothesis)))
        if len(hypothesis) < closest_len:
            bp = math.exp(1 - closest_len / len(hypothesis))
        else:
            bp = 1.0

        return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# GenerationTask (abstract base)
# ---------------------------------------------------------------------------


class GenerationTask(ABC):
    """Abstract base class for concrete generation task domains.

    Subclasses implement prompt loading, formatting, and domain-specific
    evaluation.  The base class provides common utilities such as constraint
    validation, post-processing, and a task registry.

    Parameters
    ----------
    config : TaskConfig
        Configuration for this task run.
    """

    # Class-level registry mapping task names → classes
    _registry: ClassVar[Dict[str, Type["GenerationTask"]]] = _TASK_REGISTRY

    def __init__(self, config: Optional[TaskConfig] = None) -> None:
        self.config = config or self.get_default_config()
        errors = self.config.validate()
        if errors:
            raise ValueError(
                f"Invalid TaskConfig: {'; '.join(errors)}"
            )
        self._evaluator = TaskEvaluator(
            metrics_config={m: {} for m in self.config.evaluation_metrics}
        )
        self._dataset: Optional[PromptDataset] = None

    # -----------------------------------------------------------------
    # Abstract interface
    # -----------------------------------------------------------------

    @abstractmethod
    def load_prompts(self) -> PromptDataset:
        """Load and return the prompt dataset for this task.

        Implementations should construct :class:`TaskPrompt` objects from
        their domain-specific data sources (files, APIs, procedural
        generation, etc.).
        """
        ...

    @abstractmethod
    def format_prompt(self, prompt: TaskPrompt) -> str:
        """Format a prompt for model consumption.

        Apply the task's prompt template and any domain-specific
        transformations (e.g. adding system instructions).

        Parameters
        ----------
        prompt : TaskPrompt
            The raw prompt.

        Returns
        -------
        str
            The formatted string ready for the model.
        """
        ...

    @abstractmethod
    def evaluate(
        self, generations: List[str], prompts: List[TaskPrompt]
    ) -> Dict[str, float]:
        """Domain-specific evaluation of a batch of generations.

        Parameters
        ----------
        generations : List[str]
            Generated texts.
        prompts : List[TaskPrompt]
            Corresponding prompts.

        Returns
        -------
        Dict[str, float]
            Metric name → score.
        """
        ...

    @abstractmethod
    def get_constraints(self) -> List[TaskConstraint]:
        """Return the list of domain-specific constraints.

        These are merged with any per-prompt constraints during validation.
        """
        ...

    # -----------------------------------------------------------------
    # Default / common implementations
    # -----------------------------------------------------------------

    @classmethod
    def get_default_config(cls) -> TaskConfig:
        """Return a sensible default :class:`TaskConfig` for this task.

        Subclasses may override to provide domain-specific defaults.
        """
        return TaskConfig(
            name=cls.__name__,
            domain=TaskDomain.OPEN_ENDED_GENERATION,
            num_prompts=100,
            max_length=512,
            min_length=10,
            temperature=1.0,
        )

    def describe(self) -> str:
        """Return a human-readable description of this task."""
        doc = self.__class__.__doc__ or "No description available."
        first_line = doc.strip().split("\n")[0]
        return (
            f"Task: {self.config.name}\n"
            f"Domain: {self.config.domain.name}\n"
            f"Description: {first_line}\n"
            f"Prompts: {self.config.num_prompts}\n"
            f"Length: [{self.config.min_length}, {self.config.max_length}]\n"
            f"Temperature: {self.config.temperature}\n"
            f"Constraints: {len(self.config.constraints)}\n"
            f"Metrics: {', '.join(self.config.evaluation_metrics)}"
        )

    def validate_generation(
        self, text: str, prompt: TaskPrompt
    ) -> Tuple[bool, List[str]]:
        """Validate a generation against all applicable constraints.

        Parameters
        ----------
        text : str
            The generated text to validate.
        prompt : TaskPrompt
            The prompt that produced the generation.

        Returns
        -------
        Tuple[bool, List[str]]
            A boolean indicating overall validity and a list of human-readable
            reasons for any failures.
        """
        reasons: List[str] = []

        # Length checks
        word_count = len(text.split())
        if word_count < self.config.min_length:
            reasons.append(
                f"Too short: {word_count} words < min {self.config.min_length}"
            )
        if word_count > self.config.max_length:
            reasons.append(
                f"Too long: {word_count} words > max {self.config.max_length}"
            )

        # Empty check
        if not text.strip():
            reasons.append("Generation is empty or whitespace-only")

        # Global constraints
        for constraint in self.config.constraints:
            if not constraint.check(text):
                reasons.append(f"Failed global constraint: {constraint.describe()}")

        # Prompt-level constraints
        for constraint in prompt.constraints:
            if not constraint.check(text):
                reasons.append(f"Failed prompt constraint: {constraint.describe()}")

        # Domain-specific constraints
        for constraint in self.get_constraints():
            if not constraint.check(text):
                reasons.append(f"Failed domain constraint: {constraint.describe()}")

        is_valid = len(reasons) == 0
        return is_valid, reasons

    def post_process(self, text: str) -> str:
        """Apply post-processing to generated text.

        Default implementation strips whitespace and removes common artefacts.
        Subclasses may override for domain-specific cleaning.

        Parameters
        ----------
        text : str
            Raw generated text.

        Returns
        -------
        str
            Cleaned text.
        """
        # Strip leading/trailing whitespace
        text = text.strip()

        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)

        # Remove trailing incomplete sentence (no terminal punctuation)
        lines = text.split("\n")
        if lines and lines[-1].strip() and not re.search(r"[.!?\"')\]]$", lines[-1].strip()):
            # Only remove if there are other complete lines
            if len(lines) > 1:
                text = "\n".join(lines[:-1])

        return text.strip()

    def get_metric_names(self) -> List[str]:
        """Return the list of metric names this task evaluates.

        Combines the metrics from :pyattr:`config.evaluation_metrics` with
        any domain-specific metrics that subclasses add.
        """
        base_metrics = list(self.config.evaluation_metrics)
        # Subclasses can extend by overriding
        return base_metrics

    def summary(self) -> Dict[str, Any]:
        """Return a machine-readable summary of this task configuration.

        Returns
        -------
        dict
            Includes config, constraints, metrics, and dataset stats
            (if loaded).
        """
        result: Dict[str, Any] = {
            "task_class": self.__class__.__name__,
            "config": self.config.to_dict(),
            "constraints": [c.to_dict() for c in self.get_constraints()],
            "metrics": self.get_metric_names(),
        }
        if self._dataset is not None:
            result["dataset_stats"] = self._dataset.statistics()
        return result

    def get_dataset(self) -> PromptDataset:
        """Load (if necessary) and return the prompt dataset."""
        if self._dataset is None:
            self._dataset = self.load_prompts()
        return self._dataset

    def format_all_prompts(self, dataset: Optional[PromptDataset] = None) -> List[str]:
        """Format every prompt in *dataset* (or the loaded dataset).

        Returns
        -------
        List[str]
            Formatted prompt strings ready for the model.
        """
        ds = dataset or self.get_dataset()
        return [self.format_prompt(p) for p in ds]

    def evaluate_with_references(
        self,
        generations: List[str],
        prompts: List[TaskPrompt],
    ) -> Dict[str, float]:
        """Evaluate using the base evaluator plus domain-specific evaluation.

        Merges scores from :meth:`evaluate` with scores from the base
        :class:`TaskEvaluator` to give a comprehensive picture.
        """
        # Domain-specific scores
        domain_scores = self.evaluate(generations, prompts)

        # Base evaluator scores
        references = [
            p.reference_outputs[0] if p.reference_outputs else None
            for p in prompts
        ]
        base_scores = self._evaluator.evaluate_set(generations, prompts, references)

        # Merge (domain-specific takes precedence on key collisions)
        merged = {**base_scores, **domain_scores}
        return merged

    def validate_batch(
        self,
        generations: List[str],
        prompts: List[TaskPrompt],
    ) -> Dict[str, Any]:
        """Validate an entire batch and return aggregate validation stats.

        Returns
        -------
        dict
            Keys: ``valid_count``, ``invalid_count``, ``valid_fraction``,
            ``failure_reasons`` (Counter of reason strings).
        """
        valid_count = 0
        reason_counts: Counter = Counter()

        for gen, prompt in zip(generations, prompts):
            is_valid, reasons = self.validate_generation(gen, prompt)
            if is_valid:
                valid_count += 1
            for r in reasons:
                reason_counts[r] += 1

        total = len(generations)
        return {
            "valid_count": valid_count,
            "invalid_count": total - valid_count,
            "valid_fraction": valid_count / total if total > 0 else 0.0,
            "failure_reasons": dict(reason_counts),
        }

    # -----------------------------------------------------------------
    # Task registry
    # -----------------------------------------------------------------

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        """Class decorator / classmethod to register a task in the global registry.

        Can be used as a decorator::

            @GenerationTask.register("creative-writing")
            class CreativeWritingTask(GenerationTask):
                ...

        Or called directly::

            GenerationTask.register("my-task")(MyTask)

        Parameters
        ----------
        name : str, optional
            Registry key.  Defaults to the class name.

        Returns
        -------
        Callable
            The decorator that performs registration.
        """

        def decorator(task_cls: Type["GenerationTask"]) -> Type["GenerationTask"]:
            key = name or task_cls.__name__
            _TASK_REGISTRY[key] = task_cls
            return task_cls

        return decorator

    @classmethod
    def from_registry(cls, name: str, config: Optional[TaskConfig] = None) -> "GenerationTask":
        """Instantiate a registered task by name.

        Parameters
        ----------
        name : str
            The registry key.
        config : TaskConfig, optional
            Override config; uses default if ``None``.

        Returns
        -------
        GenerationTask

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        if name not in _TASK_REGISTRY:
            available = ", ".join(sorted(_TASK_REGISTRY.keys()))
            raise KeyError(
                f"Unknown task {name!r}. Available: {available}"
            )
        task_cls = _TASK_REGISTRY[name]
        return task_cls(config=config)

    @classmethod
    def list_registered(cls) -> List[str]:
        """Return sorted list of registered task names."""
        return sorted(_TASK_REGISTRY.keys())

    # -----------------------------------------------------------------
    # Dunder helpers
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.config.name!r}, "
            f"domain={self.config.domain.name}, "
            f"prompts={self.config.num_prompts})"
        )
