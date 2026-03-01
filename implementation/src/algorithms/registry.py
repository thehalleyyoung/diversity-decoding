"""
Algorithm registry, factory, validation, profiling, and selection utilities
for the Diversity Decoding Arena.

This module extends the base :class:`AlgorithmRegistry` with rich metadata,
a validation layer, parameter sweep helpers, profiling infrastructure, and
an intelligent algorithm selector.  It intentionally avoids importing from
``src.algorithms.__init__`` to prevent circular imports.

Typical usage::

    from src.algorithms.registry import (
        ExtendedRegistry,
        AlgorithmFactory,
        ConfigValidator,
        AlgorithmProfiler,
        AlgorithmSelector,
    )

    # Create an algorithm with defaults
    algo = AlgorithmFactory.create_with_defaults("temperature_sampling")

    # Validate a config
    issues = ConfigValidator.validate(my_config)

    # Profile an algorithm
    results = AlgorithmProfiler.profile(algo, logit_source, config)

    # Select algorithms for a task
    recs = AlgorithmSelector.select_for_task("creative_writing")
"""

from __future__ import annotations

import abc
import copy
import enum
import inspect
import logging
import math
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np

from src.algorithms.base import (
    AlgorithmRegistry,
    DecodingAlgorithm,
    DecodingConfig,
    DecodingState,
    LogitSource,
    TokenSequence,
)

logger = logging.getLogger(__name__)

# =========================================================================
# AlgorithmFamily enum
# =========================================================================


class AlgorithmFamily(enum.Enum):
    """Broad family / category of a decoding algorithm."""

    SAMPLING = "sampling"
    BEAM_SEARCH = "beam_search"
    RERANKING = "reranking"
    PARTICLE = "particle"
    QUALITY_DIVERSITY = "quality_diversity"

    @classmethod
    def from_string(cls, value: str) -> "AlgorithmFamily":
        """Case-insensitive lookup by value or name."""
        value_lower = value.lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == value_lower or member.name.lower() == value_lower:
                return member
        raise ValueError(
            f"Unknown algorithm family '{value}'. "
            f"Valid: {[m.value for m in cls]}"
        )

    def label(self) -> str:
        """Human-readable label."""
        return self.value.replace("_", " ").title()


# =========================================================================
# Complexity information
# =========================================================================


@dataclass(frozen=True)
class ComplexityInfo:
    """Asymptotic complexity description for an algorithm."""

    time: str = "O(n)"
    space: str = "O(n)"
    description: str = ""
    scales_with_vocab: bool = True
    scales_with_beam_width: bool = False
    parallelisable: bool = True

    def summary(self) -> str:
        parts = [f"Time {self.time}, Space {self.space}"]
        if self.description:
            parts.append(self.description)
        return "; ".join(parts)


# =========================================================================
# ParamSpec — description of a single hyper-parameter
# =========================================================================


@dataclass(frozen=True)
class ParamSpec:
    """Specification for one hyper-parameter of an algorithm."""

    name: str
    description: str = ""
    dtype: str = "float"  # float, int, bool, str, list
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None
    log_scale: bool = False
    required: bool = False

    # -- validation ---------------------------------------------------------

    def validate(self, value: Any) -> List[str]:
        """Return a list of issues for *value* (empty == OK)."""
        issues: List[str] = []
        if value is None:
            if self.required:
                issues.append(f"Parameter '{self.name}' is required")
            return issues

        if self.dtype == "float":
            if not isinstance(value, (int, float)):
                issues.append(
                    f"Parameter '{self.name}' expected float, got {type(value).__name__}"
                )
                return issues
            value = float(value)
            if self.min_value is not None and value < self.min_value:
                issues.append(
                    f"Parameter '{self.name}' = {value} is below minimum {self.min_value}"
                )
            if self.max_value is not None and value > self.max_value:
                issues.append(
                    f"Parameter '{self.name}' = {value} is above maximum {self.max_value}"
                )
        elif self.dtype == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                issues.append(
                    f"Parameter '{self.name}' expected int, got {type(value).__name__}"
                )
                return issues
            if self.min_value is not None and value < int(self.min_value):
                issues.append(
                    f"Parameter '{self.name}' = {value} is below minimum {int(self.min_value)}"
                )
            if self.max_value is not None and value > int(self.max_value):
                issues.append(
                    f"Parameter '{self.name}' = {value} is above maximum {int(self.max_value)}"
                )
        elif self.dtype == "bool":
            if not isinstance(value, bool):
                issues.append(
                    f"Parameter '{self.name}' expected bool, got {type(value).__name__}"
                )
        elif self.dtype == "str":
            if not isinstance(value, str):
                issues.append(
                    f"Parameter '{self.name}' expected str, got {type(value).__name__}"
                )

        if self.choices is not None and value not in self.choices:
            issues.append(
                f"Parameter '{self.name}' = {value!r} not in allowed choices {self.choices}"
            )
        return issues

    def suggested_range(self) -> Tuple[Any, Any]:
        """Return a (low, high) suggestion for sweeps."""
        low = self.min_value if self.min_value is not None else (0.0 if self.dtype == "float" else 0)
        high = self.max_value if self.max_value is not None else (10.0 if self.dtype == "float" else 100)
        return (low, high)


# =========================================================================
# AlgorithmMetadata
# =========================================================================


@dataclass
class AlgorithmMetadata:
    """Rich metadata about a registered algorithm."""

    name: str
    description: str = ""
    category: str = "sampling"  # sampling / beam / postprocessing / novel
    family: AlgorithmFamily = AlgorithmFamily.SAMPLING
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_specs: Dict[str, ParamSpec] = field(default_factory=dict)
    complexity: ComplexityInfo = field(default_factory=ComplexityInfo)
    paper_reference: str = ""
    tags: FrozenSet[str] = field(default_factory=frozenset)
    supports_batching: bool = True
    deterministic: bool = False
    typical_time_ms: float = 0.0
    typical_memory_mb: float = 0.0

    # -- helpers ------------------------------------------------------------

    def param_names(self) -> List[str]:
        return sorted(self.param_specs.keys())

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "family": self.family.value,
            "default_params": dict(self.default_params),
            "params": {k: asdict(v) for k, v in self.param_specs.items()},
            "complexity": asdict(self.complexity),
            "paper_reference": self.paper_reference,
            "tags": sorted(self.tags),
            "supports_batching": self.supports_batching,
            "deterministic": self.deterministic,
            "typical_time_ms": self.typical_time_ms,
            "typical_memory_mb": self.typical_memory_mb,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlgorithmMetadata":
        family = AlgorithmFamily.from_string(d.get("family", "sampling"))
        complexity = ComplexityInfo(**d["complexity"]) if "complexity" in d else ComplexityInfo()
        param_specs: Dict[str, ParamSpec] = {}
        for k, v in d.get("params", {}).items():
            if "choices" in v and v["choices"] is not None:
                v["choices"] = tuple(v["choices"])
            param_specs[k] = ParamSpec(**v)
        tags = frozenset(d.get("tags", []))
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            category=d.get("category", "sampling"),
            family=family,
            default_params=d.get("default_params", {}),
            param_specs=param_specs,
            complexity=complexity,
            paper_reference=d.get("paper_reference", ""),
            tags=tags,
            supports_batching=d.get("supports_batching", True),
            deterministic=d.get("deterministic", False),
            typical_time_ms=d.get("typical_time_ms", 0.0),
            typical_memory_mb=d.get("typical_memory_mb", 0.0),
        )


# =========================================================================
# Built-in metadata catalogue
# =========================================================================

_BUILTIN_METADATA: Dict[str, AlgorithmMetadata] = {}


def _register_builtin(meta: AlgorithmMetadata) -> None:
    _BUILTIN_METADATA[meta.name] = meta


# -- Temperature Sampling ---------------------------------------------------

_register_builtin(AlgorithmMetadata(
    name="temperature_sampling",
    description=(
        "Scales logits by a temperature parameter before sampling.  Higher "
        "temperatures increase randomness (diversity) while lower temperatures "
        "concentrate probability mass on the most likely tokens."
    ),
    category="sampling",
    family=AlgorithmFamily.SAMPLING,
    default_params={"temperature": 1.0},
    param_specs={
        "temperature": ParamSpec(
            name="temperature",
            description="Softmax temperature. >1 increases diversity, <1 decreases it.",
            dtype="float",
            default=1.0,
            min_value=0.01,
            max_value=10.0,
        ),
        "gumbel": ParamSpec(
            name="gumbel",
            description="Enable Gumbel-max sampling trick.",
            dtype="bool",
            default=False,
        ),
    },
    complexity=ComplexityInfo(
        time="O(V)",
        space="O(V)",
        description="Linear in vocabulary size per step.",
    ),
    paper_reference="Ackley et al., 1985 (Boltzmann machines)",
    tags=frozenset({"basic", "sampling", "stochastic"}),
    deterministic=False,
    typical_time_ms=0.5,
    typical_memory_mb=1.0,
))

# -- Top-K Sampling ----------------------------------------------------------

_register_builtin(AlgorithmMetadata(
    name="top_k_sampling",
    description=(
        "Restricts sampling to the top-K most probable tokens, zeroing out "
        "all others.  Combines truncation with temperature sampling for "
        "controllable diversity."
    ),
    category="sampling",
    family=AlgorithmFamily.SAMPLING,
    default_params={"k": 50, "temperature": 1.0},
    param_specs={
        "k": ParamSpec(
            name="k",
            description="Number of top tokens to keep.",
            dtype="int",
            default=50,
            min_value=1,
            max_value=10000,
        ),
        "temperature": ParamSpec(
            name="temperature",
            description="Temperature applied after truncation.",
            dtype="float",
            default=1.0,
            min_value=0.01,
            max_value=10.0,
        ),
    },
    complexity=ComplexityInfo(
        time="O(V log V)",
        space="O(V)",
        description="Dominated by the partial sort / top-k selection.",
    ),
    paper_reference="Fan et al., 2018 (Hierarchical Neural Story Generation)",
    tags=frozenset({"basic", "sampling", "truncation", "stochastic"}),
    deterministic=False,
    typical_time_ms=0.8,
    typical_memory_mb=1.0,
))

# -- Nucleus (Top-P) Sampling -----------------------------------------------

_register_builtin(AlgorithmMetadata(
    name="nucleus_sampling",
    description=(
        "Dynamically selects the smallest set of tokens whose cumulative "
        "probability mass exceeds p (top-p), then samples from that nucleus.  "
        "Adapts the candidate set size per step."
    ),
    category="sampling",
    family=AlgorithmFamily.SAMPLING,
    default_params={"top_p": 0.9, "temperature": 1.0, "min_tokens_to_keep": 1},
    param_specs={
        "top_p": ParamSpec(
            name="top_p",
            description="Cumulative probability threshold for the nucleus.",
            dtype="float",
            default=0.9,
            min_value=0.0,
            max_value=1.0,
        ),
        "temperature": ParamSpec(
            name="temperature",
            description="Temperature applied before nucleus selection.",
            dtype="float",
            default=1.0,
            min_value=0.01,
            max_value=10.0,
        ),
        "min_tokens_to_keep": ParamSpec(
            name="min_tokens_to_keep",
            description="Minimum tokens in the nucleus regardless of p.",
            dtype="int",
            default=1,
            min_value=1,
            max_value=1000,
        ),
    },
    complexity=ComplexityInfo(
        time="O(V log V)",
        space="O(V)",
        description="Sort required for cumulative probability computation.",
    ),
    paper_reference="Holtzman et al., 2020 (The Curious Case of Neural Text Degeneration)",
    tags=frozenset({"sampling", "adaptive", "stochastic"}),
    deterministic=False,
    typical_time_ms=1.0,
    typical_memory_mb=1.5,
))

# -- Typical Decoding --------------------------------------------------------

_register_builtin(AlgorithmMetadata(
    name="typical_decoding",
    description=(
        "Selects tokens whose information content (negative log-probability) is "
        "close to the conditional entropy, favouring 'typical' tokens over "
        "unlikely or overly likely ones."
    ),
    category="sampling",
    family=AlgorithmFamily.SAMPLING,
    default_params={"typical_p": 0.95, "temperature": 1.0},
    param_specs={
        "typical_p": ParamSpec(
            name="typical_p",
            description="Cumulative probability mass threshold for the typical set.",
            dtype="float",
            default=0.95,
            min_value=0.0,
            max_value=1.0,
        ),
        "temperature": ParamSpec(
            name="temperature",
            description="Temperature for the underlying distribution.",
            dtype="float",
            default=1.0,
            min_value=0.01,
            max_value=10.0,
        ),
    },
    complexity=ComplexityInfo(
        time="O(V log V)",
        space="O(V)",
        description="Entropy computation plus sort.",
    ),
    paper_reference="Meister et al., 2023 (Locally Typical Sampling)",
    tags=frozenset({"sampling", "information-theoretic", "stochastic"}),
    deterministic=False,
    typical_time_ms=1.2,
    typical_memory_mb=1.5,
))

# -- Diverse Beam Search -----------------------------------------------------

_register_builtin(AlgorithmMetadata(
    name="diverse_beam_search",
    description=(
        "Partitions beams into groups and applies a diversity-promoting "
        "penalty across groups, encouraging the search to explore different "
        "modes of the output space."
    ),
    category="beam",
    family=AlgorithmFamily.BEAM_SEARCH,
    default_params={
        "num_beams": 20,
        "num_beam_groups": 4,
        "diversity_penalty": 1.0,
        "length_penalty": 1.0,
    },
    param_specs={
        "num_beams": ParamSpec(
            name="num_beams",
            description="Total number of beams.",
            dtype="int",
            default=20,
            min_value=2,
            max_value=200,
        ),
        "num_beam_groups": ParamSpec(
            name="num_beam_groups",
            description="Number of diverse beam groups.",
            dtype="int",
            default=4,
            min_value=2,
            max_value=50,
        ),
        "diversity_penalty": ParamSpec(
            name="diversity_penalty",
            description="Strength of the inter-group diversity penalty.",
            dtype="float",
            default=1.0,
            min_value=0.0,
            max_value=20.0,
        ),
        "length_penalty": ParamSpec(
            name="length_penalty",
            description="Exponential length penalty (alpha).",
            dtype="float",
            default=1.0,
            min_value=0.0,
            max_value=5.0,
        ),
    },
    complexity=ComplexityInfo(
        time="O(G * B * V)",
        space="O(B * L)",
        description="G groups × B beams per group × V vocab per step.",
        scales_with_beam_width=True,
    ),
    paper_reference="Vijayakumar et al., 2018 (Diverse Beam Search)",
    tags=frozenset({"beam_search", "diversity", "deterministic"}),
    deterministic=True,
    typical_time_ms=50.0,
    typical_memory_mb=20.0,
))

# -- Contrastive Search ------------------------------------------------------

_register_builtin(AlgorithmMetadata(
    name="contrastive_search",
    description=(
        "Combines a likelihood term with a degeneration penalty that measures "
        "cosine similarity to previously generated tokens, selecting the token "
        "that balances quality and distinctiveness."
    ),
    category="novel",
    family=AlgorithmFamily.SAMPLING,
    default_params={"alpha": 0.6, "k": 5},
    param_specs={
        "alpha": ParamSpec(
            name="alpha",
            description="Trade-off between likelihood and degeneration penalty.",
            dtype="float",
            default=0.6,
            min_value=0.0,
            max_value=1.0,
        ),
        "k": ParamSpec(
            name="k",
            description="Number of candidate tokens considered at each step.",
            dtype="int",
            default=5,
            min_value=1,
            max_value=100,
        ),
    },
    complexity=ComplexityInfo(
        time="O(k * L)",
        space="O(L * d)",
        description="k candidates compared against L previous embeddings of dimension d.",
    ),
    paper_reference="Su et al., 2022 (A Contrastive Framework for Neural Text Generation)",
    tags=frozenset({"contrastive", "degeneration_penalty", "novel"}),
    deterministic=True,
    typical_time_ms=15.0,
    typical_memory_mb=10.0,
))

# -- DPP Reranking -----------------------------------------------------------

_register_builtin(AlgorithmMetadata(
    name="dpp_reranking",
    description=(
        "Generates candidate sequences with a base sampler, builds a "
        "Determinantal Point Process kernel over them, and re-ranks / "
        "subselects to maximise diversity measured via the DPP log-probability."
    ),
    category="postprocessing",
    family=AlgorithmFamily.RERANKING,
    default_params={
        "num_candidates": 100,
        "num_select": 20,
        "quality_weight": 0.5,
        "diversity_weight": 0.5,
        "kernel_type": "cosine",
    },
    param_specs={
        "num_candidates": ParamSpec(
            name="num_candidates",
            description="Number of candidate sequences to generate before reranking.",
            dtype="int",
            default=100,
            min_value=10,
            max_value=1000,
        ),
        "num_select": ParamSpec(
            name="num_select",
            description="Number of sequences to select from candidates.",
            dtype="int",
            default=20,
            min_value=1,
            max_value=500,
        ),
        "quality_weight": ParamSpec(
            name="quality_weight",
            description="Weight for quality term in the DPP kernel.",
            dtype="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
        ),
        "diversity_weight": ParamSpec(
            name="diversity_weight",
            description="Weight for diversity term in the DPP kernel.",
            dtype="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
        ),
        "kernel_type": ParamSpec(
            name="kernel_type",
            description="Kernel function for measuring similarity.",
            dtype="str",
            default="cosine",
            choices=("cosine", "rbf", "polynomial"),
        ),
    },
    complexity=ComplexityInfo(
        time="O(C^3)",
        space="O(C^2)",
        description="DPP inference cubic in candidate count C.",
        scales_with_vocab=False,
    ),
    paper_reference="Kulesza & Taskar, 2012 (Determinantal Point Processes for Machine Learning)",
    tags=frozenset({"reranking", "dpp", "diversity", "postprocessing"}),
    deterministic=False,
    typical_time_ms=200.0,
    typical_memory_mb=50.0,
))

# -- MBR Diversity -----------------------------------------------------------

_register_builtin(AlgorithmMetadata(
    name="mbr_diversity",
    description=(
        "Minimum Bayes Risk decoding with a diversity-aware utility function. "
        "Generates many samples and selects a subset that minimises expected "
        "risk while maintaining coverage of the output space."
    ),
    category="postprocessing",
    family=AlgorithmFamily.RERANKING,
    default_params={
        "num_samples": 100,
        "num_references": 50,
        "metric": "bleu",
        "diversity_bonus": 0.3,
    },
    param_specs={
        "num_samples": ParamSpec(
            name="num_samples",
            description="Hypothesis samples to draw.",
            dtype="int",
            default=100,
            min_value=10,
            max_value=1000,
        ),
        "num_references": ParamSpec(
            name="num_references",
            description="Pseudo-reference samples for risk estimation.",
            dtype="int",
            default=50,
            min_value=5,
            max_value=500,
        ),
        "metric": ParamSpec(
            name="metric",
            description="Utility / similarity metric.",
            dtype="str",
            default="bleu",
            choices=("bleu", "rouge", "bertscore", "chrf"),
        ),
        "diversity_bonus": ParamSpec(
            name="diversity_bonus",
            description="Bonus for diversity in the utility function.",
            dtype="float",
            default=0.3,
            min_value=0.0,
            max_value=2.0,
        ),
    },
    complexity=ComplexityInfo(
        time="O(S * R * M)",
        space="O(S * R)",
        description="S samples × R references × M metric cost.",
        scales_with_vocab=False,
    ),
    paper_reference="Eikema & Aziz, 2022 (Sampling-Based MBR Decoding)",
    tags=frozenset({"mbr", "reranking", "diversity", "postprocessing"}),
    deterministic=False,
    typical_time_ms=500.0,
    typical_memory_mb=30.0,
))

# -- Stein Variational Decoding ----------------------------------------------

_register_builtin(AlgorithmMetadata(
    name="stein_variational_decoding",
    description=(
        "Treats decoding as variational inference, maintaining a set of "
        "particles (partial sequences) and pushing them apart via a Stein "
        "variational gradient descent repulsive kernel."
    ),
    category="novel",
    family=AlgorithmFamily.PARTICLE,
    default_params={
        "num_particles": 20,
        "kernel_bandwidth": 1.0,
        "step_size": 0.1,
        "num_svgd_steps": 5,
    },
    param_specs={
        "num_particles": ParamSpec(
            name="num_particles",
            description="Number of particles (candidate sequences).",
            dtype="int",
            default=20,
            min_value=2,
            max_value=200,
        ),
        "kernel_bandwidth": ParamSpec(
            name="kernel_bandwidth",
            description="RBF kernel bandwidth for SVGD.",
            dtype="float",
            default=1.0,
            min_value=0.01,
            max_value=100.0,
            log_scale=True,
        ),
        "step_size": ParamSpec(
            name="step_size",
            description="SVGD update step size.",
            dtype="float",
            default=0.1,
            min_value=0.001,
            max_value=10.0,
            log_scale=True,
        ),
        "num_svgd_steps": ParamSpec(
            name="num_svgd_steps",
            description="Number of SVGD optimisation steps per decoding step.",
            dtype="int",
            default=5,
            min_value=1,
            max_value=50,
        ),
    },
    complexity=ComplexityInfo(
        time="O(P^2 * V * S)",
        space="O(P * V)",
        description="P particles, V vocab, S SVGD steps per decoding step.",
        scales_with_vocab=True,
    ),
    paper_reference="Inspired by Liu & Wang, 2016 (Stein Variational Gradient Descent)",
    tags=frozenset({"particle", "variational", "novel", "svgd"}),
    deterministic=False,
    typical_time_ms=300.0,
    typical_memory_mb=40.0,
))

# -- Quality-Diversity Beam Search -------------------------------------------

_register_builtin(AlgorithmMetadata(
    name="quality_diversity_beam_search",
    description=(
        "MAP-Elites-inspired beam search that maintains an archive of high-"
        "quality, behaviourally diverse partial sequences.  Beams fill niches "
        "in a user-defined behaviour space, promoting both quality and coverage."
    ),
    category="novel",
    family=AlgorithmFamily.QUALITY_DIVERSITY,
    default_params={
        "archive_size": 100,
        "num_beams": 20,
        "behavior_dimensions": 2,
        "niche_count": 50,
        "quality_weight": 0.7,
        "novelty_weight": 0.3,
    },
    param_specs={
        "archive_size": ParamSpec(
            name="archive_size",
            description="Maximum size of the MAP-Elites archive.",
            dtype="int",
            default=100,
            min_value=10,
            max_value=5000,
        ),
        "num_beams": ParamSpec(
            name="num_beams",
            description="Number of active beams.",
            dtype="int",
            default=20,
            min_value=2,
            max_value=200,
        ),
        "behavior_dimensions": ParamSpec(
            name="behavior_dimensions",
            description="Dimensionality of the behaviour descriptor.",
            dtype="int",
            default=2,
            min_value=1,
            max_value=10,
        ),
        "niche_count": ParamSpec(
            name="niche_count",
            description="Number of niches in the behaviour archive grid.",
            dtype="int",
            default=50,
            min_value=5,
            max_value=1000,
        ),
        "quality_weight": ParamSpec(
            name="quality_weight",
            description="Weight for quality in the combined objective.",
            dtype="float",
            default=0.7,
            min_value=0.0,
            max_value=1.0,
        ),
        "novelty_weight": ParamSpec(
            name="novelty_weight",
            description="Weight for novelty / archive coverage.",
            dtype="float",
            default=0.3,
            min_value=0.0,
            max_value=1.0,
        ),
    },
    complexity=ComplexityInfo(
        time="O(A * B * V)",
        space="O(A * L)",
        description="A archive slots × B beams × V vocab per step.",
        scales_with_beam_width=True,
    ),
    paper_reference="Inspired by Mouret & Clune, 2015 (Illuminating search spaces)",
    tags=frozenset({"quality_diversity", "map_elites", "beam_search", "novel"}),
    deterministic=False,
    typical_time_ms=400.0,
    typical_memory_mb=60.0,
))

# -- MCTS Decoding -----------------------------------------------------------

_register_builtin(AlgorithmMetadata(
    name="mcts_decoding",
    description=(
        "Monte Carlo Tree Search applied to text decoding.  Explores the "
        "token tree via UCB-guided selection, expansion, random rollouts, "
        "and back-propagation of rewards."
    ),
    category="novel",
    family=AlgorithmFamily.SAMPLING,
    default_params={
        "num_simulations": 100,
        "exploration_constant": 1.414,
        "max_rollout_length": 50,
    },
    param_specs={
        "num_simulations": ParamSpec(
            name="num_simulations",
            description="MCTS simulations per decoding step.",
            dtype="int",
            default=100,
            min_value=10,
            max_value=5000,
        ),
        "exploration_constant": ParamSpec(
            name="exploration_constant",
            description="UCB exploration constant (c_puct).",
            dtype="float",
            default=1.414,
            min_value=0.0,
            max_value=10.0,
        ),
        "max_rollout_length": ParamSpec(
            name="max_rollout_length",
            description="Maximum length of random rollouts.",
            dtype="int",
            default=50,
            min_value=5,
            max_value=500,
        ),
    },
    complexity=ComplexityInfo(
        time="O(N * L * V)",
        space="O(N * V)",
        description="N simulations × L rollout length × V vocab.",
    ),
    paper_reference="Inspired by Coulom, 2006 & Silver et al., 2016",
    tags=frozenset({"mcts", "tree_search", "novel", "stochastic"}),
    deterministic=False,
    typical_time_ms=800.0,
    typical_memory_mb=100.0,
))

# -- Stochastic Beam Search --------------------------------------------------

_register_builtin(AlgorithmMetadata(
    name="stochastic_beam_search",
    description=(
        "Beam search augmented with Gumbel top-k sampling to produce "
        "stochastic yet high-quality beams.  Provides unbiased sampling "
        "without replacement from the sequence-level distribution."
    ),
    category="beam",
    family=AlgorithmFamily.BEAM_SEARCH,
    default_params={
        "num_beams": 20,
        "temperature": 1.0,
        "length_penalty": 1.0,
    },
    param_specs={
        "num_beams": ParamSpec(
            name="num_beams",
            description="Number of beams.",
            dtype="int",
            default=20,
            min_value=2,
            max_value=200,
        ),
        "temperature": ParamSpec(
            name="temperature",
            description="Gumbel noise temperature.",
            dtype="float",
            default=1.0,
            min_value=0.01,
            max_value=10.0,
        ),
        "length_penalty": ParamSpec(
            name="length_penalty",
            description="Exponential length penalty.",
            dtype="float",
            default=1.0,
            min_value=0.0,
            max_value=5.0,
        ),
    },
    complexity=ComplexityInfo(
        time="O(B * V)",
        space="O(B * L)",
        description="B beams × V vocab per step.",
        scales_with_beam_width=True,
    ),
    paper_reference="Kool et al., 2019 (Stochastic Beams and Where to Find Them)",
    tags=frozenset({"beam_search", "stochastic", "gumbel"}),
    deterministic=False,
    typical_time_ms=40.0,
    typical_memory_mb=15.0,
))

# -- Ancestral Diverse Sampling ----------------------------------------------

_register_builtin(AlgorithmMetadata(
    name="ancestral_diverse_sampling",
    description=(
        "Ancestral sampling with explicit diversity mechanisms such as "
        "n-gram blocking, embedding-based repulsion, and adaptive temperature "
        "schedules to encourage varied generations."
    ),
    category="sampling",
    family=AlgorithmFamily.SAMPLING,
    default_params={
        "temperature": 1.0,
        "block_ngram": 4,
        "repulsion_strength": 0.5,
    },
    param_specs={
        "temperature": ParamSpec(
            name="temperature",
            description="Base sampling temperature.",
            dtype="float",
            default=1.0,
            min_value=0.01,
            max_value=10.0,
        ),
        "block_ngram": ParamSpec(
            name="block_ngram",
            description="N-gram size for repeat blocking between sequences.",
            dtype="int",
            default=4,
            min_value=0,
            max_value=20,
        ),
        "repulsion_strength": ParamSpec(
            name="repulsion_strength",
            description="Strength of embedding-based inter-sequence repulsion.",
            dtype="float",
            default=0.5,
            min_value=0.0,
            max_value=5.0,
        ),
    },
    complexity=ComplexityInfo(
        time="O(S * L * V)",
        space="O(S * L)",
        description="S sequences × L length × V vocab per step.",
    ),
    paper_reference="Custom implementation combining ancestral sampling with diversity heuristics.",
    tags=frozenset({"sampling", "ancestral", "diversity", "stochastic"}),
    deterministic=False,
    typical_time_ms=5.0,
    typical_memory_mb=5.0,
))


# =========================================================================
# Name aliases — map user-friendly names to canonical names
# =========================================================================

_NAME_ALIASES: Dict[str, str] = {
    # Temperature
    "temperature": "temperature_sampling",
    "temp": "temperature_sampling",
    "temperature_sample": "temperature_sampling",
    # Top-K
    "top_k": "top_k_sampling",
    "topk": "top_k_sampling",
    "top-k": "top_k_sampling",
    # Nucleus
    "nucleus": "nucleus_sampling",
    "top_p": "nucleus_sampling",
    "topp": "nucleus_sampling",
    "top-p": "nucleus_sampling",
    # Typical
    "typical": "typical_decoding",
    "typical_sampling": "typical_decoding",
    "locally_typical": "typical_decoding",
    # Diverse Beam Search
    "dbs": "diverse_beam_search",
    "diverse_beam": "diverse_beam_search",
    # Contrastive
    "contrastive": "contrastive_search",
    # DPP
    "dpp": "dpp_reranking",
    "dpp_rerank": "dpp_reranking",
    # MBR
    "mbr": "mbr_diversity",
    "mbr_decoding": "mbr_diversity",
    # SVD
    "svd": "stein_variational_decoding",
    "stein": "stein_variational_decoding",
    "svgd": "stein_variational_decoding",
    # QDBS
    "qdbs": "quality_diversity_beam_search",
    "qd_beam": "quality_diversity_beam_search",
    "map_elites_beam": "quality_diversity_beam_search",
    # MCTS
    "mcts": "mcts_decoding",
    "monte_carlo": "mcts_decoding",
    # Stochastic Beam
    "stochastic_beam": "stochastic_beam_search",
    "sbs": "stochastic_beam_search",
    # Ancestral
    "ancestral": "ancestral_diverse_sampling",
    "ancestral_sampling": "ancestral_diverse_sampling",
}


def resolve_name(name: str) -> str:
    """Resolve an algorithm name through aliases and normalisation."""
    normalised = name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalised in _NAME_ALIASES:
        return _NAME_ALIASES[normalised]
    if normalised in _BUILTIN_METADATA:
        return normalised
    return normalised


# =========================================================================
# ExtendedRegistry — wraps AlgorithmRegistry with metadata
# =========================================================================


class ExtendedRegistry:
    """Singleton-style registry that wraps :class:`AlgorithmRegistry` and
    adds rich metadata, alias resolution, and factory conveniences.

    This class **does not** replace ``AlgorithmRegistry``; it layers on
    top of it.  Algorithm classes are still registered with the base
    registry; this class manages the accompanying metadata.
    """

    _metadata: Dict[str, AlgorithmMetadata] = {}
    _initialised: bool = False

    # -- initialisation -----------------------------------------------------

    @classmethod
    def _ensure_init(cls) -> None:
        """Lazily populate metadata from the built-in catalogue."""
        if cls._initialised:
            return
        for name, meta in _BUILTIN_METADATA.items():
            if name not in cls._metadata:
                cls._metadata[name] = meta
        cls._initialised = True

    # -- registration -------------------------------------------------------

    @classmethod
    def register(
        cls,
        name: str,
        algorithm_cls: Type[DecodingAlgorithm],
        metadata: Optional[AlgorithmMetadata] = None,
    ) -> None:
        """Register an algorithm class **and** its metadata.

        Parameters
        ----------
        name:
            Canonical name.
        algorithm_cls:
            The class implementing the algorithm.
        metadata:
            Optional rich metadata.  If not provided, a minimal stub is
            created from the class docstring.
        """
        cls._ensure_init()
        AlgorithmRegistry.register(name, algorithm_cls)
        if metadata is not None:
            cls._metadata[name] = metadata
        elif name not in cls._metadata:
            cls._metadata[name] = AlgorithmMetadata(
                name=name,
                description=(algorithm_cls.__doc__ or "").strip().split("\n")[0],
            )
        logger.debug("ExtendedRegistry: registered '%s'", name)

    @classmethod
    def register_decorator(
        cls,
        name: str,
        metadata: Optional[AlgorithmMetadata] = None,
    ) -> Callable[[Type[DecodingAlgorithm]], Type[DecodingAlgorithm]]:
        """Class decorator for registration with metadata."""
        def decorator(algorithm_cls: Type[DecodingAlgorithm]) -> Type[DecodingAlgorithm]:
            cls.register(name, algorithm_cls, metadata)
            return algorithm_cls
        return decorator

    # -- queries ------------------------------------------------------------

    @classmethod
    def get(cls, name: str) -> Type[DecodingAlgorithm]:
        """Retrieve the algorithm class, resolving aliases."""
        cls._ensure_init()
        resolved = resolve_name(name)
        return AlgorithmRegistry.get(resolved)

    @classmethod
    def list_algorithms(cls) -> List[str]:
        """Return sorted list of all known algorithm names."""
        cls._ensure_init()
        base_names = set(AlgorithmRegistry.list_algorithms())
        meta_names = set(cls._metadata.keys())
        return sorted(base_names | meta_names)

    @classmethod
    def list_by_family(cls, family: AlgorithmFamily) -> List[str]:
        """Return algorithm names belonging to *family*."""
        cls._ensure_init()
        return sorted(
            name for name, meta in cls._metadata.items()
            if meta.family == family
        )

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """Return algorithm names matching *category*."""
        cls._ensure_init()
        cat = category.lower().strip()
        return sorted(
            name for name, meta in cls._metadata.items()
            if meta.category == cat
        )

    @classmethod
    def list_by_tag(cls, tag: str) -> List[str]:
        """Return algorithm names having *tag*."""
        cls._ensure_init()
        tag = tag.lower().strip()
        return sorted(
            name for name, meta in cls._metadata.items()
            if tag in meta.tags
        )

    @classmethod
    def get_metadata(cls, name: str) -> AlgorithmMetadata:
        """Retrieve metadata for *name* (alias-resolved).

        Raises ``KeyError`` if the algorithm is not known.
        """
        cls._ensure_init()
        resolved = resolve_name(name)
        if resolved in cls._metadata:
            return cls._metadata[resolved]
        if AlgorithmRegistry.is_registered(resolved):
            return AlgorithmMetadata(name=resolved, description="No metadata available.")
        raise KeyError(f"No metadata for '{name}' (resolved to '{resolved}')")

    @classmethod
    def search(cls, query: str) -> List[AlgorithmMetadata]:
        """Fuzzy search across names, descriptions, and tags."""
        cls._ensure_init()
        query_lower = query.lower()
        results: List[Tuple[int, AlgorithmMetadata]] = []
        for meta in cls._metadata.values():
            score = 0
            if query_lower in meta.name:
                score += 10
            if query_lower in meta.description.lower():
                score += 5
            if query_lower in meta.category.lower():
                score += 3
            if any(query_lower in t for t in meta.tags):
                score += 7
            if meta.family.value.startswith(query_lower):
                score += 6
            if score > 0:
                results.append((score, meta))
        results.sort(key=lambda x: (-x[0], x[1].name))
        return [meta for _, meta in results]

    # -- factory shortcuts --------------------------------------------------

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> DecodingAlgorithm:
        """Create an algorithm instance by name with optional overrides.

        Parameters
        ----------
        name:
            Algorithm name (aliases accepted).
        **kwargs:
            Overrides for ``DecodingConfig`` fields or ``params`` dict entries.
        """
        cls._ensure_init()
        resolved = resolve_name(name)
        meta = cls._metadata.get(resolved)

        config_fields = {f.name for f in DecodingConfig.__dataclass_fields__.values()}
        config_kwargs: Dict[str, Any] = {"algorithm_name": resolved}
        params: Dict[str, Any] = {}

        if meta:
            params.update(meta.default_params)

        for k, v in kwargs.items():
            if k in config_fields:
                config_kwargs[k] = v
            else:
                params[k] = v

        config_kwargs["params"] = params
        config = DecodingConfig(**config_kwargs)
        return AlgorithmRegistry.create(resolved, config)

    @classmethod
    def create_from_config(cls, config: DecodingConfig) -> DecodingAlgorithm:
        """Create an algorithm instance from a ``DecodingConfig``.

        The ``algorithm_name`` field of the config is used to look up the
        class.  Aliases are resolved.
        """
        cls._ensure_init()
        resolved = resolve_name(config.algorithm_name)
        config = copy.deepcopy(config)
        config.algorithm_name = resolved

        meta = cls._metadata.get(resolved)
        if meta:
            merged_params = dict(meta.default_params)
            merged_params.update(config.params)
            config.params = merged_params

        return AlgorithmRegistry.create(resolved, config)

    # -- utilities ----------------------------------------------------------

    @classmethod
    def summary_table(cls) -> str:
        """Return a formatted text table summarising all algorithms."""
        cls._ensure_init()
        lines: List[str] = []
        header = f"{'Name':<35} {'Family':<20} {'Category':<15} {'Deterministic':<14}"
        lines.append(header)
        lines.append("-" * len(header))
        for name in sorted(cls._metadata.keys()):
            meta = cls._metadata[name]
            lines.append(
                f"{name:<35} {meta.family.value:<20} {meta.category:<15} "
                f"{'Yes' if meta.deterministic else 'No':<14}"
            )
        return "\n".join(lines)

    @classmethod
    def aliases_for(cls, canonical_name: str) -> List[str]:
        """Return all aliases that map to *canonical_name*."""
        resolved = resolve_name(canonical_name)
        return sorted(
            alias for alias, target in _NAME_ALIASES.items()
            if target == resolved
        )

    @classmethod
    def clear(cls) -> None:
        """Clear extended metadata (does **not** clear base registry)."""
        cls._metadata.clear()
        cls._initialised = False

    @classmethod
    def reset(cls) -> None:
        """Clear and re-initialise from built-in catalogue."""
        cls.clear()
        cls._ensure_init()


# =========================================================================
# AlgorithmFactory
# =========================================================================


class AlgorithmFactory:
    """High-level factory for creating algorithm instances with validation,
    parameter sweeps, and ensemble construction.
    """

    # -- single-instance creation -------------------------------------------

    @staticmethod
    def create_with_defaults(name: str) -> DecodingAlgorithm:
        """Create an algorithm instance with all default parameters.

        Parameters
        ----------
        name:
            Algorithm name (aliases accepted).

        Returns
        -------
        DecodingAlgorithm
            A ready-to-use algorithm instance.
        """
        resolved = resolve_name(name)
        meta = ExtendedRegistry.get_metadata(resolved)
        config = DecodingConfig(
            algorithm_name=resolved,
            params=dict(meta.default_params),
        )
        return AlgorithmRegistry.create(resolved, config)

    @staticmethod
    def create_configured(
        name: str,
        num_sequences: int = 20,
        max_new_tokens: int = 100,
        seed: Optional[int] = None,
        **params: Any,
    ) -> DecodingAlgorithm:
        """Create with explicit generation parameters and algo-specific params."""
        resolved = resolve_name(name)
        meta = ExtendedRegistry.get_metadata(resolved)
        merged_params = dict(meta.default_params)
        merged_params.update(params)
        config = DecodingConfig(
            algorithm_name=resolved,
            num_sequences=num_sequences,
            max_new_tokens=max_new_tokens,
            seed=seed,
            params=merged_params,
        )
        return AlgorithmRegistry.create(resolved, config)

    # -- parameter sweeps ---------------------------------------------------

    @staticmethod
    def create_sweep(
        name: str,
        param_name: str,
        values: Sequence[Any],
        base_params: Optional[Dict[str, Any]] = None,
    ) -> List[DecodingAlgorithm]:
        """Create a list of algorithm instances sweeping one parameter.

        Parameters
        ----------
        name:
            Algorithm name (aliases accepted).
        param_name:
            Name of the parameter to sweep.
        values:
            Sequence of values for the swept parameter.
        base_params:
            Optional base parameters for all instances.

        Returns
        -------
        list of DecodingAlgorithm
            One instance per value.
        """
        resolved = resolve_name(name)
        meta = ExtendedRegistry.get_metadata(resolved)
        instances: List[DecodingAlgorithm] = []

        for val in values:
            params = dict(meta.default_params)
            if base_params:
                params.update(base_params)
            params[param_name] = val
            config = DecodingConfig(
                algorithm_name=resolved,
                params=params,
            )
            instances.append(AlgorithmRegistry.create(resolved, config))

        return instances

    @staticmethod
    def create_grid_sweep(
        name: str,
        param_grid: Dict[str, Sequence[Any]],
    ) -> List[DecodingAlgorithm]:
        """Create instances for all combinations in a parameter grid.

        Parameters
        ----------
        name:
            Algorithm name.
        param_grid:
            Mapping of parameter names to sequences of values.

        Returns
        -------
        list of DecodingAlgorithm
            One instance per combination (Cartesian product).
        """
        import itertools

        resolved = resolve_name(name)
        meta = ExtendedRegistry.get_metadata(resolved)
        param_names = sorted(param_grid.keys())
        value_lists = [param_grid[p] for p in param_names]
        instances: List[DecodingAlgorithm] = []

        for combo in itertools.product(*value_lists):
            params = dict(meta.default_params)
            for pname, pval in zip(param_names, combo):
                params[pname] = pval
            config = DecodingConfig(
                algorithm_name=resolved,
                params=params,
            )
            instances.append(AlgorithmRegistry.create(resolved, config))

        return instances

    @staticmethod
    def create_random_sweep(
        name: str,
        n_samples: int = 10,
        seed: Optional[int] = None,
    ) -> List[DecodingAlgorithm]:
        """Create instances with randomly sampled parameters.

        Uses the ``ParamSpec`` ranges from metadata to sample each parameter
        uniformly (or log-uniformly if ``log_scale`` is set).

        Parameters
        ----------
        name:
            Algorithm name.
        n_samples:
            Number of random configurations to generate.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        list of DecodingAlgorithm
        """
        resolved = resolve_name(name)
        meta = ExtendedRegistry.get_metadata(resolved)
        rng = np.random.default_rng(seed)
        instances: List[DecodingAlgorithm] = []

        for _ in range(n_samples):
            params = dict(meta.default_params)
            for pname, spec in meta.param_specs.items():
                if spec.choices is not None:
                    params[pname] = rng.choice(list(spec.choices))
                    continue
                if spec.dtype == "bool":
                    params[pname] = bool(rng.integers(0, 2))
                    continue
                low, high = spec.suggested_range()
                if spec.dtype == "float":
                    if spec.log_scale and float(low) > 0:
                        log_val = rng.uniform(math.log(float(low)), math.log(float(high)))
                        params[pname] = math.exp(log_val)
                    else:
                        params[pname] = float(rng.uniform(float(low), float(high)))
                elif spec.dtype == "int":
                    params[pname] = int(rng.integers(int(low), int(high) + 1))
            config = DecodingConfig(algorithm_name=resolved, params=params)
            instances.append(AlgorithmRegistry.create(resolved, config))

        return instances

    # -- ensemble -----------------------------------------------------------

    @staticmethod
    def create_ensemble(
        names: Sequence[str],
        weights: Optional[Sequence[float]] = None,
        params_per_algo: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> "EnsembleConfig":
        """Create an ensemble configuration from multiple algorithms.

        Parameters
        ----------
        names:
            Algorithm names.
        weights:
            Mixing weights (normalised internally).  Defaults to uniform.
        params_per_algo:
            Optional per-algorithm parameter overrides.

        Returns
        -------
        EnsembleConfig
            A dataclass describing the ensemble.
        """
        resolved_names = [resolve_name(n) for n in names]

        if weights is None:
            weight_list = [1.0 / len(resolved_names)] * len(resolved_names)
        else:
            total = sum(weights)
            if total <= 0:
                raise ValueError("Ensemble weights must sum to a positive value")
            weight_list = [w / total for w in weights]

        if len(weight_list) != len(resolved_names):
            raise ValueError(
                f"Number of weights ({len(weight_list)}) must match "
                f"number of algorithms ({len(resolved_names)})"
            )

        members: List[EnsembleMember] = []
        for name, weight in zip(resolved_names, weight_list):
            p = {}
            if params_per_algo and name in params_per_algo:
                p = dict(params_per_algo[name])
            meta = ExtendedRegistry.get_metadata(name)
            merged = dict(meta.default_params)
            merged.update(p)
            members.append(EnsembleMember(name=name, weight=weight, params=merged))

        return EnsembleConfig(members=members)

    @staticmethod
    def instantiate_ensemble(
        ensemble: "EnsembleConfig",
        num_sequences: int = 20,
        max_new_tokens: int = 100,
        seed: Optional[int] = None,
    ) -> List[Tuple[DecodingAlgorithm, float]]:
        """Instantiate all algorithms described by an EnsembleConfig.

        Returns a list of ``(algorithm, weight)`` tuples.  The caller is
        responsible for combining their outputs.
        """
        result: List[Tuple[DecodingAlgorithm, float]] = []
        for member in ensemble.members:
            seqs_for_member = max(1, int(round(num_sequences * member.weight)))
            config = DecodingConfig(
                algorithm_name=member.name,
                num_sequences=seqs_for_member,
                max_new_tokens=max_new_tokens,
                seed=seed,
                params=dict(member.params),
            )
            algo = AlgorithmRegistry.create(member.name, config)
            result.append((algo, member.weight))
        return result


# =========================================================================
# Ensemble helper dataclasses
# =========================================================================


@dataclass
class EnsembleMember:
    """One member of an algorithm ensemble."""
    name: str
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleConfig:
    """Configuration for an ensemble of algorithms."""
    members: List[EnsembleMember] = field(default_factory=list)

    @property
    def num_members(self) -> int:
        return len(self.members)

    @property
    def algorithm_names(self) -> List[str]:
        return [m.name for m in self.members]

    @property
    def weights(self) -> List[float]:
        return [m.weight for m in self.members]

    def normalise_weights(self) -> None:
        total = sum(m.weight for m in self.members)
        if total > 0:
            for m in self.members:
                m.weight /= total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "members": [
                {"name": m.name, "weight": m.weight, "params": m.params}
                for m in self.members
            ]
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnsembleConfig":
        members = [
            EnsembleMember(
                name=m["name"],
                weight=m.get("weight", 1.0),
                params=m.get("params", {}),
            )
            for m in d.get("members", [])
        ]
        return cls(members=members)


# =========================================================================
# ConfigValidator
# =========================================================================


class ConfigValidator:
    """Validates ``DecodingConfig`` instances and algorithm-specific parameters
    against the metadata-defined parameter specifications.
    """

    @staticmethod
    def validate(config: DecodingConfig) -> List[str]:
        """Validate a full ``DecodingConfig``.

        Checks both the base config fields (via ``config.validate()``) and
        the algorithm-specific ``params`` against their ``ParamSpec``.

        Returns
        -------
        list of str
            Validation issues.  Empty list means valid.
        """
        issues: List[str] = list(config.validate())

        resolved = resolve_name(config.algorithm_name)
        try:
            meta = ExtendedRegistry.get_metadata(resolved)
        except KeyError:
            issues.append(f"Unknown algorithm: '{config.algorithm_name}'")
            return issues

        issues.extend(ConfigValidator.validate_params(resolved, config.params))
        return issues

    @staticmethod
    def validate_params(algorithm_name: str, params: Dict[str, Any]) -> List[str]:
        """Validate algorithm-specific parameters against their specs.

        Parameters
        ----------
        algorithm_name:
            Canonical or aliased algorithm name.
        params:
            The parameters dict to validate.

        Returns
        -------
        list of str
            Validation issues.
        """
        issues: List[str] = []
        resolved = resolve_name(algorithm_name)
        try:
            meta = ExtendedRegistry.get_metadata(resolved)
        except KeyError:
            issues.append(f"Cannot validate params: unknown algorithm '{algorithm_name}'")
            return issues

        # Check required params
        for pname, spec in meta.param_specs.items():
            if spec.required and pname not in params:
                issues.append(f"Missing required parameter '{pname}' for {resolved}")

        # Validate provided params
        for pname, pval in params.items():
            if pname in meta.param_specs:
                spec = meta.param_specs[pname]
                issues.extend(spec.validate(pval))
            else:
                issues.append(
                    f"Unknown parameter '{pname}' for algorithm '{resolved}'. "
                    f"Known: {meta.param_names()}"
                )

        return issues

    @staticmethod
    def validate_cross_params(config: DecodingConfig) -> List[str]:
        """Validate cross-parameter constraints.

        For example, ``num_beam_groups`` must divide ``num_beams`` evenly
        for diverse beam search.

        Returns
        -------
        list of str
            Constraint violation messages.
        """
        issues: List[str] = []
        resolved = resolve_name(config.algorithm_name)
        params = config.params

        if resolved == "diverse_beam_search":
            nb = params.get("num_beams", 20)
            ng = params.get("num_beam_groups", 4)
            if nb % ng != 0:
                issues.append(
                    f"num_beams ({nb}) must be divisible by "
                    f"num_beam_groups ({ng}) for diverse_beam_search"
                )
            if ng > nb:
                issues.append(
                    f"num_beam_groups ({ng}) must be <= num_beams ({nb})"
                )

        if resolved == "dpp_reranking":
            nc = params.get("num_candidates", 100)
            ns = params.get("num_select", 20)
            if ns > nc:
                issues.append(
                    f"num_select ({ns}) cannot exceed num_candidates ({nc}) for dpp_reranking"
                )
            qw = params.get("quality_weight", 0.5)
            dw = params.get("diversity_weight", 0.5)
            if abs((qw + dw) - 1.0) > 0.01:
                issues.append(
                    f"quality_weight ({qw}) + diversity_weight ({dw}) should sum to 1.0"
                )

        if resolved == "quality_diversity_beam_search":
            qw = params.get("quality_weight", 0.7)
            nw = params.get("novelty_weight", 0.3)
            if abs((qw + nw) - 1.0) > 0.01:
                issues.append(
                    f"quality_weight ({qw}) + novelty_weight ({nw}) should sum to 1.0"
                )
            nb = params.get("num_beams", 20)
            asize = params.get("archive_size", 100)
            if nb > asize:
                issues.append(
                    f"num_beams ({nb}) should not exceed archive_size ({asize})"
                )

        if resolved == "mbr_diversity":
            ns = params.get("num_samples", 100)
            nr = params.get("num_references", 50)
            if nr > ns:
                issues.append(
                    f"num_references ({nr}) should not exceed num_samples ({ns}) for mbr_diversity"
                )

        return issues

    @staticmethod
    def suggest_params(algorithm_name: str) -> Dict[str, Dict[str, Any]]:
        """Suggest parameter ranges for an algorithm.

        Parameters
        ----------
        algorithm_name:
            Canonical or aliased algorithm name.

        Returns
        -------
        dict
            Mapping param name → ``{"default": ..., "range": (lo, hi), "description": ...}``.
        """
        resolved = resolve_name(algorithm_name)
        meta = ExtendedRegistry.get_metadata(resolved)
        suggestions: Dict[str, Dict[str, Any]] = {}
        for pname, spec in meta.param_specs.items():
            lo, hi = spec.suggested_range()
            suggestions[pname] = {
                "default": spec.default,
                "range": (lo, hi),
                "dtype": spec.dtype,
                "description": spec.description,
                "log_scale": spec.log_scale,
            }
            if spec.choices is not None:
                suggestions[pname]["choices"] = list(spec.choices)
        return suggestions

    @staticmethod
    def auto_fix(config: DecodingConfig) -> Tuple[DecodingConfig, List[str]]:
        """Attempt to automatically fix validation issues.

        Returns a (fixed_config, fixes_applied) tuple.
        """
        config = copy.deepcopy(config)
        fixes: List[str] = []
        resolved = resolve_name(config.algorithm_name)

        try:
            meta = ExtendedRegistry.get_metadata(resolved)
        except KeyError:
            return config, fixes

        # Clamp numeric params to valid ranges
        for pname, spec in meta.param_specs.items():
            if pname not in config.params:
                continue
            val = config.params[pname]
            if spec.dtype in ("float", "int") and isinstance(val, (int, float)):
                clamped = val
                if spec.min_value is not None and val < spec.min_value:
                    clamped = spec.min_value if spec.dtype == "float" else int(spec.min_value)
                    fixes.append(f"Clamped {pname} from {val} to {clamped} (min)")
                if spec.max_value is not None and val > spec.max_value:
                    clamped = spec.max_value if spec.dtype == "float" else int(spec.max_value)
                    fixes.append(f"Clamped {pname} from {val} to {clamped} (max)")
                config.params[pname] = clamped

        # Fill missing required params with defaults
        for pname, spec in meta.param_specs.items():
            if spec.required and pname not in config.params:
                if spec.default is not None:
                    config.params[pname] = spec.default
                    fixes.append(f"Added missing required param {pname} = {spec.default}")

        # Fix cross-param issues
        if resolved == "diverse_beam_search":
            nb = config.params.get("num_beams", 20)
            ng = config.params.get("num_beam_groups", 4)
            if nb % ng != 0:
                new_ng = max(1, nb // (nb // ng))
                while nb % new_ng != 0 and new_ng > 1:
                    new_ng -= 1
                config.params["num_beam_groups"] = new_ng
                fixes.append(
                    f"Adjusted num_beam_groups from {ng} to {new_ng} "
                    f"to evenly divide num_beams ({nb})"
                )

        return config, fixes


# =========================================================================
# ProfileResult
# =========================================================================


@dataclass
class ProfileResult:
    """Results of profiling a single algorithm run."""

    algorithm_name: str
    wall_time_seconds: float = 0.0
    step_times_ms: List[float] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    num_sequences_generated: int = 0
    total_tokens_generated: int = 0
    tokens_per_second: float = 0.0
    avg_step_time_ms: float = 0.0
    median_step_time_ms: float = 0.0
    p95_step_time_ms: float = 0.0
    p99_step_time_ms: float = 0.0
    std_step_time_ms: float = 0.0
    config_used: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def compute_statistics(self) -> None:
        """Recompute derived statistics from ``step_times_ms``."""
        if not self.step_times_ms:
            return
        self.avg_step_time_ms = statistics.mean(self.step_times_ms)
        self.median_step_time_ms = statistics.median(self.step_times_ms)
        self.std_step_time_ms = (
            statistics.stdev(self.step_times_ms) if len(self.step_times_ms) > 1 else 0.0
        )
        sorted_times = sorted(self.step_times_ms)
        idx_95 = min(int(len(sorted_times) * 0.95), len(sorted_times) - 1)
        idx_99 = min(int(len(sorted_times) * 0.99), len(sorted_times) - 1)
        self.p95_step_time_ms = sorted_times[idx_95]
        self.p99_step_time_ms = sorted_times[idx_99]
        if self.wall_time_seconds > 0 and self.total_tokens_generated > 0:
            self.tokens_per_second = self.total_tokens_generated / self.wall_time_seconds

    def summary(self) -> str:
        lines = [
            f"Profile: {self.algorithm_name}",
            f"  Wall time:       {self.wall_time_seconds:.3f}s",
            f"  Sequences:       {self.num_sequences_generated}",
            f"  Total tokens:    {self.total_tokens_generated}",
            f"  Tokens/sec:      {self.tokens_per_second:.1f}",
            f"  Avg step:        {self.avg_step_time_ms:.2f}ms",
            f"  Median step:     {self.median_step_time_ms:.2f}ms",
            f"  P95 step:        {self.p95_step_time_ms:.2f}ms",
            f"  P99 step:        {self.p99_step_time_ms:.2f}ms",
            f"  Peak memory:     {self.peak_memory_mb:.1f}MB",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_name": self.algorithm_name,
            "wall_time_seconds": self.wall_time_seconds,
            "step_times_ms": self.step_times_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "num_sequences_generated": self.num_sequences_generated,
            "total_tokens_generated": self.total_tokens_generated,
            "tokens_per_second": self.tokens_per_second,
            "avg_step_time_ms": self.avg_step_time_ms,
            "median_step_time_ms": self.median_step_time_ms,
            "p95_step_time_ms": self.p95_step_time_ms,
            "p99_step_time_ms": self.p99_step_time_ms,
            "std_step_time_ms": self.std_step_time_ms,
            "config_used": self.config_used,
            "extra": self.extra,
        }


# =========================================================================
# AlgorithmProfiler
# =========================================================================


class AlgorithmProfiler:
    """Profile algorithm performance by timing generation runs and
    collecting resource-usage statistics.
    """

    @staticmethod
    def profile(
        algorithm: DecodingAlgorithm,
        logit_source: LogitSource,
        prompt_ids: List[int],
        num_warmup: int = 1,
        num_runs: int = 3,
    ) -> ProfileResult:
        """Profile a single algorithm.

        Parameters
        ----------
        algorithm:
            The algorithm to profile.
        logit_source:
            Callable producing logits.
        prompt_ids:
            Prompt token IDs.
        num_warmup:
            Number of warmup runs (not timed).
        num_runs:
            Number of timed runs (results averaged).

        Returns
        -------
        ProfileResult
        """
        # Warmup
        for _ in range(num_warmup):
            try:
                algorithm.generate(logit_source, prompt_ids)
            except Exception:
                logger.debug("Warmup run failed for %s", algorithm.name)

        all_step_times: List[float] = []
        wall_times: List[float] = []
        total_tokens_list: List[int] = []
        num_seqs_list: List[int] = []

        for run_idx in range(num_runs):
            t0 = time.perf_counter()
            try:
                sequences = algorithm.generate(logit_source, prompt_ids)
            except Exception as exc:
                logger.warning("Profile run %d failed for %s: %s", run_idx, algorithm.name, exc)
                continue
            t1 = time.perf_counter()

            wall_times.append(t1 - t0)
            num_seqs_list.append(len(sequences))
            total_toks = sum(len(s) - len(prompt_ids) for s in sequences)
            total_tokens_list.append(max(0, total_toks))

            # Estimate per-step time: total wall / (num_seqs * avg_new_tokens)
            if total_toks > 0:
                step_time_ms = (t1 - t0) * 1000.0 / total_toks
                all_step_times.extend([step_time_ms] * total_toks)

        if not wall_times:
            return ProfileResult(
                algorithm_name=algorithm.name,
                config_used=algorithm.config.to_dict(),
            )

        avg_wall = statistics.mean(wall_times)
        avg_tokens = statistics.mean(total_tokens_list) if total_tokens_list else 0
        avg_seqs = int(statistics.mean(num_seqs_list)) if num_seqs_list else 0

        result = ProfileResult(
            algorithm_name=algorithm.name,
            wall_time_seconds=avg_wall,
            step_times_ms=all_step_times,
            num_sequences_generated=avg_seqs,
            total_tokens_generated=int(avg_tokens),
            config_used=algorithm.config.to_dict(),
        )
        result.compute_statistics()

        # Estimate memory from config
        meta = _BUILTIN_METADATA.get(resolve_name(algorithm.name))
        if meta:
            result.peak_memory_mb = meta.typical_memory_mb

        return result

    @staticmethod
    def profile_batch(
        algorithms: Sequence[DecodingAlgorithm],
        logit_source: LogitSource,
        prompt_ids: List[int],
        num_warmup: int = 1,
        num_runs: int = 3,
    ) -> List[ProfileResult]:
        """Profile multiple algorithms on the same input."""
        results: List[ProfileResult] = []
        for algo in algorithms:
            result = AlgorithmProfiler.profile(
                algo, logit_source, prompt_ids,
                num_warmup=num_warmup, num_runs=num_runs,
            )
            results.append(result)
        return results

    @staticmethod
    def compare_profiles(results: List[ProfileResult]) -> str:
        """Format a comparison table from multiple profile results.

        Parameters
        ----------
        results:
            List of ``ProfileResult`` objects.

        Returns
        -------
        str
            Formatted comparison table.
        """
        if not results:
            return "(no results to compare)"

        header = (
            f"{'Algorithm':<35} {'Wall(s)':<10} {'Tok/s':<10} "
            f"{'AvgStep(ms)':<12} {'P95(ms)':<10} {'Seqs':<6} {'Tokens':<8}"
        )
        lines = [header, "-" * len(header)]

        sorted_results = sorted(results, key=lambda r: r.wall_time_seconds)
        fastest = sorted_results[0].wall_time_seconds if sorted_results else 1.0

        for r in sorted_results:
            slowdown = r.wall_time_seconds / fastest if fastest > 0 else 0.0
            line = (
                f"{r.algorithm_name:<35} "
                f"{r.wall_time_seconds:<10.3f} "
                f"{r.tokens_per_second:<10.1f} "
                f"{r.avg_step_time_ms:<12.2f} "
                f"{r.p95_step_time_ms:<10.2f} "
                f"{r.num_sequences_generated:<6} "
                f"{r.total_tokens_generated:<8}"
            )
            if slowdown > 1.05:
                line += f"  ({slowdown:.1f}x slower)"
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def estimate_cost(
        algorithm_name: str,
        num_sequences: int = 20,
        max_new_tokens: int = 100,
        vocab_size: int = 50000,
    ) -> Dict[str, float]:
        """Estimate computational cost without running the algorithm.

        Uses metadata complexity info and typical benchmarks to produce
        rough estimates.

        Returns
        -------
        dict
            Estimated ``time_ms``, ``memory_mb``, ``flops``.
        """
        resolved = resolve_name(algorithm_name)
        meta = _BUILTIN_METADATA.get(resolved)
        if meta is None:
            return {"time_ms": 0.0, "memory_mb": 0.0, "flops": 0.0}

        # Scale typical time by sequence count and length
        base_time = meta.typical_time_ms
        base_mem = meta.typical_memory_mb

        seq_scale = num_sequences / 20.0
        len_scale = max_new_tokens / 100.0

        estimated_time = base_time * seq_scale * len_scale
        estimated_mem = base_mem * seq_scale

        # Rough FLOPS estimate
        if meta.complexity.scales_with_beam_width:
            estimated_flops = float(num_sequences * max_new_tokens * vocab_size)
        else:
            estimated_flops = float(num_sequences * max_new_tokens * math.log2(vocab_size + 1))

        return {
            "time_ms": estimated_time,
            "memory_mb": estimated_mem,
            "flops": estimated_flops,
        }


# =========================================================================
# TaskDomain — describes properties of a generation task
# =========================================================================


@dataclass
class TaskDomain:
    """Description of a generation task used by AlgorithmSelector."""

    name: str = "general"
    needs_diversity: bool = True
    needs_quality: bool = True
    diversity_importance: float = 0.5  # 0..1
    quality_importance: float = 0.5   # 0..1
    max_latency_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None
    deterministic_required: bool = False
    num_sequences: int = 20
    max_new_tokens: int = 100
    vocab_size: int = 50000
    preferred_families: Optional[List[AlgorithmFamily]] = None
    excluded_algorithms: Optional[Set[str]] = None


# =========================================================================
# Recommendation
# =========================================================================


@dataclass
class AlgorithmRecommendation:
    """A single algorithm recommendation with rationale."""

    algorithm_name: str
    score: float
    rationale: str
    estimated_time_ms: float = 0.0
    estimated_memory_mb: float = 0.0
    suggested_params: Optional[Dict[str, Any]] = None


# =========================================================================
# AlgorithmSelector
# =========================================================================


class AlgorithmSelector:
    """Select algorithms based on task requirements and resource constraints.

    The selector scores each registered algorithm against the task domain
    properties and returns ranked recommendations.
    """

    # -- domain presets -----------------------------------------------------

    _TASK_PRESETS: Dict[str, TaskDomain] = {
        "creative_writing": TaskDomain(
            name="creative_writing",
            needs_diversity=True,
            needs_quality=True,
            diversity_importance=0.7,
            quality_importance=0.3,
        ),
        "translation": TaskDomain(
            name="translation",
            needs_diversity=False,
            needs_quality=True,
            diversity_importance=0.2,
            quality_importance=0.8,
            preferred_families=[AlgorithmFamily.BEAM_SEARCH, AlgorithmFamily.RERANKING],
        ),
        "code_generation": TaskDomain(
            name="code_generation",
            needs_diversity=True,
            needs_quality=True,
            diversity_importance=0.5,
            quality_importance=0.5,
        ),
        "summarisation": TaskDomain(
            name="summarisation",
            needs_diversity=False,
            needs_quality=True,
            diversity_importance=0.3,
            quality_importance=0.7,
            preferred_families=[AlgorithmFamily.BEAM_SEARCH],
        ),
        "brainstorming": TaskDomain(
            name="brainstorming",
            needs_diversity=True,
            needs_quality=False,
            diversity_importance=0.9,
            quality_importance=0.1,
        ),
        "dialogue": TaskDomain(
            name="dialogue",
            needs_diversity=True,
            needs_quality=True,
            diversity_importance=0.6,
            quality_importance=0.4,
            max_latency_ms=100.0,
        ),
        "data_augmentation": TaskDomain(
            name="data_augmentation",
            needs_diversity=True,
            needs_quality=False,
            diversity_importance=0.8,
            quality_importance=0.2,
        ),
        "question_answering": TaskDomain(
            name="question_answering",
            needs_diversity=False,
            needs_quality=True,
            diversity_importance=0.1,
            quality_importance=0.9,
            preferred_families=[AlgorithmFamily.BEAM_SEARCH, AlgorithmFamily.SAMPLING],
        ),
    }

    # -- family affinity scores: how well a family matches diversity needs ---
    _FAMILY_DIVERSITY_SCORES: Dict[AlgorithmFamily, float] = {
        AlgorithmFamily.SAMPLING: 0.6,
        AlgorithmFamily.BEAM_SEARCH: 0.7,
        AlgorithmFamily.RERANKING: 0.9,
        AlgorithmFamily.PARTICLE: 0.85,
        AlgorithmFamily.QUALITY_DIVERSITY: 0.95,
    }

    _FAMILY_QUALITY_SCORES: Dict[AlgorithmFamily, float] = {
        AlgorithmFamily.SAMPLING: 0.5,
        AlgorithmFamily.BEAM_SEARCH: 0.9,
        AlgorithmFamily.RERANKING: 0.85,
        AlgorithmFamily.PARTICLE: 0.7,
        AlgorithmFamily.QUALITY_DIVERSITY: 0.8,
    }

    @classmethod
    def get_preset(cls, task_name: str) -> TaskDomain:
        """Return a preset task domain by name."""
        key = task_name.lower().replace(" ", "_").replace("-", "_")
        if key in cls._TASK_PRESETS:
            return copy.deepcopy(cls._TASK_PRESETS[key])
        raise KeyError(
            f"Unknown task preset '{task_name}'. "
            f"Available: {sorted(cls._TASK_PRESETS.keys())}"
        )

    @classmethod
    def list_presets(cls) -> List[str]:
        """Return all available task preset names."""
        return sorted(cls._TASK_PRESETS.keys())

    @classmethod
    def select_for_task(
        cls,
        task: Union[str, TaskDomain],
        top_k: int = 5,
    ) -> List[AlgorithmRecommendation]:
        """Select and rank algorithms for a given task.

        Parameters
        ----------
        task:
            A task preset name (str) or a ``TaskDomain`` object.
        top_k:
            Number of top recommendations to return.

        Returns
        -------
        list of AlgorithmRecommendation
            Ranked recommendations with scores and rationale.
        """
        if isinstance(task, str):
            task = cls.get_preset(task)

        ExtendedRegistry._ensure_init()
        recommendations: List[AlgorithmRecommendation] = []

        for name, meta in ExtendedRegistry._metadata.items():
            if task.excluded_algorithms and name in task.excluded_algorithms:
                continue

            score = cls._score_algorithm(meta, task)

            # Check latency constraint
            cost = AlgorithmProfiler.estimate_cost(
                name,
                num_sequences=task.num_sequences,
                max_new_tokens=task.max_new_tokens,
                vocab_size=task.vocab_size,
            )
            if task.max_latency_ms is not None and cost["time_ms"] > task.max_latency_ms:
                score *= 0.3  # heavy penalty but don't exclude

            if task.max_memory_mb is not None and cost["memory_mb"] > task.max_memory_mb:
                score *= 0.3

            if task.deterministic_required and not meta.deterministic:
                score *= 0.1

            if task.preferred_families:
                if meta.family in task.preferred_families:
                    score *= 1.3
                else:
                    score *= 0.7

            rationale = cls._build_rationale(meta, task, score)
            params = cls._suggest_params_for_task(meta, task)

            recommendations.append(AlgorithmRecommendation(
                algorithm_name=name,
                score=score,
                rationale=rationale,
                estimated_time_ms=cost["time_ms"],
                estimated_memory_mb=cost["memory_mb"],
                suggested_params=params,
            ))

        recommendations.sort(key=lambda r: -r.score)
        return recommendations[:top_k]

    @classmethod
    def select_for_budget(
        cls,
        time_budget_ms: float,
        memory_budget_mb: float,
        num_sequences: int = 20,
        max_new_tokens: int = 100,
        vocab_size: int = 50000,
    ) -> List[AlgorithmRecommendation]:
        """Select algorithms that fit within resource constraints.

        Parameters
        ----------
        time_budget_ms:
            Maximum wall-clock time in milliseconds.
        memory_budget_mb:
            Maximum memory in megabytes.
        num_sequences:
            Target number of sequences.
        max_new_tokens:
            Maximum generation length.
        vocab_size:
            Vocabulary size.

        Returns
        -------
        list of AlgorithmRecommendation
            Feasible algorithms, sorted by estimated efficiency.
        """
        ExtendedRegistry._ensure_init()
        feasible: List[AlgorithmRecommendation] = []

        for name, meta in ExtendedRegistry._metadata.items():
            cost = AlgorithmProfiler.estimate_cost(
                name,
                num_sequences=num_sequences,
                max_new_tokens=max_new_tokens,
                vocab_size=vocab_size,
            )

            fits_time = cost["time_ms"] <= time_budget_ms
            fits_mem = cost["memory_mb"] <= memory_budget_mb

            if not fits_time or not fits_mem:
                continue

            # Score by how much headroom is left (prefer efficient algorithms)
            time_ratio = 1.0 - (cost["time_ms"] / time_budget_ms)
            mem_ratio = 1.0 - (cost["memory_mb"] / memory_budget_mb)
            score = 0.6 * time_ratio + 0.4 * mem_ratio

            rationale_parts = [
                f"Estimated {cost['time_ms']:.0f}ms "
                f"({time_ratio * 100:.0f}% time headroom)",
                f"Estimated {cost['memory_mb']:.0f}MB "
                f"({mem_ratio * 100:.0f}% memory headroom)",
            ]

            feasible.append(AlgorithmRecommendation(
                algorithm_name=name,
                score=score,
                rationale="; ".join(rationale_parts),
                estimated_time_ms=cost["time_ms"],
                estimated_memory_mb=cost["memory_mb"],
            ))

        feasible.sort(key=lambda r: -r.score)
        return feasible

    @classmethod
    def select_diverse_set(
        cls,
        n: int = 5,
        task: Optional[Union[str, TaskDomain]] = None,
    ) -> List[AlgorithmRecommendation]:
        """Select a diverse set of algorithms covering different families.

        Ensures the returned set spans as many algorithm families as possible
        while still being relevant to the task.

        Parameters
        ----------
        n:
            Target number of algorithms.
        task:
            Optional task context for scoring.

        Returns
        -------
        list of AlgorithmRecommendation
        """
        if task is not None:
            all_recs = cls.select_for_task(task, top_k=100)
        else:
            # Score all algorithms equally
            ExtendedRegistry._ensure_init()
            all_recs = []
            for name, meta in ExtendedRegistry._metadata.items():
                all_recs.append(AlgorithmRecommendation(
                    algorithm_name=name,
                    score=1.0,
                    rationale="Included for family coverage.",
                ))

        # Greedily pick the top algorithm from each family
        selected: List[AlgorithmRecommendation] = []
        families_covered: Set[AlgorithmFamily] = set()
        remaining = list(all_recs)

        while len(selected) < n and remaining:
            # Prefer algorithms from uncovered families
            best_idx = -1
            best_score = -1.0
            for idx, rec in enumerate(remaining):
                meta = ExtendedRegistry._metadata.get(rec.algorithm_name)
                family_bonus = 2.0 if (meta and meta.family not in families_covered) else 0.0
                adjusted = rec.score + family_bonus
                if adjusted > best_score:
                    best_score = adjusted
                    best_idx = idx

            if best_idx < 0:
                break
            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            meta = ExtendedRegistry._metadata.get(chosen.algorithm_name)
            if meta:
                families_covered.add(meta.family)

        return selected

    # -- internal helpers ---------------------------------------------------

    @classmethod
    def _score_algorithm(cls, meta: AlgorithmMetadata, task: TaskDomain) -> float:
        """Compute a relevance score for an algorithm given a task."""
        div_score = cls._FAMILY_DIVERSITY_SCORES.get(meta.family, 0.5)
        qual_score = cls._FAMILY_QUALITY_SCORES.get(meta.family, 0.5)

        score = (
            task.diversity_importance * div_score
            + task.quality_importance * qual_score
        )

        # Bonus for algorithms with explicit diversity tags
        if task.needs_diversity and "diversity" in meta.tags:
            score += 0.15

        # Bonus for novel algorithms when diversity is paramount
        if task.diversity_importance > 0.6 and "novel" in meta.tags:
            score += 0.1

        # Slight penalty for very heavy algorithms when latency matters
        if task.max_latency_ms is not None and meta.typical_time_ms > task.max_latency_ms * 0.5:
            score *= 0.8

        return max(0.0, min(1.0, score))

    @classmethod
    def _build_rationale(
        cls,
        meta: AlgorithmMetadata,
        task: TaskDomain,
        score: float,
    ) -> str:
        """Build a human-readable rationale string."""
        parts: List[str] = []

        if score >= 0.8:
            parts.append(f"Excellent fit for {task.name}")
        elif score >= 0.6:
            parts.append(f"Good fit for {task.name}")
        elif score >= 0.4:
            parts.append(f"Moderate fit for {task.name}")
        else:
            parts.append(f"Marginal fit for {task.name}")

        parts.append(f"Family: {meta.family.label()}")

        if task.needs_diversity and "diversity" in meta.tags:
            parts.append("Has explicit diversity mechanisms")

        if meta.deterministic:
            parts.append("Deterministic output")

        if meta.typical_time_ms > 0:
            parts.append(f"~{meta.typical_time_ms:.0f}ms typical latency")

        return ". ".join(parts) + "."

    @classmethod
    def _suggest_params_for_task(
        cls,
        meta: AlgorithmMetadata,
        task: TaskDomain,
    ) -> Dict[str, Any]:
        """Suggest parameter values tuned for the task."""
        params = dict(meta.default_params)

        # Increase temperature for high-diversity tasks
        if "temperature" in meta.param_specs and task.diversity_importance > 0.6:
            default_temp = meta.default_params.get("temperature", 1.0)
            boost = 1.0 + 0.5 * (task.diversity_importance - 0.5)
            suggested_temp = default_temp * boost
            spec = meta.param_specs["temperature"]
            if spec.max_value is not None:
                suggested_temp = min(suggested_temp, spec.max_value)
            params["temperature"] = round(suggested_temp, 2)

        # Decrease temperature for high-quality tasks
        if "temperature" in meta.param_specs and task.quality_importance > 0.7:
            default_temp = meta.default_params.get("temperature", 1.0)
            suggested_temp = default_temp * 0.8
            spec = meta.param_specs["temperature"]
            if spec.min_value is not None:
                suggested_temp = max(suggested_temp, spec.min_value)
            params["temperature"] = round(suggested_temp, 2)

        # Increase diversity penalty for beam methods in high-diversity tasks
        if "diversity_penalty" in meta.param_specs and task.diversity_importance > 0.5:
            default_dp = meta.default_params.get("diversity_penalty", 1.0)
            params["diversity_penalty"] = round(
                default_dp * (1.0 + task.diversity_importance), 2
            )

        # Increase quality weight for quality-focused tasks
        if "quality_weight" in meta.param_specs and task.quality_importance > 0.6:
            params["quality_weight"] = round(
                min(0.95, 0.5 + task.quality_importance * 0.4), 2
            )
            if "novelty_weight" in params:
                params["novelty_weight"] = round(1.0 - params["quality_weight"], 2)
            if "diversity_weight" in params:
                params["diversity_weight"] = round(1.0 - params["quality_weight"], 2)

        return params


# =========================================================================
# ParameterAnalyser — understand param sensitivity
# =========================================================================


class ParameterAnalyser:
    """Analyse parameter sensitivity and importance for algorithms."""

    @staticmethod
    def param_sensitivity(
        algorithm_name: str,
        param_name: str,
        logit_source: LogitSource,
        prompt_ids: List[int],
        n_points: int = 5,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Measure output sensitivity to a single parameter.

        Sweeps the parameter across its range and measures how much the
        outputs change.

        Parameters
        ----------
        algorithm_name:
            Algorithm to analyse.
        param_name:
            Parameter to sweep.
        logit_source:
            Logit source for generation.
        prompt_ids:
            Prompt token IDs.
        n_points:
            Number of sweep points.
        seed:
            Random seed.

        Returns
        -------
        dict
            Contains ``values``, ``output_lengths``, ``unique_tokens``,
            ``variance``, ``sensitivity_score``.
        """
        resolved = resolve_name(algorithm_name)
        meta = ExtendedRegistry.get_metadata(resolved)
        spec = meta.param_specs.get(param_name)
        if spec is None:
            raise ValueError(
                f"Parameter '{param_name}' not found for algorithm '{resolved}'. "
                f"Known: {meta.param_names()}"
            )

        lo, hi = spec.suggested_range()
        if spec.dtype == "float":
            if spec.log_scale and float(lo) > 0:
                values = list(np.exp(np.linspace(
                    math.log(float(lo)), math.log(float(hi)), n_points
                )))
            else:
                values = list(np.linspace(float(lo), float(hi), n_points))
        elif spec.dtype == "int":
            values = list(range(int(lo), int(hi) + 1, max(1, (int(hi) - int(lo)) // n_points)))
            values = values[:n_points]
        elif spec.choices is not None:
            values = list(spec.choices)[:n_points]
        else:
            return {"error": f"Cannot sweep parameter of dtype '{spec.dtype}'"}

        output_lengths: List[float] = []
        unique_tokens_counts: List[int] = []
        all_sequences: List[List[TokenSequence]] = []

        for val in values:
            params = dict(meta.default_params)
            params[param_name] = val
            config = DecodingConfig(
                algorithm_name=resolved,
                num_sequences=5,
                max_new_tokens=50,
                seed=seed,
                params=params,
            )
            try:
                algo = AlgorithmRegistry.create(resolved, config)
                seqs = algo.generate(logit_source, prompt_ids)
                all_sequences.append(seqs)
                avg_len = statistics.mean(len(s) for s in seqs) if seqs else 0
                output_lengths.append(avg_len)
                all_toks: Set[int] = set()
                for s in seqs:
                    all_toks.update(s)
                unique_tokens_counts.append(len(all_toks))
            except Exception as exc:
                logger.warning("Sensitivity analysis failed for %s=%s: %s", param_name, val, exc)
                output_lengths.append(0.0)
                unique_tokens_counts.append(0)
                all_sequences.append([])

        # Compute sensitivity score
        len_variance = statistics.variance(output_lengths) if len(output_lengths) > 1 else 0.0
        tok_variance = statistics.variance(
            [float(x) for x in unique_tokens_counts]
        ) if len(unique_tokens_counts) > 1 else 0.0
        len_range = max(output_lengths) - min(output_lengths) if output_lengths else 0
        tok_range = max(unique_tokens_counts) - min(unique_tokens_counts) if unique_tokens_counts else 0

        mean_len = statistics.mean(output_lengths) if output_lengths else 1.0
        mean_tok = statistics.mean(unique_tokens_counts) if unique_tokens_counts else 1.0
        normalised_len_var = len_variance / (mean_len ** 2) if mean_len > 0 else 0.0
        normalised_tok_var = tok_variance / (mean_tok ** 2) if mean_tok > 0 else 0.0
        sensitivity_score = (normalised_len_var + normalised_tok_var) / 2.0

        return {
            "parameter": param_name,
            "values": values,
            "output_lengths": output_lengths,
            "unique_tokens": unique_tokens_counts,
            "length_variance": len_variance,
            "token_variance": tok_variance,
            "length_range": len_range,
            "token_range": tok_range,
            "sensitivity_score": sensitivity_score,
        }

    @staticmethod
    def rank_param_importance(
        algorithm_name: str,
        logit_source: LogitSource,
        prompt_ids: List[int],
        seed: Optional[int] = 42,
    ) -> List[Tuple[str, float]]:
        """Rank parameters by their sensitivity score.

        Sweeps each parameter and returns them sorted by decreasing
        sensitivity.

        Returns
        -------
        list of (param_name, sensitivity_score)
        """
        resolved = resolve_name(algorithm_name)
        meta = ExtendedRegistry.get_metadata(resolved)
        rankings: List[Tuple[str, float]] = []

        for pname in meta.param_specs:
            try:
                result = ParameterAnalyser.param_sensitivity(
                    resolved, pname, logit_source, prompt_ids,
                    n_points=4, seed=seed,
                )
                rankings.append((pname, result.get("sensitivity_score", 0.0)))
            except Exception:
                rankings.append((pname, 0.0))

        rankings.sort(key=lambda x: -x[1])
        return rankings


# =========================================================================
# AlgorithmComparator — structured comparison
# =========================================================================


class AlgorithmComparator:
    """Compare algorithms along multiple axes."""

    @staticmethod
    def compare_metadata(names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Compare metadata of multiple algorithms side by side.

        Parameters
        ----------
        names:
            Algorithm names to compare.

        Returns
        -------
        dict
            Mapping algorithm name → metadata dict.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for name in names:
            resolved = resolve_name(name)
            meta = ExtendedRegistry.get_metadata(resolved)
            result[resolved] = meta.to_dict()
        return result

    @staticmethod
    def compare_params(names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Compare parameter specifications across algorithms.

        Returns
        -------
        dict
            Mapping algorithm name → param specs dict.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for name in names:
            resolved = resolve_name(name)
            meta = ExtendedRegistry.get_metadata(resolved)
            result[resolved] = {
                pname: {
                    "dtype": spec.dtype,
                    "default": spec.default,
                    "range": spec.suggested_range(),
                    "description": spec.description,
                }
                for pname, spec in meta.param_specs.items()
            }
        return result

    @staticmethod
    def feature_matrix(names: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Build a feature matrix for algorithm comparison.

        Parameters
        ----------
        names:
            Algorithms to include.  Defaults to all registered.

        Returns
        -------
        dict
            Mapping algorithm name → feature dict.
        """
        ExtendedRegistry._ensure_init()
        if names is None:
            names = list(ExtendedRegistry._metadata.keys())

        matrix: Dict[str, Dict[str, Any]] = {}
        for name in names:
            resolved = resolve_name(name)
            meta = ExtendedRegistry._metadata.get(resolved)
            if meta is None:
                continue
            matrix[resolved] = {
                "family": meta.family.value,
                "category": meta.category,
                "deterministic": meta.deterministic,
                "supports_batching": meta.supports_batching,
                "num_params": len(meta.param_specs),
                "has_temperature": "temperature" in meta.param_specs,
                "typical_time_ms": meta.typical_time_ms,
                "typical_memory_mb": meta.typical_memory_mb,
                "tags": sorted(meta.tags),
                "complexity_time": meta.complexity.time,
                "complexity_space": meta.complexity.space,
                "scales_with_vocab": meta.complexity.scales_with_vocab,
                "scales_with_beam_width": meta.complexity.scales_with_beam_width,
            }
        return matrix

    @staticmethod
    def format_comparison_table(
        names: Sequence[str],
        fields: Optional[Sequence[str]] = None,
    ) -> str:
        """Format a text comparison table for the given algorithms.

        Parameters
        ----------
        names:
            Algorithm names.
        fields:
            Feature fields to include.  Defaults to a standard set.

        Returns
        -------
        str
            Formatted table.
        """
        if fields is None:
            fields = [
                "family", "category", "deterministic",
                "typical_time_ms", "typical_memory_mb", "num_params",
            ]

        matrix = AlgorithmComparator.feature_matrix(names)
        col_width = 22
        name_width = 35

        # Header
        header = f"{'Algorithm':<{name_width}}"
        for f in fields:
            header += f" {f:<{col_width}}"
        lines = [header, "-" * len(header)]

        for name in names:
            resolved = resolve_name(name)
            features = matrix.get(resolved, {})
            row = f"{resolved:<{name_width}}"
            for f in fields:
                val = features.get(f, "N/A")
                row += f" {str(val):<{col_width}}"
            lines.append(row)

        return "\n".join(lines)


# =========================================================================
# Convenience functions — module-level API
# =========================================================================


def list_all_algorithms() -> List[str]:
    """Return all known algorithm names."""
    return ExtendedRegistry.list_algorithms()


def get_algorithm_info(name: str) -> Dict[str, Any]:
    """Return metadata dict for an algorithm."""
    return ExtendedRegistry.get_metadata(name).to_dict()


def create_algorithm(name: str, **kwargs: Any) -> DecodingAlgorithm:
    """Create an algorithm instance by name with optional overrides."""
    return ExtendedRegistry.create(name, **kwargs)


def create_algorithm_from_config(config: DecodingConfig) -> DecodingAlgorithm:
    """Create an algorithm instance from a DecodingConfig."""
    return ExtendedRegistry.create_from_config(config)


def validate_config(config: DecodingConfig) -> List[str]:
    """Validate a DecodingConfig, returning a list of issues."""
    issues = ConfigValidator.validate(config)
    issues.extend(ConfigValidator.validate_cross_params(config))
    return issues


def suggest_algorithm(
    task: str = "general",
    time_budget_ms: Optional[float] = None,
    memory_budget_mb: Optional[float] = None,
) -> List[AlgorithmRecommendation]:
    """Suggest algorithms for a task, optionally within resource budgets.

    Parameters
    ----------
    task:
        Task preset name or "general".
    time_budget_ms:
        Optional time budget.
    memory_budget_mb:
        Optional memory budget.

    Returns
    -------
    list of AlgorithmRecommendation
    """
    try:
        domain = AlgorithmSelector.get_preset(task)
    except KeyError:
        domain = TaskDomain(name=task)

    if time_budget_ms is not None:
        domain.max_latency_ms = time_budget_ms
    if memory_budget_mb is not None:
        domain.max_memory_mb = memory_budget_mb

    return AlgorithmSelector.select_for_task(domain)


def algorithm_summary() -> str:
    """Return a formatted summary table of all algorithms."""
    return ExtendedRegistry.summary_table()


# =========================================================================
# Registration helpers for algorithm modules
# =========================================================================


def register_algorithm(
    name: str,
    cls: Type[DecodingAlgorithm],
    metadata: Optional[AlgorithmMetadata] = None,
) -> None:
    """Register an algorithm with both the base and extended registries.

    This is the recommended entry point for algorithm module ``__init__``
    blocks or explicit registration calls.
    """
    ExtendedRegistry.register(name, cls, metadata)


def register_with_metadata(
    name: str,
    description: str = "",
    category: str = "sampling",
    family: Union[str, AlgorithmFamily] = AlgorithmFamily.SAMPLING,
    default_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Callable[[Type[DecodingAlgorithm]], Type[DecodingAlgorithm]]:
    """Decorator that registers an algorithm with inline metadata.

    Usage::

        @register_with_metadata(
            "my_algo",
            description="My custom algorithm",
            category="novel",
            family="sampling",
            default_params={"temp": 1.0},
        )
        class MyAlgo(DecodingAlgorithm):
            ...
    """
    if isinstance(family, str):
        family = AlgorithmFamily.from_string(family)

    meta = AlgorithmMetadata(
        name=name,
        description=description,
        category=category,
        family=family,
        default_params=default_params or {},
        **kwargs,
    )

    def decorator(cls: Type[DecodingAlgorithm]) -> Type[DecodingAlgorithm]:
        register_algorithm(name, cls, meta)
        return cls

    return decorator


# =========================================================================
# Discovery / import helpers
# =========================================================================


_ALGORITHM_MODULES: Dict[str, str] = {
    "temperature_sampling": "src.algorithms.temperature",
    "top_k_sampling": "src.algorithms.topk",
    "nucleus_sampling": "src.algorithms.nucleus",
    "typical_decoding": "src.algorithms.typical",
    "diverse_beam_search": "src.algorithms.diverse_beam",
    "contrastive_search": "src.algorithms.contrastive",
    "dpp_reranking": "src.algorithms.dpp",
    "mbr_diversity": "src.algorithms.mbr",
    "stein_variational_decoding": "src.algorithms.svd",
    "quality_diversity_beam_search": "src.algorithms.qdbs",
    "mcts_decoding": "src.algorithms.mcts",
    "stochastic_beam_search": "src.algorithms.stochastic_beam",
    "ancestral_diverse_sampling": "src.algorithms.ancestral",
}

_ALGORITHM_CLASSES: Dict[str, str] = {
    "temperature_sampling": "TemperatureSampling",
    "top_k_sampling": "TopKSampling",
    "nucleus_sampling": "NucleusSampling",
    "typical_decoding": "TypicalDecoding",
    "diverse_beam_search": "DiverseBeamSearch",
    "contrastive_search": "ContrastiveSearch",
    "dpp_reranking": "DPPReranking",
    "mbr_diversity": "MBRDiversity",
    "stein_variational_decoding": "SteinVariationalDecoding",
    "quality_diversity_beam_search": "QualityDiversityBeamSearch",
    "mcts_decoding": "MCTSDecoding",
    "stochastic_beam_search": "StochasticBeamSearch",
    "ancestral_diverse_sampling": "AncestralDiverseSampling",
}


def discover_algorithms(auto_register: bool = True) -> Dict[str, Type[DecodingAlgorithm]]:
    """Attempt to import all known algorithm modules and return the
    discovered classes.

    Parameters
    ----------
    auto_register:
        If True, register discovered classes with the extended registry.

    Returns
    -------
    dict
        Mapping canonical name → algorithm class.
    """
    import importlib

    discovered: Dict[str, Type[DecodingAlgorithm]] = {}
    for canon_name, module_path in _ALGORITHM_MODULES.items():
        class_name = _ALGORITHM_CLASSES.get(canon_name)
        if class_name is None:
            continue
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name, None)
            if cls is not None and isinstance(cls, type) and issubclass(cls, DecodingAlgorithm):
                discovered[canon_name] = cls
                if auto_register:
                    meta = _BUILTIN_METADATA.get(canon_name)
                    ExtendedRegistry.register(canon_name, cls, meta)
                logger.debug("Discovered algorithm '%s' from %s", canon_name, module_path)
        except ImportError as exc:
            logger.debug("Could not import %s: %s", module_path, exc)
        except Exception as exc:
            logger.warning("Error importing %s: %s", module_path, exc)

    return discovered


def ensure_registered(name: str) -> Type[DecodingAlgorithm]:
    """Ensure an algorithm is registered, importing its module if needed.

    Returns the algorithm class.

    Raises
    ------
    KeyError
        If the algorithm cannot be found or imported.
    """
    resolved = resolve_name(name)
    if AlgorithmRegistry.is_registered(resolved):
        return AlgorithmRegistry.get(resolved)

    module_path = _ALGORITHM_MODULES.get(resolved)
    class_name = _ALGORITHM_CLASSES.get(resolved)
    if module_path is None or class_name is None:
        raise KeyError(f"Unknown algorithm '{name}' (resolved: '{resolved}')")

    import importlib
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise KeyError(
            f"Could not import module '{module_path}' for algorithm '{resolved}': {exc}"
        ) from exc

    cls = getattr(mod, class_name, None)
    if cls is None:
        raise KeyError(
            f"Class '{class_name}' not found in module '{module_path}'"
        )

    meta = _BUILTIN_METADATA.get(resolved)
    ExtendedRegistry.register(resolved, cls, meta)
    return cls


# =========================================================================
# ConfigBuilder — fluent API for building configs
# =========================================================================


class ConfigBuilder:
    """Fluent builder for ``DecodingConfig`` instances.

    Usage::

        config = (
            ConfigBuilder("temperature_sampling")
            .num_sequences(10)
            .max_new_tokens(50)
            .param("temperature", 1.5)
            .seed(42)
            .build()
        )
    """

    def __init__(self, algorithm_name: str) -> None:
        self._name = resolve_name(algorithm_name)
        self._num_sequences = 20
        self._max_new_tokens = 100
        self._min_new_tokens = 10
        self._seed: Optional[int] = None
        self._temperature = 1.0
        self._repetition_penalty = 1.0
        self._no_repeat_ngram_size = 0
        self._eos_token_id: Optional[int] = None
        self._pad_token_id: Optional[int] = None
        self._params: Dict[str, Any] = {}

        # Pre-fill defaults from metadata
        meta = _BUILTIN_METADATA.get(self._name)
        if meta:
            self._params.update(meta.default_params)

    def num_sequences(self, n: int) -> "ConfigBuilder":
        self._num_sequences = n
        return self

    def max_new_tokens(self, n: int) -> "ConfigBuilder":
        self._max_new_tokens = n
        return self

    def min_new_tokens(self, n: int) -> "ConfigBuilder":
        self._min_new_tokens = n
        return self

    def seed(self, s: Optional[int]) -> "ConfigBuilder":
        self._seed = s
        return self

    def temperature(self, t: float) -> "ConfigBuilder":
        self._temperature = t
        return self

    def repetition_penalty(self, p: float) -> "ConfigBuilder":
        self._repetition_penalty = p
        return self

    def no_repeat_ngram_size(self, n: int) -> "ConfigBuilder":
        self._no_repeat_ngram_size = n
        return self

    def eos_token_id(self, tid: Optional[int]) -> "ConfigBuilder":
        self._eos_token_id = tid
        return self

    def pad_token_id(self, tid: Optional[int]) -> "ConfigBuilder":
        self._pad_token_id = tid
        return self

    def param(self, name: str, value: Any) -> "ConfigBuilder":
        self._params[name] = value
        return self

    def params(self, **kwargs: Any) -> "ConfigBuilder":
        self._params.update(kwargs)
        return self

    def from_defaults(self) -> "ConfigBuilder":
        """Reset params to algorithm defaults from metadata."""
        meta = _BUILTIN_METADATA.get(self._name)
        if meta:
            self._params = dict(meta.default_params)
        return self

    def build(self) -> DecodingConfig:
        """Build and return the ``DecodingConfig``."""
        return DecodingConfig(
            algorithm_name=self._name,
            num_sequences=self._num_sequences,
            max_new_tokens=self._max_new_tokens,
            min_new_tokens=self._min_new_tokens,
            seed=self._seed,
            temperature=self._temperature,
            repetition_penalty=self._repetition_penalty,
            no_repeat_ngram_size=self._no_repeat_ngram_size,
            eos_token_id=self._eos_token_id,
            pad_token_id=self._pad_token_id,
            params=dict(self._params),
        )

    def build_and_create(self) -> DecodingAlgorithm:
        """Build the config and instantiate the algorithm."""
        config = self.build()
        return ExtendedRegistry.create_from_config(config)

    def validate(self) -> List[str]:
        """Validate the current configuration without building."""
        config = self.build()
        return ConfigValidator.validate(config)


# =========================================================================
# RegistrySnapshot — serialise and restore registry state
# =========================================================================


@dataclass
class RegistrySnapshot:
    """Serialisable snapshot of registry metadata."""

    metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def capture(cls) -> "RegistrySnapshot":
        """Capture the current state of the extended registry."""
        ExtendedRegistry._ensure_init()
        meta_dict: Dict[str, Dict[str, Any]] = {}
        for name, meta in ExtendedRegistry._metadata.items():
            meta_dict[name] = meta.to_dict()
        return cls(
            metadata=meta_dict,
            aliases=dict(_NAME_ALIASES),
        )

    def restore_metadata(self) -> None:
        """Restore metadata from this snapshot into the extended registry."""
        for name, meta_dict in self.metadata.items():
            meta = AlgorithmMetadata.from_dict(meta_dict)
            ExtendedRegistry._metadata[name] = meta
        ExtendedRegistry._initialised = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "aliases": self.aliases,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegistrySnapshot":
        return cls(
            metadata=d.get("metadata", {}),
            aliases=d.get("aliases", {}),
            timestamp=d.get("timestamp", 0.0),
        )


# =========================================================================
# __all__
# =========================================================================

__all__ = [
    # Enums
    "AlgorithmFamily",
    # Dataclasses
    "AlgorithmMetadata",
    "ComplexityInfo",
    "ParamSpec",
    "ProfileResult",
    "EnsembleMember",
    "EnsembleConfig",
    "TaskDomain",
    "AlgorithmRecommendation",
    "RegistrySnapshot",
    # Core classes
    "ExtendedRegistry",
    "AlgorithmFactory",
    "ConfigValidator",
    "AlgorithmProfiler",
    "AlgorithmSelector",
    "AlgorithmComparator",
    "ParameterAnalyser",
    "ConfigBuilder",
    # Functions
    "resolve_name",
    "list_all_algorithms",
    "get_algorithm_info",
    "create_algorithm",
    "create_algorithm_from_config",
    "validate_config",
    "suggest_algorithm",
    "algorithm_summary",
    "register_algorithm",
    "register_with_metadata",
    "discover_algorithms",
    "ensure_registered",
]
