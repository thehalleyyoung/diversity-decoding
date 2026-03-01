"""
Diversity Decoding Arena Dataset Module.

Builds the "ImageNet of diversity decoding" — a standardized benchmark dataset
for evaluating diversity in language model decoding strategies across domains.
"""

import json
import hashlib
import copy
import re
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

DOMAINS = [
    "creative_writing",
    "code",
    "dialogue",
    "summarization",
    "translation",
    "qa",
    "brainstorming",
    "planning",
]

DATASET_SCHEMA_VERSION = "1.0.0"


class DomainType(Enum):
    CREATIVE_WRITING = "creative_writing"
    CODE = "code"
    DIALOGUE = "dialogue"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QA = "qa"
    BRAINSTORMING = "brainstorming"
    PLANNING = "planning"


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Full configuration for dataset construction."""

    name: str = "diversity-decoding-arena"
    version: str = "1.0.0"
    description: str = "Benchmark dataset for diversity decoding evaluation"
    domains: List[str] = field(default_factory=lambda: list(DOMAINS))
    prompts_per_domain: int = 50
    outputs_per_prompt: int = 10
    algorithms: List[str] = field(
        default_factory=lambda: [
            "greedy",
            "temperature_0.7",
            "temperature_1.0",
            "top_k_50",
            "top_p_0.9",
            "typical_p_0.95",
            "beam_search_5",
            "diverse_beam_5",
            "mirostat_v2",
            "contrastive_search",
        ]
    )
    models: List[str] = field(
        default_factory=lambda: ["gpt2", "llama-7b", "mistral-7b"]
    )
    max_output_tokens: int = 256
    seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    cache_dir: str = ".cache/dataset"
    output_dir: str = "output/dataset"
    num_judges: int = 3
    judgment_dimensions: List[str] = field(
        default_factory=lambda: [
            "lexical_diversity",
            "semantic_diversity",
            "structural_diversity",
            "overall_quality",
            "coherence",
        ]
    )
    min_prompt_length: int = 10
    max_prompt_length: int = 500
    min_output_length: int = 20
    stratify_by: str = "domain"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class PromptEntry:
    """A single curated prompt."""

    prompt_id: str
    text: str
    domain: str
    subdomain: str = ""
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "curated"
    char_length: int = 0
    word_count: int = 0

    def __post_init__(self):
        self.char_length = len(self.text)
        self.word_count = len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PromptEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GeneratedOutput:
    """An output generated for a prompt by a specific algorithm/model pair."""

    output_id: str
    prompt_id: str
    text: str
    algorithm: str
    model: str
    tokens: int = 0
    generation_time_ms: float = 0.0
    logprob_sum: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeneratedOutput":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DiversityJudgment:
    """A diversity judgment (simulated or human) for a set of outputs."""

    judgment_id: str
    prompt_id: str
    output_ids: List[str]
    algorithm: str
    model: str
    scores: Dict[str, float] = field(default_factory=dict)
    judge_id: str = "simulated"
    judge_type: str = "heuristic"
    confidence: float = 1.0
    rationale: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def overall_score(self) -> float:
        if not self.scores:
            return 0.0
        return float(np.mean(list(self.scores.values())))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["overall_score"] = self.overall_score
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiversityJudgment":
        d.pop("overall_score", None)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DatasetSplit:
    """A split (train/val/test) of the dataset."""

    name: str
    prompt_ids: List[str] = field(default_factory=list)
    size: int = 0
    domain_distribution: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.size = len(self.prompt_ids)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# DatasetVersion — semantic versioning for dataset releases
# ---------------------------------------------------------------------------

class DatasetVersion:
    """Manages semantic versioning for the dataset."""

    def __init__(self, version_str: str = "1.0.0"):
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version string: {version_str}")
        self.major = int(parts[0])
        self.minor = int(parts[1])
        self.patch = int(parts[2])
        self._changelog: List[Dict[str, str]] = []

    @property
    def version_string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def bump_major(self, note: str = "") -> "DatasetVersion":
        new = DatasetVersion(f"{self.major + 1}.0.0")
        new._changelog = list(self._changelog)
        new._changelog.append({"type": "major", "version": new.version_string, "note": note})
        return new

    def bump_minor(self, note: str = "") -> "DatasetVersion":
        new = DatasetVersion(f"{self.major}.{self.minor + 1}.0")
        new._changelog = list(self._changelog)
        new._changelog.append({"type": "minor", "version": new.version_string, "note": note})
        return new

    def bump_patch(self, note: str = "") -> "DatasetVersion":
        new = DatasetVersion(f"{self.major}.{self.minor}.{self.patch + 1}")
        new._changelog = list(self._changelog)
        new._changelog.append({"type": "patch", "version": new.version_string, "note": note})
        return new

    def is_compatible(self, other: "DatasetVersion") -> bool:
        return self.major == other.major

    def __lt__(self, other: "DatasetVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DatasetVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __repr__(self) -> str:
        return f"DatasetVersion({self.version_string})"

    def get_changelog(self) -> List[Dict[str, str]]:
        return list(self._changelog)


# ---------------------------------------------------------------------------
# DatasetRegistry — manage multiple dataset versions
# ---------------------------------------------------------------------------

class DatasetRegistry:
    """Registry for tracking and loading multiple dataset versions."""

    def __init__(self, registry_dir: str = ".registry"):
        self.registry_dir = Path(registry_dir)
        self._entries: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, version: DatasetVersion, path: str,
                 config: Optional[DatasetConfig] = None, tags: Optional[List[str]] = None) -> str:
        entry_id = f"{name}@{version.version_string}"
        self._entries[entry_id] = {
            "name": name,
            "version": version.version_string,
            "path": path,
            "config": config.to_dict() if config else {},
            "tags": tags or [],
            "registered_at": time.time(),
        }
        logger.info("Registered dataset %s", entry_id)
        return entry_id

    def lookup(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if version:
            return self._entries.get(f"{name}@{version}")
        # Return latest version
        candidates = {k: v for k, v in self._entries.items() if v["name"] == name}
        if not candidates:
            return None
        latest_key = max(candidates, key=lambda k: DatasetVersion(candidates[k]["version"]))
        return candidates[latest_key]

    def list_versions(self, name: str) -> List[str]:
        return sorted(
            [v["version"] for v in self._entries.values() if v["name"] == name],
            key=lambda s: DatasetVersion(s),
        )

    def list_all(self) -> List[Dict[str, Any]]:
        return list(self._entries.values())

    def remove(self, name: str, version: str) -> bool:
        key = f"{name}@{version}"
        if key in self._entries:
            del self._entries[key]
            return True
        return False

    def save(self, path: Optional[str] = None) -> None:
        out_path = Path(path) if path else self.registry_dir / "registry.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(self._entries, f, indent=2)

    def load(self, path: Optional[str] = None) -> None:
        in_path = Path(path) if path else self.registry_dir / "registry.json"
        if in_path.exists():
            with open(in_path) as f:
                self._entries = json.load(f)


# ---------------------------------------------------------------------------
# PromptCurator — filter and balance prompts across domains
# ---------------------------------------------------------------------------

class PromptCurator:
    """Curates, filters, and balances prompts across target domains."""

    DOMAIN_TEMPLATES: Dict[str, List[str]] = {
        "creative_writing": [
            "Write a short story about {topic}.",
            "Compose a poem inspired by {topic}.",
            "Create a dialogue between two characters discussing {topic}.",
            "Describe a scene where {topic} plays a central role.",
            "Write the opening paragraph of a novel set in {topic}.",
        ],
        "code": [
            "Write a Python function that {task}.",
            "Implement a class in Python that {task}.",
            "Debug the following code snippet: {task}.",
            "Refactor this code to be more efficient: {task}.",
            "Write unit tests for a function that {task}.",
        ],
        "dialogue": [
            "Simulate a conversation between a customer and a support agent about {topic}.",
            "Write a debate between two experts on {topic}.",
            "Create an interview with a specialist in {topic}.",
            "Write a Socratic dialogue exploring {topic}.",
            "Compose a negotiation scene involving {topic}.",
        ],
        "summarization": [
            "Summarize the key points of the following text about {topic}.",
            "Provide a one-paragraph summary of {topic}.",
            "Create an executive summary for a report on {topic}.",
            "Distill the main arguments in a discussion about {topic}.",
            "Write a TL;DR for an article about {topic}.",
        ],
        "translation": [
            "Translate the following English text about {topic} to French.",
            "Provide three different translations of this sentence about {topic}.",
            "Translate this technical passage about {topic} for a general audience.",
            "Localize this marketing copy about {topic} for a Japanese audience.",
            "Translate and adapt this idiom related to {topic} into Spanish.",
        ],
        "qa": [
            "What are the main causes of {topic}?",
            "Explain how {topic} works in simple terms.",
            "Compare and contrast {topic_a} and {topic_b}.",
            "What are the advantages and disadvantages of {topic}?",
            "Provide a step-by-step explanation of {topic}.",
        ],
        "brainstorming": [
            "Generate 10 creative ideas for {topic}.",
            "Brainstorm potential solutions to the problem of {topic}.",
            "List innovative approaches to {topic}.",
            "What are unconventional ways to address {topic}?",
            "Propose a novel business idea related to {topic}.",
        ],
        "planning": [
            "Create a project plan for {topic}.",
            "Outline a strategy to achieve {topic}.",
            "Design a roadmap for implementing {topic}.",
            "Write a detailed action plan for {topic}.",
            "Propose a timeline for completing {topic}.",
        ],
    }

    TOPIC_POOL = [
        "climate change", "artificial intelligence", "space exploration",
        "renewable energy", "education reform", "urban planning",
        "healthcare access", "data privacy", "mental health",
        "sustainable agriculture", "ocean conservation", "quantum computing",
        "remote work", "digital literacy", "food security",
        "biodiversity", "autonomous vehicles", "cybersecurity",
        "social media impact", "genetic engineering",
    ]

    def __init__(self, config: DatasetConfig, rng: Optional[np.random.RandomState] = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.RandomState(config.seed)

    def curate(self) -> List[PromptEntry]:
        """Build a balanced collection of prompts across all domains."""
        prompts: List[PromptEntry] = []
        for domain in self.config.domains:
            domain_prompts = self._generate_domain_prompts(domain)
            prompts.extend(domain_prompts)
        return prompts

    def _generate_domain_prompts(self, domain: str) -> List[PromptEntry]:
        templates = self.DOMAIN_TEMPLATES.get(domain, [])
        if not templates:
            return []
        results: List[PromptEntry] = []
        topics = list(self.TOPIC_POOL)
        self.rng.shuffle(topics)
        count = 0
        idx = 0
        while count < self.config.prompts_per_domain:
            template = templates[count % len(templates)]
            topic = topics[idx % len(topics)]
            text = template.replace("{topic}", topic)
            text = text.replace("{task}", topic)
            text = text.replace("{topic_a}", topic)
            second_topic = topics[(idx + 1) % len(topics)]
            text = text.replace("{topic_b}", second_topic)
            difficulty = self.rng.choice(["easy", "medium", "hard"])
            pid = self._make_id(domain, count)
            entry = PromptEntry(
                prompt_id=pid,
                text=text,
                domain=domain,
                subdomain=f"{domain}_{count // 5}",
                difficulty=difficulty,
                source="template_generated",
            )
            if self._passes_filter(entry):
                results.append(entry)
                count += 1
            idx += 1
            if idx > len(topics) * len(templates) * 2:
                break
        return results

    def _passes_filter(self, entry: PromptEntry) -> bool:
        if entry.char_length < self.config.min_prompt_length:
            return False
        if entry.char_length > self.config.max_prompt_length:
            return False
        return True

    def _make_id(self, domain: str, index: int) -> str:
        raw = f"{domain}_{index}_{self.config.seed}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def balance_domains(self, prompts: List[PromptEntry]) -> List[PromptEntry]:
        """Downsample domains to the smallest domain count for balance."""
        by_domain: Dict[str, List[PromptEntry]] = {}
        for p in prompts:
            by_domain.setdefault(p.domain, []).append(p)
        min_count = min(len(v) for v in by_domain.values()) if by_domain else 0
        balanced: List[PromptEntry] = []
        for domain, entries in by_domain.items():
            self.rng.shuffle(entries)
            balanced.extend(entries[:min_count])
        return balanced


# ---------------------------------------------------------------------------
# OutputGenerator — run algorithms and store results
# ---------------------------------------------------------------------------

class OutputGenerator:
    """Generates (simulated) outputs from multiple algorithm configurations."""

    def __init__(self, config: DatasetConfig, rng: Optional[np.random.RandomState] = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.RandomState(config.seed)
        self._vocab = self._build_vocab()

    def _build_vocab(self) -> List[str]:
        base_words = [
            "the", "a", "is", "was", "are", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall",
            "should", "may", "might", "can", "could", "must", "need",
            "in", "on", "at", "to", "for", "with", "by", "from", "of",
            "and", "but", "or", "nor", "not", "so", "yet", "both",
            "this", "that", "these", "those", "it", "they", "we", "he", "she",
            "data", "model", "system", "process", "method", "approach",
            "result", "analysis", "function", "structure", "algorithm",
            "diversity", "output", "input", "score", "quality", "measure",
            "generate", "compute", "evaluate", "compare", "select", "optimize",
            "creative", "novel", "unique", "different", "varied", "distinct",
        ]
        return base_words

    def generate_for_prompt(self, prompt: PromptEntry) -> List[GeneratedOutput]:
        """Generate simulated outputs for a prompt across all algorithm/model combos."""
        outputs: List[GeneratedOutput] = []
        for algorithm in self.config.algorithms:
            for model in self.config.models:
                text = self._simulate_output(prompt, algorithm, model)
                oid = self._make_output_id(prompt.prompt_id, algorithm, model)
                tokens = len(text.split())
                gen_time = self.rng.exponential(50.0)
                logprob = -self.rng.exponential(2.0) * tokens
                out = GeneratedOutput(
                    output_id=oid,
                    prompt_id=prompt.prompt_id,
                    text=text,
                    algorithm=algorithm,
                    model=model,
                    tokens=tokens,
                    generation_time_ms=round(gen_time, 2),
                    logprob_sum=round(logprob, 4),
                )
                outputs.append(out)
        return outputs

    def _simulate_output(self, prompt: PromptEntry, algorithm: str, model: str) -> str:
        seed_val = hash((prompt.prompt_id, algorithm, model)) % (2 ** 31)
        local_rng = np.random.RandomState(seed_val)
        diversity_factor = self._algorithm_diversity_factor(algorithm)
        length = max(20, int(local_rng.normal(80, 20 * diversity_factor)))
        length = min(length, self.config.max_output_tokens)
        vocab_size = max(10, int(len(self._vocab) * diversity_factor))
        active_vocab = self._vocab[:vocab_size]
        words = [active_vocab[i] for i in local_rng.randint(0, len(active_vocab), size=length)]
        # Insert sentence breaks
        for i in range(5, length, local_rng.randint(8, 20)):
            words[i] = words[i] + "."
        return " ".join(words)

    def _algorithm_diversity_factor(self, algorithm: str) -> float:
        factors = {
            "greedy": 0.3,
            "temperature_0.7": 0.6,
            "temperature_1.0": 0.85,
            "top_k_50": 0.7,
            "top_p_0.9": 0.75,
            "typical_p_0.95": 0.72,
            "beam_search_5": 0.4,
            "diverse_beam_5": 0.65,
            "mirostat_v2": 0.78,
            "contrastive_search": 0.68,
        }
        return factors.get(algorithm, 0.5)

    def _make_output_id(self, prompt_id: str, algorithm: str, model: str) -> str:
        raw = f"{prompt_id}_{algorithm}_{model}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# JudgmentSimulator — create simulated human judgments
# ---------------------------------------------------------------------------

class JudgmentSimulator:
    """Simulates human diversity judgments using heuristic scoring functions."""

    def __init__(self, config: DatasetConfig, rng: Optional[np.random.RandomState] = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.RandomState(config.seed)

    def judge_outputs(self, prompt: PromptEntry, outputs: List[GeneratedOutput]) -> List[DiversityJudgment]:
        """Create judgments for groups of outputs sharing the same algorithm+model."""
        groups: Dict[Tuple[str, str], List[GeneratedOutput]] = {}
        for out in outputs:
            key = (out.algorithm, out.model)
            groups.setdefault(key, []).append(out)

        judgments: List[DiversityJudgment] = []
        for (algo, model), group in groups.items():
            for judge_idx in range(self.config.num_judges):
                scores = self._compute_heuristic_scores(group)
                noise = {dim: float(self.rng.normal(0, 0.05)) for dim in scores}
                noisy_scores = {dim: np.clip(scores[dim] + noise[dim], 0.0, 1.0) for dim in scores}
                jid = self._make_judgment_id(prompt.prompt_id, algo, model, judge_idx)
                j = DiversityJudgment(
                    judgment_id=jid,
                    prompt_id=prompt.prompt_id,
                    output_ids=[o.output_id for o in group],
                    algorithm=algo,
                    model=model,
                    scores={k: round(float(v), 4) for k, v in noisy_scores.items()},
                    judge_id=f"sim_judge_{judge_idx}",
                    judge_type="heuristic",
                    confidence=round(float(self.rng.uniform(0.6, 1.0)), 3),
                )
                judgments.append(j)
        return judgments

    def _compute_heuristic_scores(self, outputs: List[GeneratedOutput]) -> Dict[str, float]:
        texts = [o.text for o in outputs]
        lexical = self._lexical_diversity(texts)
        semantic = self._semantic_diversity_proxy(texts)
        structural = self._structural_diversity(texts)
        coherence = self._coherence_proxy(texts)
        overall = 0.3 * lexical + 0.25 * semantic + 0.2 * structural + 0.25 * coherence
        return {
            "lexical_diversity": lexical,
            "semantic_diversity": semantic,
            "structural_diversity": structural,
            "coherence": coherence,
            "overall_quality": overall,
        }

    def _lexical_diversity(self, texts: List[str]) -> float:
        if not texts:
            return 0.0
        all_tokens: List[Set[str]] = [set(t.lower().split()) for t in texts]
        if len(all_tokens) < 2:
            return 0.5
        union = set().union(*all_tokens)
        intersection = all_tokens[0].intersection(*all_tokens[1:])
        if not union:
            return 0.0
        jaccard_dist = 1.0 - len(intersection) / len(union)
        ttr_values = []
        for t in texts:
            words = t.lower().split()
            ttr_values.append(len(set(words)) / max(len(words), 1))
        mean_ttr = float(np.mean(ttr_values))
        return float(np.clip(0.5 * jaccard_dist + 0.5 * mean_ttr, 0.0, 1.0))

    def _semantic_diversity_proxy(self, texts: List[str]) -> float:
        """Approximate semantic diversity via n-gram overlap variance."""
        if len(texts) < 2:
            return 0.5
        bigram_sets = []
        for t in texts:
            words = t.lower().split()
            bigrams = set(zip(words[:-1], words[1:]))
            bigram_sets.append(bigrams)
        overlaps = []
        for i in range(len(bigram_sets)):
            for j in range(i + 1, len(bigram_sets)):
                union = bigram_sets[i] | bigram_sets[j]
                inter = bigram_sets[i] & bigram_sets[j]
                if union:
                    overlaps.append(1.0 - len(inter) / len(union))
                else:
                    overlaps.append(0.5)
        return float(np.clip(np.mean(overlaps), 0.0, 1.0))

    def _structural_diversity(self, texts: List[str]) -> float:
        """Measure structural diversity via sentence-length variance."""
        if not texts:
            return 0.0
        lengths = [len(t.split()) for t in texts]
        sentence_counts = [max(t.count("."), 1) for t in texts]
        length_cv = float(np.std(lengths) / max(np.mean(lengths), 1))
        sent_cv = float(np.std(sentence_counts) / max(np.mean(sentence_counts), 1))
        return float(np.clip(0.5 * length_cv + 0.5 * sent_cv, 0.0, 1.0))

    def _coherence_proxy(self, texts: List[str]) -> float:
        """Proxy for coherence: ratio of repeated word transitions."""
        scores = []
        for t in texts:
            words = t.lower().split()
            if len(words) < 2:
                scores.append(0.5)
                continue
            transitions = sum(1 for a, b in zip(words[:-1], words[1:]) if a != b)
            scores.append(transitions / (len(words) - 1))
        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def _make_judgment_id(self, prompt_id: str, algo: str, model: str, judge: int) -> str:
        raw = f"j_{prompt_id}_{algo}_{model}_{judge}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# DatasetAnalyzer — compute comprehensive statistics
# ---------------------------------------------------------------------------

class DatasetAnalyzer:
    """Computes comprehensive statistics over the dataset."""

    def __init__(self, prompts: List[PromptEntry], outputs: List[GeneratedOutput],
                 judgments: List[DiversityJudgment]):
        self.prompts = prompts
        self.outputs = outputs
        self.judgments = judgments

    def full_report(self) -> Dict[str, Any]:
        return {
            "overview": self._overview(),
            "domain_stats": self._domain_stats(),
            "algorithm_stats": self._algorithm_stats(),
            "judgment_stats": self._judgment_stats(),
            "quality_flags": self._quality_flags(),
        }

    def _overview(self) -> Dict[str, Any]:
        return {
            "num_prompts": len(self.prompts),
            "num_outputs": len(self.outputs),
            "num_judgments": len(self.judgments),
            "domains": list({p.domain for p in self.prompts}),
            "algorithms": list({o.algorithm for o in self.outputs}),
            "models": list({o.model for o in self.outputs}),
        }

    def _domain_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        for domain in {p.domain for p in self.prompts}:
            domain_prompts = [p for p in self.prompts if p.domain == domain]
            domain_outputs = [o for o in self.outputs
                              if any(p.prompt_id == o.prompt_id for p in domain_prompts)]
            lengths = [p.word_count for p in domain_prompts]
            stats[domain] = {
                "num_prompts": len(domain_prompts),
                "num_outputs": len(domain_outputs),
                "avg_prompt_words": round(float(np.mean(lengths)), 1) if lengths else 0,
                "difficulty_distribution": self._count_values([p.difficulty for p in domain_prompts]),
            }
        return stats

    def _algorithm_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        for algo in {o.algorithm for o in self.outputs}:
            algo_outputs = [o for o in self.outputs if o.algorithm == algo]
            tokens = [o.tokens for o in algo_outputs]
            times = [o.generation_time_ms for o in algo_outputs]
            algo_judgments = [j for j in self.judgments if j.algorithm == algo]
            overall_scores = [j.overall_score for j in algo_judgments]
            stats[algo] = {
                "num_outputs": len(algo_outputs),
                "avg_tokens": round(float(np.mean(tokens)), 1) if tokens else 0,
                "std_tokens": round(float(np.std(tokens)), 1) if tokens else 0,
                "avg_gen_time_ms": round(float(np.mean(times)), 2) if times else 0,
                "avg_diversity_score": round(float(np.mean(overall_scores)), 4) if overall_scores else 0,
                "std_diversity_score": round(float(np.std(overall_scores)), 4) if overall_scores else 0,
            }
        return stats

    def _judgment_stats(self) -> Dict[str, Any]:
        if not self.judgments:
            return {}
        all_scores = [j.overall_score for j in self.judgments]
        dimension_scores: Dict[str, List[float]] = {}
        for j in self.judgments:
            for dim, val in j.scores.items():
                dimension_scores.setdefault(dim, []).append(val)
        dim_stats = {}
        for dim, vals in dimension_scores.items():
            dim_stats[dim] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }
        return {
            "total_judgments": len(self.judgments),
            "overall_mean": round(float(np.mean(all_scores)), 4),
            "overall_std": round(float(np.std(all_scores)), 4),
            "dimension_stats": dim_stats,
        }

    def _quality_flags(self) -> List[str]:
        flags = []
        if len(self.prompts) < 10:
            flags.append("LOW_PROMPT_COUNT")
        domains = {p.domain for p in self.prompts}
        if len(domains) < len(DOMAINS):
            flags.append("MISSING_DOMAINS")
        domain_counts = {}
        for p in self.prompts:
            domain_counts[p.domain] = domain_counts.get(p.domain, 0) + 1
        if domain_counts:
            ratio = max(domain_counts.values()) / max(min(domain_counts.values()), 1)
            if ratio > 2.0:
                flags.append("DOMAIN_IMBALANCE")
        empty_outputs = [o for o in self.outputs if len(o.text.strip()) == 0]
        if empty_outputs:
            flags.append(f"EMPTY_OUTPUTS:{len(empty_outputs)}")
        if not self.judgments:
            flags.append("NO_JUDGMENTS")
        return flags

    @staticmethod
    def _count_values(items: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# DiversityDecodingDataset — main dataset class
# ---------------------------------------------------------------------------

class DiversityDecodingDataset:
    """Main dataset class for the Diversity Decoding Arena benchmark."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.rng = np.random.RandomState(self.config.seed)
        self.prompts: List[PromptEntry] = []
        self.outputs: List[GeneratedOutput] = []
        self.judgments: List[DiversityJudgment] = []
        self.splits: Dict[str, DatasetSplit] = {}
        self.version = DatasetVersion(self.config.version)
        self._prompt_index: Dict[str, PromptEntry] = {}
        self._output_index: Dict[str, GeneratedOutput] = {}
        self._cache_dir = Path(self.config.cache_dir)
        self._output_dir = Path(self.config.output_dir)

    # -- Build pipeline -------------------------------------------------------

    def build_prompt_collection(self) -> List[PromptEntry]:
        """Curate prompts across all configured domains."""
        curator = PromptCurator(self.config, rng=self.rng)
        self.prompts = curator.curate()
        self.prompts = curator.balance_domains(self.prompts)
        self._rebuild_prompt_index()
        logger.info("Built prompt collection: %d prompts across %d domains",
                     len(self.prompts), len({p.domain for p in self.prompts}))
        return self.prompts

    def generate_outputs(self) -> List[GeneratedOutput]:
        """Generate outputs for every prompt using all algorithm/model combos."""
        generator = OutputGenerator(self.config, rng=self.rng)
        self.outputs = []
        for prompt in self.prompts:
            prompt_outputs = generator.generate_for_prompt(prompt)
            self.outputs.extend(prompt_outputs)
        self._rebuild_output_index()
        logger.info("Generated %d outputs for %d prompts", len(self.outputs), len(self.prompts))
        return self.outputs

    def simulate_diversity_judgments(self) -> List[DiversityJudgment]:
        """Create simulated human diversity judgments for all prompt/output groups."""
        simulator = JudgmentSimulator(self.config, rng=self.rng)
        self.judgments = []
        for prompt in self.prompts:
            prompt_outputs = [o for o in self.outputs if o.prompt_id == prompt.prompt_id]
            if prompt_outputs:
                prompt_judgments = simulator.judge_outputs(prompt, prompt_outputs)
                self.judgments.extend(prompt_judgments)
        logger.info("Simulated %d diversity judgments", len(self.judgments))
        return self.judgments

    def compute_dataset_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive statistics about the dataset."""
        analyzer = DatasetAnalyzer(self.prompts, self.outputs, self.judgments)
        return analyzer.full_report()

    def validate_dataset(self) -> Tuple[bool, List[str]]:
        """Run quality checks and return (is_valid, list_of_issues)."""
        issues: List[str] = []
        # Check prompt IDs are unique
        prompt_ids = [p.prompt_id for p in self.prompts]
        if len(prompt_ids) != len(set(prompt_ids)):
            issues.append("Duplicate prompt IDs detected")
        # Check output IDs are unique
        output_ids = [o.output_id for o in self.outputs]
        if len(output_ids) != len(set(output_ids)):
            issues.append("Duplicate output IDs detected")
        # Check all outputs reference valid prompts
        valid_prompt_ids = set(prompt_ids)
        orphan_outputs = [o for o in self.outputs if o.prompt_id not in valid_prompt_ids]
        if orphan_outputs:
            issues.append(f"{len(orphan_outputs)} outputs reference non-existent prompts")
        # Check all judgments reference valid prompts
        orphan_judgments = [j for j in self.judgments if j.prompt_id not in valid_prompt_ids]
        if orphan_judgments:
            issues.append(f"{len(orphan_judgments)} judgments reference non-existent prompts")
        # Check judgment output_ids reference valid outputs
        valid_output_ids = set(output_ids)
        for j in self.judgments:
            invalid = [oid for oid in j.output_ids if oid not in valid_output_ids]
            if invalid:
                issues.append(f"Judgment {j.judgment_id} references {len(invalid)} invalid output IDs")
                break
        # Check domain coverage
        covered_domains = {p.domain for p in self.prompts}
        missing = set(self.config.domains) - covered_domains
        if missing:
            issues.append(f"Missing domains: {missing}")
        # Check empty outputs
        empty = sum(1 for o in self.outputs if not o.text.strip())
        if empty > 0:
            issues.append(f"{empty} empty outputs found")
        # Check score ranges
        for j in self.judgments:
            for dim, score in j.scores.items():
                if not (0.0 <= score <= 1.0):
                    issues.append(f"Score out of range in judgment {j.judgment_id}: {dim}={score}")
                    break
            else:
                continue
            break

        is_valid = len(issues) == 0
        return is_valid, issues

    # -- Splitting ------------------------------------------------------------

    def split(self) -> Dict[str, DatasetSplit]:
        """Create stratified train/val/test splits."""
        by_domain: Dict[str, List[str]] = {}
        for p in self.prompts:
            by_domain.setdefault(p.domain, []).append(p.prompt_id)

        train_ids, val_ids, test_ids = [], [], []
        for domain, ids in by_domain.items():
            shuffled = list(ids)
            self.rng.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(n * self.config.train_ratio)
            n_val = int(n * self.config.val_ratio)
            train_ids.extend(shuffled[:n_train])
            val_ids.extend(shuffled[n_train:n_train + n_val])
            test_ids.extend(shuffled[n_train + n_val:])

        self.splits = {
            "train": self._make_split("train", train_ids),
            "val": self._make_split("val", val_ids),
            "test": self._make_split("test", test_ids),
        }
        logger.info("Split dataset: train=%d, val=%d, test=%d",
                     len(train_ids), len(val_ids), len(test_ids))
        return self.splits

    def _make_split(self, name: str, prompt_ids: List[str]) -> DatasetSplit:
        domain_dist: Dict[str, int] = {}
        for pid in prompt_ids:
            p = self._prompt_index.get(pid)
            if p:
                domain_dist[p.domain] = domain_dist.get(p.domain, 0) + 1
        return DatasetSplit(name=name, prompt_ids=prompt_ids, domain_distribution=domain_dist)

    # -- Export ---------------------------------------------------------------

    def export_standard_format(self, path: Optional[str] = None) -> str:
        """Export dataset to standardized JSON format with versioning."""
        out_path = Path(path) if path else self._output_dir / "dataset.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_version": self.version.version_string,
            "config": self.config.to_dict(),
            "fingerprint": self.config.fingerprint(),
            "statistics": self.compute_dataset_statistics(),
            "prompts": [p.to_dict() for p in self.prompts],
            "outputs": [o.to_dict() for o in self.outputs],
            "judgments": [j.to_dict() for j in self.judgments],
            "splits": {k: v.to_dict() for k, v in self.splits.items()},
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Exported dataset to %s", out_path)
        return str(out_path)

    def export_huggingface_format(self, path: Optional[str] = None) -> str:
        """Export to HuggingFace Datasets-compatible JSONL + metadata."""
        out_dir = Path(path) if path else self._output_dir / "huggingface"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write dataset card
        card = {
            "dataset_info": {
                "name": self.config.name,
                "version": self.version.version_string,
                "description": self.config.description,
                "features": {
                    "prompt_id": "string",
                    "prompt_text": "string",
                    "domain": "string",
                    "outputs": "list",
                    "judgments": "list",
                },
                "splits": {
                    name: {"num_examples": split.size}
                    for name, split in self.splits.items()
                },
            }
        }
        with open(out_dir / "dataset_info.json", "w") as f:
            json.dump(card, f, indent=2)

        # Write split files as JSONL
        for split_name, split_data in self.splits.items():
            split_path = out_dir / f"{split_name}.jsonl"
            pid_set = set(split_data.prompt_ids)
            with open(split_path, "w") as f:
                for prompt in self.prompts:
                    if prompt.prompt_id not in pid_set:
                        continue
                    prompt_outputs = [o.to_dict() for o in self.outputs
                                      if o.prompt_id == prompt.prompt_id]
                    prompt_judgments = [j.to_dict() for j in self.judgments
                                       if j.prompt_id == prompt.prompt_id]
                    row = {
                        "prompt_id": prompt.prompt_id,
                        "prompt_text": prompt.text,
                        "domain": prompt.domain,
                        "subdomain": prompt.subdomain,
                        "difficulty": prompt.difficulty,
                        "outputs": prompt_outputs,
                        "judgments": prompt_judgments,
                    }
                    f.write(json.dumps(row) + "\n")
        logger.info("Exported HuggingFace format to %s", out_dir)
        return str(out_dir)

    # -- Persistence ----------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """Save the full dataset state to disk."""
        out_path = Path(path) if path else self._cache_dir / "dataset_state.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config": self.config.to_dict(),
            "version": self.version.version_string,
            "prompts": [p.to_dict() for p in self.prompts],
            "outputs": [o.to_dict() for o in self.outputs],
            "judgments": [j.to_dict() for j in self.judgments],
            "splits": {k: v.to_dict() for k, v in self.splits.items()},
        }
        with open(out_path, "w") as f:
            json.dump(state, f)
        logger.info("Saved dataset state to %s (%d bytes)", out_path, out_path.stat().st_size)
        return str(out_path)

    @classmethod
    def load(cls, path: str) -> "DiversityDecodingDataset":
        """Load a dataset from a saved state file."""
        with open(path) as f:
            state = json.load(f)
        config = DatasetConfig.from_dict(state["config"])
        ds = cls(config)
        ds.version = DatasetVersion(state.get("version", "1.0.0"))
        ds.prompts = [PromptEntry.from_dict(p) for p in state.get("prompts", [])]
        ds.outputs = [GeneratedOutput.from_dict(o) for o in state.get("outputs", [])]
        ds.judgments = [DiversityJudgment.from_dict(j) for j in state.get("judgments", [])]
        for split_name, split_data in state.get("splits", {}).items():
            ds.splits[split_name] = DatasetSplit(**split_data)
        ds._rebuild_prompt_index()
        ds._rebuild_output_index()
        logger.info("Loaded dataset from %s: %d prompts, %d outputs, %d judgments",
                     path, len(ds.prompts), len(ds.outputs), len(ds.judgments))
        return ds

    # -- Helpers --------------------------------------------------------------

    def _rebuild_prompt_index(self) -> None:
        self._prompt_index = {p.prompt_id: p for p in self.prompts}

    def _rebuild_output_index(self) -> None:
        self._output_index = {o.output_id: o for o in self.outputs}

    def get_prompt(self, prompt_id: str) -> Optional[PromptEntry]:
        return self._prompt_index.get(prompt_id)

    def get_outputs_for_prompt(self, prompt_id: str) -> List[GeneratedOutput]:
        return [o for o in self.outputs if o.prompt_id == prompt_id]

    def get_judgments_for_prompt(self, prompt_id: str) -> List[DiversityJudgment]:
        return [j for j in self.judgments if j.prompt_id == prompt_id]

    def get_outputs_by_algorithm(self, algorithm: str) -> List[GeneratedOutput]:
        return [o for o in self.outputs if o.algorithm == algorithm]

    def get_judgments_by_algorithm(self, algorithm: str) -> List[DiversityJudgment]:
        return [j for j in self.judgments if j.algorithm == algorithm]

    def filter_by_domain(self, domain: str) -> "DiversityDecodingDataset":
        """Return a new dataset filtered to a single domain."""
        new_config = copy.deepcopy(self.config)
        new_config.domains = [domain]
        new_ds = DiversityDecodingDataset(new_config)
        new_ds.prompts = [p for p in self.prompts if p.domain == domain]
        pid_set = {p.prompt_id for p in new_ds.prompts}
        new_ds.outputs = [o for o in self.outputs if o.prompt_id in pid_set]
        new_ds.judgments = [j for j in self.judgments if j.prompt_id in pid_set]
        new_ds._rebuild_prompt_index()
        new_ds._rebuild_output_index()
        return new_ds

    def sample(self, n: int) -> "DiversityDecodingDataset":
        """Return a new dataset with n randomly sampled prompts."""
        new_ds = DiversityDecodingDataset(copy.deepcopy(self.config))
        indices = self.rng.choice(len(self.prompts), size=min(n, len(self.prompts)), replace=False)
        new_ds.prompts = [self.prompts[i] for i in indices]
        pid_set = {p.prompt_id for p in new_ds.prompts}
        new_ds.outputs = [o for o in self.outputs if o.prompt_id in pid_set]
        new_ds.judgments = [j for j in self.judgments if j.prompt_id in pid_set]
        new_ds._rebuild_prompt_index()
        new_ds._rebuild_output_index()
        return new_ds

    def __len__(self) -> int:
        return len(self.prompts)

    def __repr__(self) -> str:
        return (
            f"DiversityDecodingDataset(v{self.version.version_string}, "
            f"prompts={len(self.prompts)}, outputs={len(self.outputs)}, "
            f"judgments={len(self.judgments)})"
        )
