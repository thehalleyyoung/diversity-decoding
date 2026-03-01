"""
Failure mode taxonomy for adversarial metric divergences.

Addresses reviewer critique: "No failure mode taxonomy classifying
adversarial findings into categories (degenerate inputs, boundary
cases, dimensionality artifacts)."

Implements:
  1. Systematic classification of metric divergence failure modes
  2. Symbolic characterization of what structural properties cause disagreement
  3. Adversarial input construction for each failure mode
  4. Severity scoring and mitigation recommendations
  5. Failure mode detection on new inputs
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Failure mode classification
# ---------------------------------------------------------------------------


class FailureCategory(Enum):
    """Top-level failure mode categories."""
    DEGENERATE_INPUT = auto()
    BOUNDARY_CASE = auto()
    DIMENSIONALITY_ARTIFACT = auto()
    SCALE_SENSITIVITY = auto()
    DISTRIBUTION_MISMATCH = auto()
    METRIC_DEFINITION_GAP = auto()


class FailureSeverity(Enum):
    """Severity levels for failure modes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FailureMode:
    """A characterized failure mode with structural explanation."""
    id: str
    name: str
    category: FailureCategory
    severity: FailureSeverity
    description: str
    structural_cause: str
    affected_metrics: List[str]
    detection_criterion: str
    example_texts: Optional[List[str]] = None
    metric_values: Optional[Dict[str, float]] = None
    tau_normal: Optional[float] = None
    tau_adversarial: Optional[float] = None
    mitigation: str = ""


@dataclass
class FailureModeTaxonomy:
    """Complete taxonomy of failure modes."""
    modes: List[FailureMode]
    category_counts: Dict[str, int]
    severity_distribution: Dict[str, int]
    total_modes: int
    coverage_metrics: Dict[str, int]  # metrics → # failure modes affecting them
    summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# Tokenization and metrics
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _distinct_n(texts: List[str], n: int = 2) -> float:
    all_ng = []
    for t in texts:
        tokens = _tokenize(t)
        for i in range(len(tokens) - n + 1):
            all_ng.append(tuple(tokens[i:i + n]))
    if not all_ng:
        return 0.0
    return len(set(all_ng)) / len(all_ng)


def _self_bleu_approx(texts: List[str], n: int = 4) -> float:
    if len(texts) < 2:
        return 0.0
    tokenized = [_tokenize(t) for t in texts]
    scores = []
    for i in range(len(texts)):
        ref_ng = Counter(tuple(tokenized[i][j:j + n])
                         for j in range(len(tokenized[i]) - n + 1))
        if not ref_ng:
            continue
        for j in range(len(texts)):
            if i == j:
                continue
            hyp_ng = Counter(tuple(tokenized[j][l:l + n])
                             for l in range(len(tokenized[j]) - n + 1))
            if not hyp_ng:
                scores.append(0.0)
                continue
            overlap = sum((ref_ng & hyp_ng).values())
            total = sum(hyp_ng.values())
            scores.append(overlap / total if total > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _ttr(texts: List[str]) -> float:
    all_tokens = []
    for t in texts:
        all_tokens.extend(_tokenize(t))
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)


def _ngram_entropy(texts: List[str], n: int = 2) -> float:
    counts = Counter()
    for t in texts:
        tokens = _tokenize(t)
        for i in range(len(tokens) - n + 1):
            counts[tuple(tokens[i:i + n])] += 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


METRIC_FNS = {
    "D-2": lambda texts: _distinct_n(texts, 2),
    "Self-BLEU": _self_bleu_approx,
    "TTR": _ttr,
    "Entropy": _ngram_entropy,
}


def _compute_metrics(texts: List[str]) -> Dict[str, float]:
    result = {}
    for name, fn in METRIC_FNS.items():
        try:
            result[name] = fn(texts)
        except Exception:
            result[name] = float("nan")
    return result


def _kendall_tau(x, y):
    if len(x) < 3:
        return 0.0
    tau, _ = stats.kendalltau(x, y)
    return tau if not np.isnan(tau) else 0.0


# ---------------------------------------------------------------------------
# Failure mode generators
# ---------------------------------------------------------------------------


class FailureModeGenerator:
    """Generates adversarial inputs for each failure mode category."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_degenerate_inputs(self) -> List[FailureMode]:
        """Category 1: Degenerate inputs that cause metric breakdown."""
        modes = []

        # FM1: Empty/near-empty texts
        empty_texts = ["", " ", "a", "the"]
        modes.append(FailureMode(
            id="FM-DEG-001",
            name="Empty/trivial texts",
            category=FailureCategory.DEGENERATE_INPUT,
            severity=FailureSeverity.HIGH,
            description="Empty or single-token texts cause division by zero or undefined behavior in n-gram metrics.",
            structural_cause="N-gram metrics require ≥n tokens per text; with fewer, the denominator is 0.",
            affected_metrics=["D-2", "Self-BLEU", "Entropy"],
            detection_criterion="Any text has fewer than n tokens",
            example_texts=empty_texts,
            metric_values=_compute_metrics(empty_texts),
            mitigation="Filter texts with < n tokens before computing metrics. Return NaN with warning.",
        ))

        # FM2: Identical texts (zero diversity)
        identical = ["the cat sat on the mat"] * 10
        modes.append(FailureMode(
            id="FM-DEG-002",
            name="Identical texts",
            category=FailureCategory.DEGENERATE_INPUT,
            severity=FailureSeverity.MEDIUM,
            description="All texts identical. D-2 may still be non-zero (from within-text diversity) while Self-BLEU is 1.0.",
            structural_cause="D-2 measures type-token ratio which can be >0 for a single long text, while Self-BLEU correctly identifies zero diversity between texts.",
            affected_metrics=["D-2", "Self-BLEU"],
            detection_criterion="All texts identical after normalization",
            example_texts=identical,
            metric_values=_compute_metrics(identical),
            tau_normal=None,
            tau_adversarial=None,
            mitigation="Check for exact/near-exact duplicates before computing metrics. Flag degenerate case.",
        ))

        # FM3: Single unique word repeated
        single_word = ["hello " * 20] * 5 + ["world " * 20] * 5
        modes.append(FailureMode(
            id="FM-DEG-003",
            name="Extreme repetition",
            category=FailureCategory.DEGENERATE_INPUT,
            severity=FailureSeverity.HIGH,
            description="Texts consisting of a single repeated word. TTR approaches 1/n_tokens while D-2 is near 0.",
            structural_cause="TTR denominator grows linearly with text length but numerator is bounded by vocabulary size.",
            affected_metrics=["TTR", "D-2", "Entropy"],
            detection_criterion="TTR < 0.05 or single unique token dominates >90% of text",
            example_texts=single_word,
            metric_values=_compute_metrics(single_word),
            mitigation="Use bias-corrected TTR (MSTTR) or entropy-based measures for highly repetitive text.",
        ))

        return modes

    def generate_boundary_cases(self) -> List[FailureMode]:
        """Category 2: Boundary cases where metrics disagree."""
        modes = []

        # FM4: High within-text diversity, low between-text diversity
        diverse_within = [
            "the quick brown fox jumps over the lazy dog near the river",
            "the quick brown fox jumps over the lazy dog near the mountain",
            "the quick brown fox jumps over the lazy dog near the ocean",
            "the quick brown fox jumps over the lazy dog near the forest",
            "the quick brown fox jumps over the lazy dog near the desert",
        ]
        modes.append(FailureMode(
            id="FM-BND-001",
            name="High within-text, low between-text diversity",
            category=FailureCategory.BOUNDARY_CASE,
            severity=FailureSeverity.MEDIUM,
            description="Texts have rich internal vocabulary but are near-identical to each other. D-2 (computed across all texts) may show high diversity while Self-BLEU correctly shows low diversity.",
            structural_cause="D-2 pools n-grams across all texts, mixing within-text and between-text diversity. Self-BLEU explicitly compares pairs.",
            affected_metrics=["D-2", "Self-BLEU"],
            detection_criterion="|D-2 rank - Self-BLEU rank| > 50th percentile for normal inputs",
            example_texts=diverse_within,
            metric_values=_compute_metrics(diverse_within),
            mitigation="Report both within-text and between-text diversity separately.",
        ))

        # FM5: Texts with very different lengths
        length_var = [
            "hello",
            "the cat sat on the mat and looked at the birds flying in the sky above the tall green trees swaying in the wind while the sun was setting behind the distant mountains casting long shadows across the valley below",
        ] * 5
        modes.append(FailureMode(
            id="FM-BND-002",
            name="Extreme length variation",
            category=FailureCategory.BOUNDARY_CASE,
            severity=FailureSeverity.MEDIUM,
            description="Texts vary enormously in length. Long texts dominate n-gram counts, making short texts invisible to pooled metrics.",
            structural_cause="N-gram pooling is count-weighted, so a single long text contributes O(L) n-grams while a short text contributes O(1).",
            affected_metrics=["D-2", "Entropy", "TTR"],
            detection_criterion="max(len)/min(len) > 10",
            example_texts=length_var,
            metric_values=_compute_metrics(length_var),
            mitigation="Normalize by text length or use per-text metrics averaged across texts.",
        ))

        # FM6: Permutation diversity (same words, different order)
        perm_texts = [
            "the cat sat on the mat",
            "on the mat sat the cat",
            "sat the cat on the mat",
            "the mat on sat cat the",
            "cat the sat mat on the",
        ]
        modes.append(FailureMode(
            id="FM-BND-003",
            name="Permutation-only diversity",
            category=FailureCategory.BOUNDARY_CASE,
            severity=FailureSeverity.LOW,
            description="Texts use identical vocabulary but in different orders. TTR shows zero diversity while D-2 and Self-BLEU capture n-gram differences.",
            structural_cause="TTR is a bag-of-words metric (order-invariant), while n-gram metrics are order-sensitive.",
            affected_metrics=["TTR", "D-2"],
            detection_criterion="TTR variation < 0.01 but D-2 variation > 0.1",
            example_texts=perm_texts,
            metric_values=_compute_metrics(perm_texts),
            mitigation="Use n-gram metrics when word order matters; report TTR alongside for vocabulary diversity.",
        ))

        return modes

    def generate_dimensionality_artifacts(self) -> List[FailureMode]:
        """Category 3: Artifacts from metric dimensionality assumptions."""
        modes = []

        # FM7: Vocabulary explosion (many hapax legomena)
        rng = self.rng
        hapax_texts = []
        base_words = ["algorithm", "computer", "network", "system", "data"]
        for i in range(10):
            # Mix common words with unique suffixes
            words = base_words.copy()
            for _ in range(15):
                words.append(f"word{rng.randint(0, 10000)}")
            rng.shuffle(words)
            hapax_texts.append(" ".join(words))

        modes.append(FailureMode(
            id="FM-DIM-001",
            name="Vocabulary explosion (hapax legomena)",
            category=FailureCategory.DIMENSIONALITY_ARTIFACT,
            severity=FailureSeverity.HIGH,
            description="Many unique tokens appear only once. D-2 and TTR are inflated by rare tokens that don't indicate meaningful diversity.",
            structural_cause="D-2 counts distinct n-grams / total n-grams. With many hapax legomena, the ratio approaches 1 regardless of semantic diversity.",
            affected_metrics=["D-2", "TTR", "Entropy"],
            detection_criterion="Hapax legomenon ratio > 0.5 (>50% of types occur only once)",
            example_texts=hapax_texts,
            metric_values=_compute_metrics(hapax_texts),
            mitigation="Apply frequency thresholding (remove hapax legomena) or use entropy with bias correction.",
        ))

        # FM8: Domain-specific vocabulary creating artificial clusters
        domain_texts_a = [
            "the algorithm optimizes the neural network parameters using gradient descent",
            "deep learning models train on large datasets with backpropagation",
            "convolutional networks extract features from image data automatically",
        ]
        domain_texts_b = [
            "the quarterly revenue exceeded analyst expectations by fifteen percent",
            "shareholders approved the merger during the annual general meeting",
            "market capitalization grew following the successful product launch",
        ]
        mixed = domain_texts_a + domain_texts_b
        modes.append(FailureMode(
            id="FM-DIM-002",
            name="Domain vocabulary clustering",
            category=FailureCategory.DIMENSIONALITY_ARTIFACT,
            severity=FailureSeverity.MEDIUM,
            description="Texts from different domains create vocabulary clusters. TTR and D-2 show high diversity due to disjoint vocabularies, not meaningful text variation.",
            structural_cause="Domain-specific jargon creates non-overlapping vocabulary clusters that inflate type-token and n-gram metrics.",
            affected_metrics=["D-2", "TTR", "Jaccard"],
            detection_criterion="Vocabulary Jaccard similarity between text subsets < 0.2",
            example_texts=mixed,
            metric_values=_compute_metrics(mixed),
            mitigation="Compute diversity within and between domains separately. Use embedding-based metrics for semantic diversity.",
        ))

        return modes

    def generate_scale_sensitivity(self) -> List[FailureMode]:
        """Category 4: Sensitivity to sample size and scale."""
        modes = []

        # FM9: Small sample size
        small_sample = [
            "the cat sat on the mat",
            "the dog ran in the park",
        ]
        modes.append(FailureMode(
            id="FM-SCL-001",
            name="Small sample instability",
            category=FailureCategory.SCALE_SENSITIVITY,
            severity=FailureSeverity.HIGH,
            description="With very few texts (n<5), metric estimates are unstable and bootstrap CIs are unreliable.",
            structural_cause="Shannon entropy bias is O(m/2N) where m is vocabulary size and N is sample size. For small N, this bias dominates.",
            affected_metrics=["Entropy", "D-2", "Self-BLEU"],
            detection_criterion="n_texts < 5 or n_total_tokens < 100",
            example_texts=small_sample,
            metric_values=_compute_metrics(small_sample),
            mitigation="Require minimum sample size. Use bias-corrected entropy (Miller-Madow or NSB). Report sample size alongside metrics.",
        ))

        # FM10: Metric saturation at large n
        large_sample = []
        words = ["the", "a", "in", "on", "at", "to", "for", "of", "with", "and",
                 "but", "or", "not", "is", "are", "was", "were", "be", "been",
                 "have", "has", "had", "do", "does", "did", "will", "would"]
        rng = self.rng
        for _ in range(50):
            length = rng.randint(10, 30)
            text_words = [words[rng.randint(0, len(words))] for _ in range(length)]
            large_sample.append(" ".join(text_words))

        modes.append(FailureMode(
            id="FM-SCL-002",
            name="Metric saturation",
            category=FailureCategory.SCALE_SENSITIVITY,
            severity=FailureSeverity.LOW,
            description="At large sample sizes with fixed vocabulary, D-2 saturates near its maximum. Adding more texts provides diminishing information.",
            structural_cause="D-2 = distinct_ngrams / total_ngrams. With fixed vocabulary, distinct_ngrams is bounded while total_ngrams grows linearly.",
            affected_metrics=["D-2", "TTR"],
            detection_criterion="D-2 change < 0.01 when adding 10% more texts",
            example_texts=large_sample[:10],
            metric_values=_compute_metrics(large_sample[:10]),
            mitigation="Report sample-size-adjusted metrics. Use entropy (scales logarithmically) instead of ratio-based metrics at scale.",
        ))

        return modes

    def generate_distribution_mismatches(self) -> List[FailureMode]:
        """Category 5: Distribution mismatch between metrics."""
        modes = []

        # FM11: Zipfian vs uniform n-gram distribution
        uniform_texts = []
        rng = self.rng
        vocab_uniform = [f"word{i}" for i in range(100)]
        for _ in range(10):
            length = 20
            words = [vocab_uniform[rng.randint(0, 100)] for _ in range(length)]
            uniform_texts.append(" ".join(words))

        # Zipfian distribution
        zipf_texts = []
        vocab_zipf = [f"word{i}" for i in range(100)]
        zipf_weights = 1.0 / np.arange(1, 101)
        zipf_weights /= zipf_weights.sum()
        for _ in range(10):
            length = 20
            words = [vocab_zipf[rng.choice(100, p=zipf_weights)] for _ in range(length)]
            zipf_texts.append(" ".join(words))

        modes.append(FailureMode(
            id="FM-DST-001",
            name="Uniform vs Zipfian distribution",
            category=FailureCategory.DISTRIBUTION_MISMATCH,
            severity=FailureSeverity.MEDIUM,
            description="Uniform token distribution maximizes entropy but may have lower D-2 than Zipfian distribution due to n-gram diversity patterns.",
            structural_cause="Entropy is maximized by uniform distributions. D-2 depends on n-gram uniqueness, which can be higher when some tokens are rare (creating unique bigrams).",
            affected_metrics=["Entropy", "D-2"],
            detection_criterion="Rank correlation between Entropy and D-2 < 0 for specific configurations",
            example_texts=uniform_texts[:5],
            metric_values=_compute_metrics(uniform_texts),
            mitigation="Report both entropy and D-2. Use KL divergence to quantify distribution shape differences.",
        ))

        return modes

    def generate_metric_definition_gaps(self) -> List[FailureMode]:
        """Category 6: Gaps between metric definitions and diversity intent."""
        modes = []

        # FM12: Syntactically diverse but semantically identical
        semantic_same = [
            "the cat is on the mat",
            "a feline rests upon the rug",
            "one kitty sits atop the carpet",
            "the small cat lies on the floor covering",
            "a domestic cat is positioned on the mat",
        ]
        modes.append(FailureMode(
            id="FM-DEF-001",
            name="Syntactic diversity without semantic diversity",
            category=FailureCategory.METRIC_DEFINITION_GAP,
            severity=FailureSeverity.HIGH,
            description="Texts are paraphrases of the same idea. N-gram metrics show high diversity but semantic content is identical.",
            structural_cause="N-gram metrics operate on surface forms, not meaning. Paraphrases create distinct n-grams while conveying the same information.",
            affected_metrics=["D-2", "TTR", "Entropy"],
            detection_criterion="High D-2/TTR but low semantic embedding diversity (EPD)",
            example_texts=semantic_same,
            metric_values=_compute_metrics(semantic_same),
            mitigation="Always pair surface-level metrics with embedding-based semantic diversity (EPD, Vendi Score).",
        ))

        # FM13: Semantically diverse but syntactically templated
        template_texts = [
            "I think that artificial intelligence will transform healthcare",
            "I think that quantum computing will transform finance",
            "I think that blockchain technology will transform education",
            "I think that augmented reality will transform manufacturing",
            "I think that autonomous vehicles will transform logistics",
        ]
        modes.append(FailureMode(
            id="FM-DEF-002",
            name="Semantic diversity with syntactic template",
            category=FailureCategory.METRIC_DEFINITION_GAP,
            severity=FailureSeverity.MEDIUM,
            description="Texts follow a template with slot-filling. Self-BLEU is high (similar structure) but content varies meaningfully.",
            structural_cause="Self-BLEU measures n-gram overlap between pairs. Templates create high overlap even when filled content differs.",
            affected_metrics=["Self-BLEU", "D-2"],
            detection_criterion="Self-BLEU > 0.5 but slot-fill diversity > 0.8",
            example_texts=template_texts,
            metric_values=_compute_metrics(template_texts),
            mitigation="Use content-word-only metrics or mask template words before computing diversity.",
        ))

        return modes


# ---------------------------------------------------------------------------
# Failure mode detector
# ---------------------------------------------------------------------------


class FailureModeDetector:
    """Detect which failure modes may affect a given text corpus."""

    def __init__(self, taxonomy: FailureModeTaxonomy):
        self.taxonomy = taxonomy

    def detect(self, texts: List[str]) -> List[Tuple[FailureMode, float]]:
        """Detect potential failure modes in texts.

        Returns list of (FailureMode, confidence) tuples.
        """
        detected = []
        tokens_per_text = [_tokenize(t) for t in texts]

        # Check for degenerate inputs
        n_empty = sum(1 for t in tokens_per_text if len(t) < 2)
        if n_empty > 0:
            fm = self._find_mode("FM-DEG-001")
            if fm:
                detected.append((fm, n_empty / len(texts)))

        # Check for identical texts
        unique_texts = set(t.strip().lower() for t in texts)
        if len(unique_texts) < len(texts) * 0.5:
            fm = self._find_mode("FM-DEG-002")
            if fm:
                detected.append((fm, 1 - len(unique_texts) / len(texts)))

        # Check for extreme repetition
        ttrs = []
        for tokens in tokens_per_text:
            if tokens:
                ttrs.append(len(set(tokens)) / len(tokens))
        if ttrs and min(ttrs) < 0.1:
            fm = self._find_mode("FM-DEG-003")
            if fm:
                detected.append((fm, 1 - min(ttrs)))

        # Check length variation
        lengths = [len(t) for t in tokens_per_text if t]
        if lengths and max(lengths) / max(min(lengths), 1) > 10:
            fm = self._find_mode("FM-BND-002")
            if fm:
                detected.append((fm, 0.8))

        # Check small sample
        if len(texts) < 5:
            fm = self._find_mode("FM-SCL-001")
            if fm:
                detected.append((fm, 0.9))

        # Check hapax legomena
        all_tokens = [tok for t in tokens_per_text for tok in t]
        if all_tokens:
            token_counts = Counter(all_tokens)
            hapax_ratio = sum(1 for c in token_counts.values() if c == 1) / len(token_counts)
            if hapax_ratio > 0.5:
                fm = self._find_mode("FM-DIM-001")
                if fm:
                    detected.append((fm, hapax_ratio))

        return detected

    def _find_mode(self, mode_id: str) -> Optional[FailureMode]:
        for m in self.taxonomy.modes:
            if m.id == mode_id:
                return m
        return None


# ---------------------------------------------------------------------------
# Build complete taxonomy
# ---------------------------------------------------------------------------


def build_failure_taxonomy(seed: int = 42) -> FailureModeTaxonomy:
    """Build the complete failure mode taxonomy."""
    generator = FailureModeGenerator(seed)

    all_modes = []
    all_modes.extend(generator.generate_degenerate_inputs())
    all_modes.extend(generator.generate_boundary_cases())
    all_modes.extend(generator.generate_dimensionality_artifacts())
    all_modes.extend(generator.generate_scale_sensitivity())
    all_modes.extend(generator.generate_distribution_mismatches())
    all_modes.extend(generator.generate_metric_definition_gaps())

    # Category counts
    cat_counts = defaultdict(int)
    for m in all_modes:
        cat_counts[m.category.name] += 1

    # Severity distribution
    sev_dist = defaultdict(int)
    for m in all_modes:
        sev_dist[m.severity.value] += 1

    # Coverage
    coverage = defaultdict(int)
    for m in all_modes:
        for metric in m.affected_metrics:
            coverage[metric] += 1

    summary = {
        "total_failure_modes": len(all_modes),
        "categories": dict(cat_counts),
        "severity_distribution": dict(sev_dist),
        "metric_coverage": dict(coverage),
        "modes": [
            {
                "id": m.id,
                "name": m.name,
                "category": m.category.name,
                "severity": m.severity.value,
                "description": m.description,
                "structural_cause": m.structural_cause,
                "affected_metrics": m.affected_metrics,
                "detection_criterion": m.detection_criterion,
                "mitigation": m.mitigation,
                "metric_values": {
                    k: round(v, 4) if isinstance(v, float) and not math.isnan(v) else v
                    for k, v in (m.metric_values or {}).items()
                },
            }
            for m in all_modes
        ],
    }

    return FailureModeTaxonomy(
        modes=all_modes,
        category_counts=dict(cat_counts),
        severity_distribution=dict(sev_dist),
        total_modes=len(all_modes),
        coverage_metrics=dict(coverage),
        summary=summary,
    )


def failure_taxonomy_analysis() -> Dict:
    """Run failure mode taxonomy analysis and return JSON-serializable results."""
    taxonomy = build_failure_taxonomy()
    return taxonomy.summary


if __name__ == "__main__":
    import json
    results = failure_taxonomy_analysis()
    print(json.dumps(results, indent=2))
