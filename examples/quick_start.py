#!/usr/bin/env python3
"""
Quick-start example for the Diversity Decoding Arena.

Shows the three things that make this toolkit unique:
  1. Compute ALL diversity metrics at once on your generated texts
  2. Compare two sets of generations side-by-side
  3. Build a correlation taxonomy to find which metrics are redundant

Run from the implementation/ directory:
    python examples/quick_start.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.metrics.diversity import (
    BehavioralDiversity,
    DistinctN,
    DiversityMetricSuite,
    EmbeddingPairwiseDistance,
    NGramEntropy,
    SelfBLEU,
    VendiScore,
)
from src.metrics.correlation import (
    MetricCorrelationAnalyzer,
    MetricRedundancyAnalyzer,
    effective_dimensionality,
)

# -----------------------------------------------------------------------
# 1. Compute all diversity metrics on a single set of texts
# -----------------------------------------------------------------------

print("=" * 60)
print("1. Compute diversity metrics on a set of texts")
print("=" * 60)

texts = [
    "The cat sat on the mat and purred softly in the afternoon sun.",
    "A dog ran through the park chasing squirrels and barking loudly.",
    "The scientist discovered a new species of butterfly in the Amazon.",
    "She played the piano with passion, filling the room with melody.",
    "The old lighthouse stood against the crashing waves at midnight.",
]

suite = DiversityMetricSuite([
    SelfBLEU(max_order=4),
    DistinctN(n=1),
    DistinctN(n=2),
    NGramEntropy(n=2),
    EmbeddingPairwiseDistance(distance_metric="cosine", embedding_method="tfidf"),
    VendiScore(kernel_type="cosine"),
])

results = suite.compute_all(texts)
print(f"\n{'Metric':<30} {'Value':>10}  Direction")
print("-" * 55)
for name, value in results.items():
    metric = suite.get_metric(name)
    direction = "↑ higher = more diverse" if metric.higher_is_better else "↓ lower = more diverse"
    print(f"{name:<30} {value:>10.4f}  {direction}")

# -----------------------------------------------------------------------
# 2. Compare two sets of generations
# -----------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. Compare diverse vs. repetitive generations")
print("=" * 60)

diverse_texts = [
    "The quantum computer solved the optimization problem in seconds.",
    "A butterfly emerged from its chrysalis in the morning dew.",
    "The chef prepared a five-course meal inspired by coastal cuisine.",
    "Ancient ruins revealed a previously unknown civilization's artwork.",
    "The jazz ensemble improvised a twelve-minute piece at the festival.",
]

repetitive_texts = [
    "The cat sat on the mat. The cat was happy.",
    "The cat sat on the rug. The cat was content.",
    "The cat sat on the sofa. The cat was pleased.",
    "The cat sat on the chair. The cat was relaxed.",
    "The cat sat on the bed. The cat was comfortable.",
]

diverse_scores = suite.compute_all(diverse_texts)
repetitive_scores = suite.compute_all(repetitive_texts)

print(f"\n{'Metric':<30} {'Diverse':>10} {'Repetitive':>10}  {'Winner':>10}")
print("-" * 65)
for name in suite.metric_names:
    d = diverse_scores[name]
    r = repetitive_scores[name]
    metric = suite.get_metric(name)
    if metric.higher_is_better:
        winner = "Diverse" if d > r else "Repetitive"
    else:
        winner = "Diverse" if d < r else "Repetitive"
    print(f"{name:<30} {d:>10.4f} {r:>10.4f}  {winner:>10}")

# -----------------------------------------------------------------------
# 3. Build a correlation taxonomy (which metrics are redundant?)
# -----------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. Metric correlation taxonomy")
print("=" * 60)
print("   (Which metrics capture the same signal?)\n")

# Generate multiple groups with varying diversity to build correlations
np.random.seed(42)
base_sentences = [
    "The weather today is sunny and warm with clear blue skies.",
    "Artificial intelligence is transforming the healthcare industry rapidly.",
    "The garden bloomed with colorful flowers in the spring season.",
    "A mysterious signal was detected from a distant galaxy cluster.",
    "The orchestra performed a stunning rendition of the classical symphony.",
    "Deep ocean exploration revealed unknown species of bioluminescent creatures.",
    "The architect designed a sustainable building using recycled materials.",
    "Mountain climbers reached the summit after a grueling three-day ascent.",
]


def make_group(base: list[str], shuffle_words: bool, n: int = 5) -> list[str]:
    """Create a group by selecting and optionally shuffling sentences."""
    selected = [base[i % len(base)] for i in range(n)]
    if shuffle_words:
        result = []
        for s in selected:
            words = s.split()
            np.random.shuffle(words)
            result.append(" ".join(words))
        return result
    return selected


# Create 6 groups with varying diversity levels
groups = {
    "identical":     [base_sentences[0]] * 5,
    "near_dup":      [base_sentences[0], base_sentences[0],
                      base_sentences[0], base_sentences[1], base_sentences[1]],
    "low_div":       base_sentences[:3] + base_sentences[:2],
    "medium_div":    base_sentences[:5],
    "high_div":      base_sentences[:5] + base_sentences[5:8],
    "shuffled":      make_group(base_sentences, shuffle_words=True, n=6),
}

# Compute metrics for each group
metric_names = suite.metric_names
all_results = {}
for gname, gtexts in groups.items():
    if len(gtexts) < 2:
        continue
    all_results[gname] = suite.compute_all(gtexts)

# Build correlation matrix
group_list = list(all_results.keys())
metric_values = {m: [all_results[g][m] for g in group_list] for m in metric_names}

# Filter out constant metrics
active_metrics = [
    m for m in metric_names if np.std(metric_values[m]) > 1e-12
]

if len(active_metrics) >= 2 and len(group_list) >= 3:
    analyzer = MetricCorrelationAnalyzer(metrics=active_metrics)
    active_values = {m: metric_values[m] for m in active_metrics}
    corr = analyzer.compute_correlation_matrix(active_values)

    # Print correlation matrix
    col_w = 10
    header = " " * 25 + "".join(f"{m[:col_w]:>{col_w}}" for m in active_metrics)
    print(header)
    for i, name in enumerate(active_metrics):
        row = f"{name:<25}" + "".join(
            f"{corr[i, j]:>{col_w}.3f}" for j in range(len(active_metrics))
        )
        print(row)

    # Find redundant and complementary pairs
    redundancy = MetricRedundancyAnalyzer(threshold=0.8)
    redundant = redundancy.find_redundant_pairs(corr, active_metrics)
    orthogonal = redundancy.find_orthogonal_pairs(corr, active_metrics)
    recommended = redundancy.select_representative_metrics(corr, active_metrics)

    print("\nRedundant pairs (|τ| ≥ 0.8) — these measure the same thing:")
    if redundant:
        for a, b, tau in redundant:
            print(f"  {a}  ↔  {b}  (τ = {tau:+.3f})")
    else:
        print("  None — all metrics are independent!")

    print("\nComplementary pairs (|τ| ≤ 0.1) — these capture different aspects:")
    if orthogonal:
        for a, b, tau in orthogonal:
            print(f"  {a}  ↔  {b}  (τ = {tau:+.3f})")
    else:
        print("  None at threshold 0.1")

    eigenvalues = np.linalg.eigvalsh(corr)
    eff_dim = effective_dimensionality(np.maximum(eigenvalues, 0))
    print(f"\nEffective dimensionality: {eff_dim:.1f} / {len(active_metrics)}")
    print(f"Recommended non-redundant subset: {', '.join(recommended)}")
    print("\n→ Use this subset for efficient diversity evaluation.")
else:
    print("Not enough variation across groups for correlation analysis.")

print("\nDone!")
