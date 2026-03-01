"""
Diversity and quality metric implementations.

Diversity Metrics:
    SelfBLEU, DistinctN, NGramEntropy, EmbeddingPairwiseDistance,
    VendiScore, ParseTreeDiversity, BehavioralDiversity

Neural Diversity Metrics:
    MAUVE, BERTScoreDiversity, STSDiversity, CompressionRatioDiversity

Quality Metrics:
    Perplexity, NLICoherence, ConstraintSatisfaction

Analysis:
    MetricCorrelationAnalyzer — Kendall τ correlation matrix and
    spectral clustering for metric taxonomy discovery.
"""

from src.metrics.diversity import (
    DiversityMetric,
    SelfBLEU,
    DistinctN,
    NGramEntropy,
    EmbeddingPairwiseDistance,
    VendiScore,
    ParseTreeDiversity,
    BehavioralDiversity,
)
from src.metrics.neural_diversity import (
    MAUVE,
    BERTScoreDiversity,
    STSDiversity,
    CompressionRatioDiversity,
    NeuralDiversitySuite,
)
from src.metrics.quality import (
    QualityMetric,
    Perplexity,
    NLICoherence,
    ConstraintSatisfaction,
)
from src.metrics.correlation import MetricCorrelationAnalyzer
from src.metrics.information_theoretic import (
    shannon_entropy,
    kl_divergence,
    symmetric_kl,
    mutual_information,
    entropy_rate,
    bootstrap_entropy_ci,
)
from src.metrics.bootstrap import (
    bootstrap_ci,
    bootstrap_kendall_tau,
    bootstrap_fair_retention,
)

__all__ = [
    "DiversityMetric",
    "SelfBLEU",
    "DistinctN",
    "NGramEntropy",
    "EmbeddingPairwiseDistance",
    "VendiScore",
    "ParseTreeDiversity",
    "BehavioralDiversity",
    "MAUVE",
    "BERTScoreDiversity",
    "STSDiversity",
    "CompressionRatioDiversity",
    "NeuralDiversitySuite",
    "QualityMetric",
    "Perplexity",
    "NLICoherence",
    "ConstraintSatisfaction",
    "MetricCorrelationAnalyzer",
    "shannon_entropy",
    "kl_divergence",
    "symmetric_kl",
    "mutual_information",
    "entropy_rate",
    "bootstrap_entropy_ci",
    "bootstrap_ci",
    "bootstrap_kendall_tau",
    "bootstrap_fair_retention",
]
