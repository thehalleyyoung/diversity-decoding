"""
Evaluation framework for the Diversity Decoding Arena.

Provides the main evaluation arena, Pareto frontier analysis,
Bayesian statistical comparison, and hypervolume computation
for multi-objective diversity-quality analysis.
"""

from __future__ import annotations

from src.evaluation.arena import (
    EvaluationArena,
    ArenaConfig,
    ArenaResult,
    ArenaRun,
    RunStatus,
    AlgorithmEntry,
    ComparisonResult,
)
from src.evaluation.pareto import (
    ParetoFrontier,
    ParetoAnalyzer,
    ParetoPoint,
    DominanceRelation,
    NonDominatedSorting,
)
from src.evaluation.bayesian import (
    BayesianComparison,
    PosteriorEstimate,
    ROPEResult,
    CredibleInterval,
    ModelRanking,
    BayesianSignTest,
)
from src.evaluation.hypervolume import (
    HypervolumeIndicator,
    ExactHypervolume,
    ApproximateHypervolume,
    HypervolumeContribution,
)
from src.evaluation.ranking import (
    BradleyTerryModel,
    BTRanking,
    BayesianSignTest as BayesianSignTestROPE,
    BayesianSignTestResult,
    MetricDiscriminativePower,
    DiscriminativePowerResult,
    rank_metrics_by_discriminative_power,
)

__all__ = [
    "EvaluationArena",
    "ArenaConfig",
    "ArenaResult",
    "ArenaRun",
    "RunStatus",
    "AlgorithmEntry",
    "ComparisonResult",
    "ParetoFrontier",
    "ParetoAnalyzer",
    "ParetoPoint",
    "DominanceRelation",
    "NonDominatedSorting",
    "BayesianComparison",
    "PosteriorEstimate",
    "ROPEResult",
    "CredibleInterval",
    "ModelRanking",
    "BayesianSignTest",
    "HypervolumeIndicator",
    "ExactHypervolume",
    "ApproximateHypervolume",
    "HypervolumeContribution",
    "BradleyTerryModel",
    "BTRanking",
    "BayesianSignTestROPE",
    "BayesianSignTestResult",
    "MetricDiscriminativePower",
    "DiscriminativePowerResult",
    "rank_metrics_by_discriminative_power",
]
