"""
Visualization module for the Diversity Decoding Arena.

Provides plotting utilities for Pareto frontiers, metric heatmaps,
algorithm comparison charts, and HTML report generation.
"""

from __future__ import annotations

from src.visualization.pareto_plots import (
    ParetoPlotter,
    ParetoPlotConfig,
    FrontierAnimation,
)
from src.visualization.metric_heatmaps import (
    MetricHeatmapPlotter,
    CorrelationMatrix,
    HeatmapConfig,
)
from src.visualization.algorithm_comparison import (
    AlgorithmComparisonPlotter,
    ComparisonConfig,
    RankingChart,
)
from src.visualization.reports import (
    HTMLReportGenerator,
    ReportConfig,
    ReportSection,
    ReportBuilder,
)
from src.visualization.metric_heatmaps import (
    HierarchicalClusterHeatmap,
    MetricStabilityHeatmap,
    CrossDomainCorrelationHeatmap,
    AnnotatedSignificanceHeatmap,
)
from src.visualization.pareto_plots import (
    ParetoSurface3D,
    ParetoFrontierEvolution,
    ParetoFamilyComparison,
)
from src.visualization.interactive import (
    FullDashboardGenerator,
    InteractiveMetricExplorer,
    AlgorithmComparisonView,
    StandaloneHTMLExporter,
)
from src.visualization.convergence_plots import (
    ConvergencePlotter,
    ConvergenceConfig,
)
from src.visualization.ranking_plots import (
    RankingPlotter,
    RankingPlotConfig,
)

__all__ = [
    "ParetoPlotter",
    "ParetoPlotConfig",
    "FrontierAnimation",
    "MetricHeatmapPlotter",
    "CorrelationMatrix",
    "HeatmapConfig",
    "AlgorithmComparisonPlotter",
    "ComparisonConfig",
    "RankingChart",
    "HTMLReportGenerator",
    "ReportConfig",
    "ReportSection",
    "ReportBuilder",
    "HierarchicalClusterHeatmap",
    "MetricStabilityHeatmap",
    "CrossDomainCorrelationHeatmap",
    "AnnotatedSignificanceHeatmap",
    "ParetoSurface3D",
    "ParetoFrontierEvolution",
    "ParetoFamilyComparison",
    "FullDashboardGenerator",
    "InteractiveMetricExplorer",
    "AlgorithmComparisonView",
    "StandaloneHTMLExporter",
    "ConvergencePlotter",
    "ConvergenceConfig",
    "RankingPlotter",
    "RankingPlotConfig",
]
