"""
Diversity Decoding Arena — Core Package
========================================

A comprehensive evaluation framework for systematically comparing
diversity-promoting decoding algorithms in text generation.

Provides:
- LogitSource abstraction for model inference
- 10+ algorithmically distinct decoding families
- 7 diversity metrics and 3 quality metrics
- Pareto frontier analysis and statistical comparison
- Curriculum diversity for education
- Multimodal diversity across text, image, audio, video
- Constrained diverse generation with composable constraints
- Fairness analysis through diversity measurement
- Collaborative diversity for multi-user settings
- Temporal diversity analysis with forecasting
- Domain-specific diversity (legal, medical, financial, scientific, engineering)
- Advanced optimization algorithms (DPP, facility location, dispersion)

Usage::

    from src.config import ArenaConfig
    from src.logit_source import LiveLogitSource, CachedLogitSource
    from src.algorithms import NucleusSampling, SteinVariationalDecoding
    from src.metrics import SelfBLEU, VendiScore, Perplexity
    from src.curriculum_diversity import generate_diverse_exercises
    from src.optimization_algorithms import determinantal_point_process

"""

__version__ = "0.2.0"
__author__ = "Diversity Decoding Arena Contributors"

from src.config import ArenaConfig
from src.types import (
    TokenID,
    LogitVector,
    GenerationSet,
    GenerationResult,
    MetricResult,
    AlgorithmConfig,
)
