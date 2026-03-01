"""
Decoding algorithm implementations.

Each algorithm implements the DecodingAlgorithm interface and can be
registered in the algorithm registry for automatic discovery.

Baseline algorithms:
    TemperatureSampling, TopKSampling, NucleusSampling, TypicalDecoding

Structured search algorithms:
    DiverseBeamSearch, ContrastiveSearch

Post-hoc diversity algorithms:
    DPPReranking, MBRDiversity

Novel algorithms:
    SteinVariationalDecoding (SVD), QualityDiversityBeamSearch (QD-BS)
"""

from src.algorithms.base import DecodingAlgorithm, DecodingConfig, DecodingState
from src.algorithms.nucleus import NucleusSampling
from src.algorithms.typical import TypicalDecoding
from src.algorithms.contrastive import ContrastiveSearch
from src.algorithms.diverse_beam import DiverseBeamSearch
from src.algorithms.dpp import DPPReranking
from src.algorithms.mbr import MBRDiversity
from src.algorithms.svd import SteinVariationalDecoding
from src.algorithms.qdbs import QualityDiversityBeamSearch
from src.algorithms.temperature import TemperatureSampling
from src.algorithms.topk import TopKSampling

__all__ = [
    "DecodingAlgorithm",
    "DecodingConfig",
    "DecodingState",
    "NucleusSampling",
    "TypicalDecoding",
    "ContrastiveSearch",
    "DiverseBeamSearch",
    "DPPReranking",
    "MBRDiversity",
    "SteinVariationalDecoding",
    "QualityDiversityBeamSearch",
    "TemperatureSampling",
    "TopKSampling",
]
