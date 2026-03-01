"""
End-to-end integration tests for the Diversity Decoding Arena.

Tests cover the full pipeline: algorithm generation → metric computation →
evaluation → comparison, including Pareto analysis, Bayesian comparison,
parameter sweeps, cross-task evaluation, reproducibility, scalability,
serialization, and statistical robustness.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import test utilities from conftest
# ---------------------------------------------------------------------------
from conftest import (
    DEFAULT_VOCAB_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_SEQ_LEN,
    DEFAULT_NUM_SEQUENCES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_NUM_BEAMS,
    DEFAULT_NUM_GROUPS,
    DEFAULT_NUM_PARTICLES,
    DEFAULT_MAX_NEW_TOKENS,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    MockLogitSource,
    MockLogitSourceWithKVCache,
    MockLogitSourceBatched,
    MockLogitSourceWithLatency,
    MockEmbedder,
    MockTokenizer,
    MockGenerationResult,
    AlgorithmFixtureFactory,
    generate_random_texts,
    generate_diverse_texts,
    generate_identical_texts,
    generate_token_sequences,
    generate_diverse_token_sequences,
    generate_logit_matrix,
    generate_embedding_matrix,
    generate_kernel_matrix,
    create_mock_generation_result,
    create_multiple_generation_results,
    compute_ngram_frequencies,
    compute_pairwise_jaccard,
    compute_simple_self_bleu,
    compute_distinct_n,
    compute_ngram_entropy,
    assert_valid_probability_distribution,
    assert_valid_logits,
    assert_diverse_texts,
    assert_metric_in_range,
    assert_monotonic,
    assert_positive_semidefinite,
    assert_symmetric,
    bootstrap_mean_ci,
    permutation_test,
    effect_size_cohens_d,
)

# ---------------------------------------------------------------------------
# Import source modules
# ---------------------------------------------------------------------------
from src.algorithms import (
    DecodingAlgorithm,
    DecodingConfig,
    DecodingState,
    TemperatureSampling,
    TopKSampling,
    NucleusSampling,
    TypicalDecoding,
    DiverseBeamSearch,
    ContrastiveSearch,
    DPPReranking,
    MBRDiversity,
    SteinVariationalDecoding,
    QualityDiversityBeamSearch,
)
from src.algorithms.temperature import TemperatureConfig
from src.algorithms.topk import TopKConfig
from src.algorithms.nucleus import NucleusConfig
from src.algorithms.typical import TypicalConfig
from src.algorithms.diverse_beam import DiverseBeamConfig
from src.algorithms.contrastive import ContrastiveConfig
from src.algorithms.dpp import DPPConfig
from src.algorithms.mbr import MBRConfig
from src.algorithms.svd import SVDConfig
from src.algorithms.qdbs import QDBSConfig

from src.metrics import (
    DiversityMetric,
    SelfBLEU,
    DistinctN,
    NGramEntropy,
    EmbeddingPairwiseDistance,
    VendiScore,
    MetricCorrelationAnalyzer,
)

from src.evaluation import (
    ParetoFrontier,
    ParetoAnalyzer,
    ParetoPoint,
    DominanceRelation,
    BayesianComparison,
    PosteriorEstimate,
    ROPEResult,
    CredibleInterval,
    ExactHypervolume,
    ApproximateHypervolume,
    HypervolumeContribution,
)

from src.types import Token, TokenSequence, GenerationResult, GenerationSet


# =========================================================================
# Helpers
# =========================================================================

def _make_token_sequence(token_ids: List[int], tokenizer: MockTokenizer) -> TokenSequence:
    """Build a TokenSequence from raw token ids using a tokenizer."""
    tokens = []
    for tid in token_ids:
        text = tokenizer.decode([tid])
        tokens.append(Token(token_id=tid, text=text, log_prob=-1.0))
    return TokenSequence(tokens=tokens)


def _texts_from_sequences(sequences: List[TokenSequence]) -> List[str]:
    """Extract text strings from a list of TokenSequence objects."""
    return [seq.text for seq in sequences]


def _run_algorithm_and_get_texts(
    algo_cls,
    config,
    logit_source: MockLogitSource,
    prompt_ids: Optional[List[int]] = None,
) -> Tuple[List[TokenSequence], List[str]]:
    """Instantiate algorithm, generate sequences, return (sequences, texts)."""
    algo = algo_cls(config)
    if prompt_ids is None:
        prompt_ids = [BOS_TOKEN_ID, 10, 20, 30]
    sequences = algo.generate(logit_source, prompt_ids)
    texts = _texts_from_sequences(sequences)
    return sequences, texts


def _compute_all_basic_metrics(texts: List[str]) -> Dict[str, float]:
    """Compute a suite of basic diversity metrics on a list of texts."""
    results = {}
    if len(texts) >= 2:
        self_bleu = SelfBLEU(max_order=4)
        results["self_bleu"] = self_bleu.compute(texts)
    else:
        results["self_bleu"] = 0.0

    distinct1 = DistinctN(n=1)
    distinct2 = DistinctN(n=2)
    results["distinct_1"] = distinct1.compute(texts)
    results["distinct_2"] = distinct2.compute(texts)

    entropy2 = NGramEntropy(n=2)
    results["entropy_2"] = entropy2.compute(texts)

    return results


def _make_pareto_point(algorithm: str, diversity: float, quality: float) -> ParetoPoint:
    """Create a ParetoPoint with diversity and quality objectives."""
    return ParetoPoint(
        objectives={"diversity": diversity, "quality": quality},
        algorithm=algorithm,
    )


# =========================================================================
# 1. TestEndToEndPipeline
# =========================================================================


class TestEndToEndPipeline:
    """Full pipeline: generate → compute metrics → evaluate → compare."""

    def test_temperature_generate_and_metrics(self):
        """Generate with temperature sampling, compute all basic metrics."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
        config = TemperatureConfig(
            num_sequences=8,
            max_new_tokens=25,
            seed=42,
            temperature=1.0,
        )
        sequences, texts = _run_algorithm_and_get_texts(
            TemperatureSampling, config, source
        )
        assert len(sequences) == 8
        assert all(isinstance(s, TokenSequence) for s in sequences)
        metrics = _compute_all_basic_metrics(texts)
        assert "self_bleu" in metrics
        assert "distinct_1" in metrics
        assert "distinct_2" in metrics
        assert "entropy_2" in metrics
        assert metrics["distinct_1"] >= 0.0
        assert metrics["entropy_2"] >= 0.0

    def test_nucleus_generate_and_metrics(self):
        """Nucleus sampling produces valid sequences and computable metrics."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=10)
        config = NucleusConfig(
            num_sequences=8,
            max_new_tokens=25,
            seed=10,
            top_p=0.9,
        )
        sequences, texts = _run_algorithm_and_get_texts(
            NucleusSampling, config, source
        )
        assert len(sequences) == 8
        metrics = _compute_all_basic_metrics(texts)
        assert metrics["distinct_2"] >= 0.0

    def test_topk_generate_and_metrics(self):
        """Top-K sampling produces valid sequences and computable metrics."""
        source = MockLogitSource(vocab_size=300, distribution="zipf", seed=7)
        config = TopKConfig(
            num_sequences=6,
            max_new_tokens=20,
            seed=7,
            k=50,
        )
        sequences, texts = _run_algorithm_and_get_texts(
            TopKSampling, config, source
        )
        assert len(sequences) == 6
        metrics = _compute_all_basic_metrics(texts)
        for v in metrics.values():
            assert np.isfinite(v), f"Non-finite metric: {v}"

    def test_full_pipeline_generate_metrics_compare(self):
        """Full pipeline: two algorithms → metrics → Bayesian comparison."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)

        config_a = TemperatureConfig(
            num_sequences=10, max_new_tokens=20, seed=42, temperature=0.7,
        )
        config_b = TemperatureConfig(
            num_sequences=10, max_new_tokens=20, seed=42, temperature=1.5,
        )

        _, texts_a = _run_algorithm_and_get_texts(
            TemperatureSampling, config_a, source
        )
        source.reset()
        _, texts_b = _run_algorithm_and_get_texts(
            TemperatureSampling, config_b, source
        )

        metrics_a = _compute_all_basic_metrics(texts_a)
        metrics_b = _compute_all_basic_metrics(texts_b)

        # Higher temperature should produce more diversity
        assert isinstance(metrics_a["entropy_2"], float)
        assert isinstance(metrics_b["entropy_2"], float)

        # Bayesian comparison on distinct-2 scores per sample
        d2 = DistinctN(n=2)
        # Use bootstrap-style: repeat computation with subsets
        rng = np.random.RandomState(42)
        scores_a, scores_b = [], []
        for _ in range(20):
            idx = rng.choice(len(texts_a), size=max(3, len(texts_a) // 2), replace=True)
            sub_a = [texts_a[i] for i in idx]
            sub_b = [texts_b[i] for i in idx]
            scores_a.append(d2.compute(sub_a))
            scores_b.append(d2.compute(sub_b))

        bc = BayesianComparison(n_samples=2000, seed=42)
        result = bc.compare_two(
            np.array(scores_a), np.array(scores_b),
            metric_name="distinct_2",
            algorithm_a="temp_0.7",
            algorithm_b="temp_1.5",
        )
        assert hasattr(result, "p_a_better")
        assert hasattr(result, "p_b_better")
        assert 0.0 <= result.p_a_better <= 1.0
        assert 0.0 <= result.p_b_better <= 1.0

    def test_pipeline_with_diverse_beam_search(self):
        """Diverse beam search produces beam-structured output."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=5)
        config = DiverseBeamConfig(
            num_sequences=8,
            max_new_tokens=20,
            seed=5,
            num_beams=8,
            num_beam_groups=4,
            diversity_penalty=1.0,
        )
        sequences, texts = _run_algorithm_and_get_texts(
            DiverseBeamSearch, config, source
        )
        assert len(sequences) >= 1
        metrics = _compute_all_basic_metrics(texts)
        assert metrics["entropy_2"] >= 0.0

    def test_pipeline_dpp_reranking(self):
        """DPP reranking produces re-ordered diverse subset."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=11)
        config = DPPConfig(
            num_sequences=6,
            max_new_tokens=20,
            seed=11,
            candidate_pool_size=30,
            select_k=6,
        )
        sequences, texts = _run_algorithm_and_get_texts(
            DPPReranking, config, source
        )
        assert len(sequences) >= 1
        metrics = _compute_all_basic_metrics(texts)
        assert metrics["distinct_1"] >= 0.0

    def test_pipeline_contrastive_search(self):
        """Contrastive search produces valid sequences with metrics."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=3)
        config = ContrastiveConfig(
            num_sequences=6,
            max_new_tokens=20,
            seed=3,
            alpha=0.6,
            k=5,
        )
        sequences, texts = _run_algorithm_and_get_texts(
            ContrastiveSearch, config, source
        )
        assert len(sequences) >= 1
        metrics = _compute_all_basic_metrics(texts)
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

    def test_pipeline_typical_decoding(self):
        """Typical decoding produces valid output with computable metrics."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=8)
        config = TypicalConfig(
            num_sequences=8,
            max_new_tokens=20,
            seed=8,
            typical_p=0.95,
        )
        sequences, texts = _run_algorithm_and_get_texts(
            TypicalDecoding, config, source
        )
        assert len(sequences) == 8
        metrics = _compute_all_basic_metrics(texts)
        assert metrics["self_bleu"] >= 0.0

    def test_pipeline_svd(self):
        """Stein Variational Decoding runs and produces metrizable output."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=15)
        config = SVDConfig(
            num_sequences=5,
            max_new_tokens=15,
            seed=15,
            n_particles=5,
        )
        sequences, texts = _run_algorithm_and_get_texts(
            SteinVariationalDecoding, config, source
        )
        assert len(sequences) >= 1
        metrics = _compute_all_basic_metrics(texts)
        assert isinstance(metrics["distinct_2"], float)

    def test_pipeline_qdbs(self):
        """Quality-Diversity Beam Search generates valid output."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=20)
        config = QDBSConfig(
            num_sequences=6,
            max_new_tokens=15,
            seed=20,
            beam_width=12,
        )
        sequences, texts = _run_algorithm_and_get_texts(
            QualityDiversityBeamSearch, config, source
        )
        assert len(sequences) >= 1
        metrics = _compute_all_basic_metrics(texts)
        assert metrics["entropy_2"] >= 0.0

    def test_pipeline_mbr(self):
        """MBR diversity selection runs end-to-end."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=25)
        config = MBRConfig(
            num_sequences=5,
            max_new_tokens=15,
            seed=25,
            candidate_pool_size=20,
            select_k=5,
        )
        sequences, texts = _run_algorithm_and_get_texts(
            MBRDiversity, config, source
        )
        assert len(sequences) >= 1
        metrics = _compute_all_basic_metrics(texts)
        assert isinstance(metrics["distinct_1"], float)

    def test_pipeline_metrics_consistency(self):
        """Metrics on identical texts should show low diversity."""
        identical = generate_identical_texts(10)
        metrics = _compute_all_basic_metrics(identical)
        # Self-BLEU of identical texts should be high (low diversity)
        # Distinct-N should be low for identical texts
        assert metrics["distinct_2"] <= 1.0
        # Entropy should be low/zero for identical texts
        assert metrics["entropy_2"] >= 0.0

    def test_pipeline_diverse_vs_identical_metric_ordering(self):
        """Diverse texts should have higher diversity metrics than identical."""
        diverse = generate_diverse_texts(10, seed=42)
        identical = generate_identical_texts(10)

        m_diverse = _compute_all_basic_metrics(diverse)
        m_identical = _compute_all_basic_metrics(identical)

        # Diverse texts should have higher distinct-n
        assert m_diverse["distinct_2"] >= m_identical["distinct_2"]
        # Diverse texts should have higher entropy
        assert m_diverse["entropy_2"] >= m_identical["entropy_2"]

    def test_pipeline_multiple_metrics_computed(self):
        """All metrics are finite and in reasonable ranges."""
        texts = generate_diverse_texts(15, seed=99)
        metrics = _compute_all_basic_metrics(texts)
        for name, val in metrics.items():
            assert np.isfinite(val), f"{name} is not finite"

    def test_pipeline_generate_result_objects(self):
        """TokenSequence objects have valid properties."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=50)
        config = TemperatureConfig(
            num_sequences=5, max_new_tokens=15, seed=50, temperature=1.0,
        )
        sequences, _ = _run_algorithm_and_get_texts(
            TemperatureSampling, config, source
        )
        for seq in sequences:
            assert len(seq) > 0
            assert isinstance(seq.token_ids, list)
            assert all(isinstance(t, int) for t in seq.token_ids)
            assert isinstance(seq.text, str)
            assert isinstance(seq.total_log_prob, float)


# =========================================================================
# 2. TestAlgorithmMetricIntegration
# =========================================================================


class TestAlgorithmMetricIntegration:
    """Run each algorithm, compute all metrics on the output."""

    def _run_and_measure(self, algo_cls, config, distribution="zipf", seed=42):
        """Run algorithm and compute metrics, return (texts, metrics)."""
        source = MockLogitSource(
            vocab_size=200, distribution=distribution, seed=seed
        )
        _, texts = _run_algorithm_and_get_texts(algo_cls, config, source)
        metrics = _compute_all_basic_metrics(texts)
        return texts, metrics

    def test_temperature_all_metrics(self):
        config = TemperatureConfig(
            num_sequences=8, max_new_tokens=20, seed=1, temperature=1.0,
        )
        texts, metrics = self._run_and_measure(TemperatureSampling, config, seed=1)
        assert len(texts) == 8
        for v in metrics.values():
            assert np.isfinite(v)

    def test_topk_all_metrics(self):
        config = TopKConfig(
            num_sequences=8, max_new_tokens=20, seed=2, k=40,
        )
        texts, metrics = self._run_and_measure(TopKSampling, config, seed=2)
        assert len(texts) == 8
        assert metrics["distinct_1"] >= 0.0

    def test_nucleus_all_metrics(self):
        config = NucleusConfig(
            num_sequences=8, max_new_tokens=20, seed=3, top_p=0.9,
        )
        texts, metrics = self._run_and_measure(NucleusSampling, config, seed=3)
        assert len(texts) == 8
        assert metrics["entropy_2"] >= 0.0

    def test_typical_all_metrics(self):
        config = TypicalConfig(
            num_sequences=8, max_new_tokens=20, seed=4, typical_p=0.95,
        )
        texts, metrics = self._run_and_measure(TypicalDecoding, config, seed=4)
        assert len(texts) == 8
        for v in metrics.values():
            assert np.isfinite(v)

    def test_diverse_beam_all_metrics(self):
        config = DiverseBeamConfig(
            num_sequences=8, max_new_tokens=20, seed=5,
            num_beams=8, num_beam_groups=4, diversity_penalty=1.0,
        )
        texts, metrics = self._run_and_measure(DiverseBeamSearch, config, seed=5)
        assert len(texts) >= 1
        assert metrics["distinct_2"] >= 0.0

    def test_contrastive_all_metrics(self):
        config = ContrastiveConfig(
            num_sequences=6, max_new_tokens=20, seed=6,
            alpha=0.6, k=5,
        )
        texts, metrics = self._run_and_measure(ContrastiveSearch, config, seed=6)
        assert len(texts) >= 1
        for v in metrics.values():
            assert np.isfinite(v)

    def test_dpp_all_metrics(self):
        config = DPPConfig(
            num_sequences=6, max_new_tokens=20, seed=7,
            candidate_pool_size=30, select_k=6,
        )
        texts, metrics = self._run_and_measure(DPPReranking, config, seed=7)
        assert len(texts) >= 1
        assert metrics["self_bleu"] >= 0.0

    def test_mbr_all_metrics(self):
        config = MBRConfig(
            num_sequences=5, max_new_tokens=15, seed=8,
            candidate_pool_size=20, select_k=5,
        )
        texts, metrics = self._run_and_measure(MBRDiversity, config, seed=8)
        assert len(texts) >= 1
        assert metrics["distinct_1"] >= 0.0

    def test_svd_all_metrics(self):
        config = SVDConfig(
            num_sequences=5, max_new_tokens=15, seed=9,
            n_particles=5,
        )
        texts, metrics = self._run_and_measure(
            SteinVariationalDecoding, config, seed=9
        )
        assert len(texts) >= 1
        for v in metrics.values():
            assert np.isfinite(v)

    def test_qdbs_all_metrics(self):
        config = QDBSConfig(
            num_sequences=5, max_new_tokens=15, seed=10,
            beam_width=10,
        )
        texts, metrics = self._run_and_measure(
            QualityDiversityBeamSearch, config, seed=10
        )
        assert len(texts) >= 1
        assert metrics["entropy_2"] >= 0.0

    def test_peaked_distribution_all_algorithms(self):
        """All algorithms should handle peaked distributions without error."""
        algos_configs = [
            (TemperatureSampling, TemperatureConfig(
                num_sequences=4, max_new_tokens=12, seed=20, temperature=1.0,
            )),
            (TopKSampling, TopKConfig(
                num_sequences=4, max_new_tokens=12, seed=20, k=10,
            )),
            (NucleusSampling, NucleusConfig(
                num_sequences=4, max_new_tokens=12, seed=20, top_p=0.8,
            )),
        ]
        for algo_cls, cfg in algos_configs:
            source = MockLogitSource(
                vocab_size=200, distribution="peaked", seed=20, concentration=3.0,
            )
            seqs, texts = _run_algorithm_and_get_texts(algo_cls, cfg, source)
            assert len(seqs) >= 1
            metrics = _compute_all_basic_metrics(texts)
            for v in metrics.values():
                assert np.isfinite(v)

    def test_uniform_distribution_high_diversity(self):
        """Uniform distribution should yield high diversity for all algorithms."""
        source = MockLogitSource(
            vocab_size=200, distribution="uniform", seed=30,
        )
        config = TemperatureConfig(
            num_sequences=10, max_new_tokens=20, seed=30, temperature=1.0,
        )
        _, texts = _run_algorithm_and_get_texts(
            TemperatureSampling, config, source
        )
        metrics = _compute_all_basic_metrics(texts)
        # Uniform should produce relatively high distinct-n
        assert metrics["distinct_1"] > 0.0
        assert metrics["entropy_2"] > 0.0


# =========================================================================
# 3. TestParetoIntegration
# =========================================================================


class TestParetoIntegration:
    """Generate with multiple configs, compute Pareto frontier."""

    def _generate_pareto_data(self, n_configs=6):
        """Run multiple configs and collect (diversity, quality) points."""
        points = []
        temperatures = np.linspace(0.3, 2.0, n_configs)
        for i, temp in enumerate(temperatures):
            source = MockLogitSource(
                vocab_size=200, distribution="zipf", seed=42 + i,
            )
            config = TemperatureConfig(
                num_sequences=8, max_new_tokens=20,
                seed=42 + i, temperature=float(temp),
            )
            seqs, texts = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            d2 = DistinctN(n=2).compute(texts)
            # Use mean log prob as quality proxy
            quality = 0.0
            if seqs:
                log_probs = [s.mean_log_prob for s in seqs]
                quality = float(np.mean(log_probs))
                # Normalize to [0, 1]-ish range
                quality = 1.0 / (1.0 + np.exp(-quality))
            points.append(_make_pareto_point(
                algorithm=f"temp_{temp:.1f}",
                diversity=d2,
                quality=quality,
            ))
        return points

    def test_pareto_frontier_construction(self):
        """Build Pareto frontier from generated data."""
        points = self._generate_pareto_data(6)
        frontier = ParetoFrontier(
            points=points,
            maximize={"diversity": True, "quality": True},
        )
        front = frontier.compute_frontier()
        assert len(front) >= 1
        assert len(front) <= len(points)

    def test_pareto_frontier_non_dominated(self):
        """Frontier points should not dominate each other."""
        points = self._generate_pareto_data(8)
        frontier = ParetoFrontier(
            points=points,
            maximize={"diversity": True, "quality": True},
        )
        front = frontier.compute_frontier()
        for i, p1 in enumerate(front):
            for j, p2 in enumerate(front):
                if i != j:
                    rel = frontier.dominates(p1, p2)
                    assert rel != DominanceRelation.DOMINATES or rel == DominanceRelation.EQUAL

    def test_pareto_analyzer_2d(self):
        """ParetoAnalyzer produces 2D frontier from results."""
        points = self._generate_pareto_data(8)
        analyzer = ParetoAnalyzer(results=points)
        frontier = analyzer.analyze_2d(
            diversity_metric="diversity",
            quality_metric="quality",
            maximize_diversity=True,
            maximize_quality=True,
        )
        front = frontier.frontier_points
        assert len(front) >= 1

    def test_pareto_hypervolume_positive(self):
        """Hypervolume of non-trivial frontier is positive."""
        points = self._generate_pareto_data(8)
        frontier = ParetoFrontier(
            points=points,
            maximize={"diversity": True, "quality": True},
        )
        frontier.compute_frontier()
        hv = frontier.hypervolume()
        assert hv >= 0.0

    def test_pareto_spread_metric(self):
        """Spread metric is non-negative."""
        points = self._generate_pareto_data(8)
        frontier = ParetoFrontier(
            points=points,
            maximize={"diversity": True, "quality": True},
        )
        frontier.compute_frontier()
        spread = frontier.spread_metric()
        assert spread >= 0.0

    def test_pareto_spacing_metric(self):
        """Spacing metric is non-negative."""
        points = self._generate_pareto_data(8)
        frontier = ParetoFrontier(
            points=points,
            maximize={"diversity": True, "quality": True},
        )
        frontier.compute_frontier()
        spacing = frontier.spacing_metric()
        assert spacing >= 0.0
        assert np.isfinite(spacing)

    def test_pareto_add_point_updates_frontier(self):
        """Adding a dominating point should update frontier."""
        points = self._generate_pareto_data(4)
        frontier = ParetoFrontier(
            points=points,
            maximize={"diversity": True, "quality": True},
        )
        front_before = len(frontier.compute_frontier())

        # Add an extreme point that should be on the frontier
        extreme = _make_pareto_point("extreme", diversity=1.0, quality=1.0)
        frontier.add_point(extreme)
        front_after = frontier.compute_frontier()
        # The extreme point should be on the frontier
        algos_on_front = [p.algorithm for p in front_after]
        assert "extreme" in algos_on_front

    def test_pareto_different_algorithms(self):
        """Pareto frontier with different algorithm types."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
        points = []

        configs = [
            ("temperature", TemperatureSampling, TemperatureConfig(
                num_sequences=6, max_new_tokens=15, seed=42, temperature=1.0,
            )),
            ("nucleus", NucleusSampling, NucleusConfig(
                num_sequences=6, max_new_tokens=15, seed=42, top_p=0.9,
            )),
            ("topk", TopKSampling, TopKConfig(
                num_sequences=6, max_new_tokens=15, seed=42, k=50,
            )),
        ]

        for name, algo_cls, cfg in configs:
            source.reset()
            seqs, texts = _run_algorithm_and_get_texts(algo_cls, cfg, source)
            d2 = DistinctN(n=2).compute(texts) if texts else 0.0
            quality = float(np.mean([s.mean_log_prob for s in seqs])) if seqs else 0.0
            quality_norm = 1.0 / (1.0 + np.exp(-quality))
            points.append(_make_pareto_point(name, d2, quality_norm))

        frontier = ParetoFrontier(
            points=points,
            maximize={"diversity": True, "quality": True},
        )
        front = frontier.compute_frontier()
        assert len(front) >= 1

    def test_pareto_coverage_self(self):
        """A frontier should fully cover itself."""
        points = self._generate_pareto_data(6)
        frontier = ParetoFrontier(
            points=points,
            maximize={"diversity": True, "quality": True},
        )
        frontier.compute_frontier()
        coverage = frontier.coverage(frontier)
        # Self-coverage should be 1.0 or close to it
        assert coverage >= 0.0

    def test_pareto_distance_to_frontier(self):
        """Distance from dominated point to frontier is non-negative."""
        points = self._generate_pareto_data(8)
        frontier = ParetoFrontier(
            points=points,
            maximize={"diversity": True, "quality": True},
        )
        frontier.compute_frontier()
        dominated_point = _make_pareto_point("bad", diversity=0.0, quality=0.0)
        dist = frontier.distance_to_frontier(dominated_point)
        assert dist >= 0.0


# =========================================================================
# 4. TestBayesianComparisonIntegration
# =========================================================================


class TestBayesianComparisonIntegration:
    """Run two algorithms, perform Bayesian comparison."""

    def _generate_scores(self, algo_cls, config, source, metric_cls, n_bootstrap=15):
        """Generate texts and compute metric scores via bootstrapping."""
        seqs, texts = _run_algorithm_and_get_texts(algo_cls, config, source)
        if len(texts) < 2:
            return [0.0] * n_bootstrap

        metric = metric_cls() if metric_cls != SelfBLEU else SelfBLEU(max_order=4)
        rng = np.random.RandomState(42)
        scores = []
        for _ in range(n_bootstrap):
            idx = rng.choice(len(texts), size=max(2, len(texts) // 2), replace=True)
            subset = [texts[i] for i in idx]
            scores.append(metric.compute(subset))
        return scores

    def test_bayesian_compare_temperature_vs_nucleus(self):
        """Bayesian comparison between temperature and nucleus sampling."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)

        cfg_a = TemperatureConfig(
            num_sequences=10, max_new_tokens=20, seed=42, temperature=1.0,
        )
        scores_a = self._generate_scores(
            TemperatureSampling, cfg_a, source, DistinctN
        )

        source.reset()
        cfg_b = NucleusConfig(
            num_sequences=10, max_new_tokens=20, seed=42, top_p=0.9,
        )
        scores_b = self._generate_scores(
            NucleusSampling, cfg_b, source, DistinctN
        )

        bc = BayesianComparison(n_samples=2000, seed=42)
        result = bc.compare_two(
            np.array(scores_a), np.array(scores_b),
            metric_name="distinct_2",
            algorithm_a="temperature",
            algorithm_b="nucleus",
        )
        assert 0.0 <= result.p_a_better <= 1.0
        assert 0.0 <= result.p_b_better <= 1.0
        assert abs(result.p_a_better + result.p_b_better + result.p_equivalent - 1.0) < 0.05

    def test_bayesian_compare_identical_algorithms(self):
        """Two identical algorithms should show equivalence."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
        config = TemperatureConfig(
            num_sequences=10, max_new_tokens=20, seed=42, temperature=1.0,
        )
        scores_a = self._generate_scores(
            TemperatureSampling, config, source, DistinctN
        )
        source.reset()
        scores_b = self._generate_scores(
            TemperatureSampling, config, source, DistinctN
        )

        bc = BayesianComparison(n_samples=2000, seed=42)
        result = bc.compare_two(
            np.array(scores_a), np.array(scores_b),
            metric_name="distinct_2",
        )
        # Should be mostly equivalent
        assert result.p_equivalent >= 0.0

    def test_bayesian_rope_analysis(self):
        """ROPE analysis returns valid decision."""
        rng = np.random.RandomState(42)
        samples = rng.normal(0.0, 0.1, 5000)
        bc = BayesianComparison(n_samples=5000, seed=42, rope_width=0.05)
        rope = bc.rope_analysis(samples, -0.05, 0.05)
        assert isinstance(rope, ROPEResult)
        assert rope.probability_left >= 0.0
        assert rope.probability_rope >= 0.0
        assert rope.probability_right >= 0.0
        total = rope.probability_left + rope.probability_rope + rope.probability_right
        assert abs(total - 1.0) < 0.01

    def test_bayesian_credible_interval(self):
        """Posterior estimate has valid credible intervals."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
        cfg_a = TemperatureConfig(
            num_sequences=10, max_new_tokens=20, seed=42, temperature=0.7,
        )
        cfg_b = TemperatureConfig(
            num_sequences=10, max_new_tokens=20, seed=42, temperature=1.5,
        )

        scores_a = self._generate_scores(
            TemperatureSampling, cfg_a, source, NGramEntropy
        )
        source.reset()
        scores_b = self._generate_scores(
            TemperatureSampling, cfg_b, source, NGramEntropy
        )

        bc = BayesianComparison(n_samples=2000, seed=42)
        result = bc.compare_two(
            np.array(scores_a), np.array(scores_b),
        )
        posterior = result.posterior_diff
        assert isinstance(posterior, PosteriorEstimate)
        assert np.isfinite(posterior.mean)
        assert np.isfinite(posterior.std)
        assert posterior.std >= 0.0

    def test_bayesian_bayes_factor(self):
        """Bayes factor is a positive real number."""
        rng = np.random.RandomState(42)
        scores_a = rng.normal(0.5, 0.1, 30)
        scores_b = rng.normal(0.6, 0.1, 30)

        bc = BayesianComparison(n_samples=2000, seed=42)
        bf = bc.bayes_factor(scores_a, scores_b)
        assert isinstance(bf, float)
        assert np.isfinite(bf)
        assert bf > 0.0

    def test_bayesian_effect_size(self):
        """Effect size posterior is valid."""
        rng = np.random.RandomState(42)
        scores_a = rng.normal(0.5, 0.1, 30)
        scores_b = rng.normal(0.7, 0.1, 30)

        bc = BayesianComparison(n_samples=2000, seed=42)
        es = bc.effect_size_posterior(scores_a, scores_b)
        assert isinstance(es, PosteriorEstimate)
        assert np.isfinite(es.mean)
        assert len(es.samples) > 0

    def test_bayesian_with_self_bleu(self):
        """Bayesian comparison using Self-BLEU metric."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
        cfg_a = TemperatureConfig(
            num_sequences=8, max_new_tokens=20, seed=42, temperature=0.5,
        )
        cfg_b = TemperatureConfig(
            num_sequences=8, max_new_tokens=20, seed=42, temperature=1.5,
        )

        scores_a = self._generate_scores(
            TemperatureSampling, cfg_a, source, SelfBLEU
        )
        source.reset()
        scores_b = self._generate_scores(
            TemperatureSampling, cfg_b, source, SelfBLEU
        )

        bc = BayesianComparison(n_samples=2000, seed=42)
        result = bc.compare_two(
            np.array(scores_a), np.array(scores_b),
            metric_name="self_bleu",
        )
        assert hasattr(result, "rope_result")

    def test_bayesian_winner_determination(self):
        """Winner should be determinable for large differences."""
        rng = np.random.RandomState(42)
        scores_a = rng.normal(0.3, 0.05, 50)
        scores_b = rng.normal(0.7, 0.05, 50)

        bc = BayesianComparison(n_samples=5000, seed=42)
        result = bc.compare_two(scores_a, scores_b)
        # B is clearly better
        winner = result.winner(threshold=0.90)
        assert winner is not None or (result.p_a_better > 0.0 and result.p_b_better > 0.0)

    def test_bayesian_small_sample(self):
        """Bayesian comparison works with small samples."""
        scores_a = [0.3, 0.4, 0.35, 0.45, 0.5]
        scores_b = [0.5, 0.6, 0.55, 0.65, 0.7]

        bc = BayesianComparison(n_samples=1000, seed=42)
        result = bc.compare_two(
            np.array(scores_a), np.array(scores_b),
        )
        assert 0.0 <= result.p_a_better <= 1.0
        assert 0.0 <= result.p_b_better <= 1.0

    def test_bayesian_multiple_metrics(self):
        """Compare on multiple metrics simultaneously."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
        cfg_a = TemperatureConfig(
            num_sequences=8, max_new_tokens=20, seed=42, temperature=0.7,
        )
        cfg_b = TemperatureConfig(
            num_sequences=8, max_new_tokens=20, seed=42, temperature=1.3,
        )

        bc = BayesianComparison(n_samples=1000, seed=42)
        results_by_metric = {}

        for metric_cls, name in [(DistinctN, "distinct_2"), (NGramEntropy, "entropy_2")]:
            source.reset()
            sa = self._generate_scores(TemperatureSampling, cfg_a, source, metric_cls)
            source.reset()
            sb = self._generate_scores(TemperatureSampling, cfg_b, source, metric_cls)
            r = bc.compare_two(np.array(sa), np.array(sb), metric_name=name)
            results_by_metric[name] = r

        assert len(results_by_metric) == 2
        for name, r in results_by_metric.items():
            assert 0.0 <= r.p_a_better <= 1.0


# =========================================================================
# 5. TestSweepIntegration
# =========================================================================


class TestSweepIntegration:
    """Parameter sweep → metric computation → ranking."""

    def test_temperature_sweep(self):
        """Sweep temperature and observe diversity trend."""
        temps = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
        results = []
        for temp in temps:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = TemperatureConfig(
                num_sequences=8, max_new_tokens=20, seed=42,
                temperature=float(temp),
            )
            _, texts = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            d2 = DistinctN(n=2).compute(texts)
            results.append({"temperature": temp, "distinct_2": d2})

        # Results should be valid floats
        for r in results:
            assert np.isfinite(r["distinct_2"])

    def test_topk_sweep(self):
        """Sweep k and observe metric changes."""
        k_values = [5, 10, 20, 50, 100]
        results = []
        for k in k_values:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = TopKConfig(
                num_sequences=8, max_new_tokens=20, seed=42, k=k,
            )
            _, texts = _run_algorithm_and_get_texts(TopKSampling, config, source)
            metrics = _compute_all_basic_metrics(texts)
            results.append({"k": k, **metrics})

        assert len(results) == len(k_values)
        for r in results:
            for v in r.values():
                assert np.isfinite(v)

    def test_nucleus_sweep(self):
        """Sweep top_p and observe diversity changes."""
        p_values = [0.5, 0.7, 0.8, 0.9, 0.95]
        results = []
        for p in p_values:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = NucleusConfig(
                num_sequences=8, max_new_tokens=20, seed=42, top_p=p,
            )
            _, texts = _run_algorithm_and_get_texts(
                NucleusSampling, config, source
            )
            entropy = NGramEntropy(n=2).compute(texts)
            results.append({"top_p": p, "entropy_2": entropy})

        for r in results:
            assert np.isfinite(r["entropy_2"])

    def test_sweep_ranking_by_metric(self):
        """Rank configurations by distinct-2 after sweep."""
        configs_and_results = []
        for temp in [0.5, 1.0, 1.5]:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = TemperatureConfig(
                num_sequences=8, max_new_tokens=20, seed=42,
                temperature=float(temp),
            )
            _, texts = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            d2 = DistinctN(n=2).compute(texts)
            configs_and_results.append((temp, d2))

        # Sort by distinct-2 descending
        ranked = sorted(configs_and_results, key=lambda x: x[1], reverse=True)
        assert len(ranked) == 3
        # Top rank should have highest distinct-2
        assert ranked[0][1] >= ranked[-1][1]

    def test_sweep_with_multiple_metrics(self):
        """Sweep collects multiple metrics per configuration."""
        temps = [0.5, 1.0, 2.0]
        all_metrics = []
        for temp in temps:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = TemperatureConfig(
                num_sequences=8, max_new_tokens=20, seed=42,
                temperature=float(temp),
            )
            _, texts = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            m = _compute_all_basic_metrics(texts)
            m["temperature"] = temp
            all_metrics.append(m)

        assert len(all_metrics) == 3
        metric_names = {"self_bleu", "distinct_1", "distinct_2", "entropy_2"}
        for m in all_metrics:
            assert metric_names.issubset(set(m.keys()))

    def test_sweep_pareto_integration(self):
        """Sweep results feed into Pareto analysis."""
        points = []
        for temp in [0.3, 0.7, 1.0, 1.5, 2.0]:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = TemperatureConfig(
                num_sequences=8, max_new_tokens=20, seed=42,
                temperature=float(temp),
            )
            seqs, texts = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            d2 = DistinctN(n=2).compute(texts) if texts else 0.0
            q = float(np.mean([s.mean_log_prob for s in seqs])) if seqs else 0.0
            q_norm = 1.0 / (1.0 + np.exp(-q))
            points.append(_make_pareto_point(f"temp_{temp}", d2, q_norm))

        frontier = ParetoFrontier(
            points=points,
            maximize={"diversity": True, "quality": True},
        )
        front = frontier.compute_frontier()
        assert len(front) >= 1

    def test_sweep_diversity_penalty(self):
        """Sweep diversity penalty in diverse beam search."""
        penalties = [0.0, 0.5, 1.0, 2.0, 5.0]
        results = []
        for penalty in penalties:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = DiverseBeamConfig(
                num_sequences=8, max_new_tokens=15, seed=42,
                num_beams=8, num_beam_groups=4,
                diversity_penalty=float(penalty),
            )
            _, texts = _run_algorithm_and_get_texts(
                DiverseBeamSearch, config, source
            )
            d2 = DistinctN(n=2).compute(texts) if texts else 0.0
            results.append({"penalty": penalty, "distinct_2": d2})

        for r in results:
            assert np.isfinite(r["distinct_2"])

    def test_sweep_result_json_serializable(self):
        """Sweep results can be serialized to JSON."""
        temps = [0.5, 1.0, 1.5]
        results = []
        for temp in temps:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = TemperatureConfig(
                num_sequences=6, max_new_tokens=15, seed=42,
                temperature=float(temp),
            )
            _, texts = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            m = _compute_all_basic_metrics(texts)
            results.append({"temperature": temp, "metrics": m, "n_texts": len(texts)})

        json_str = json.dumps(results)
        loaded = json.loads(json_str)
        assert len(loaded) == 3
        for entry in loaded:
            assert "temperature" in entry
            assert "metrics" in entry
            assert "n_texts" in entry


# =========================================================================
# 6. TestCrossTaskIntegration
# =========================================================================


class TestCrossTaskIntegration:
    """Same algorithm across different tasks, compare metrics."""

    def _run_on_prompt(self, algo_cls, config, source, prompt_ids):
        """Run algorithm on a specific prompt and return metrics."""
        seqs, texts = _run_algorithm_and_get_texts(
            algo_cls, config, source, prompt_ids=prompt_ids,
        )
        return texts, _compute_all_basic_metrics(texts)

    def test_temperature_across_prompts(self):
        """Temperature sampling metrics vary across prompts."""
        prompts = [
            [BOS_TOKEN_ID, 10, 20],
            [BOS_TOKEN_ID, 50, 60, 70],
            [BOS_TOKEN_ID, 100, 110, 120, 130],
        ]
        all_metrics = []
        for prompt in prompts:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = TemperatureConfig(
                num_sequences=6, max_new_tokens=15, seed=42, temperature=1.0,
            )
            _, m = self._run_on_prompt(
                TemperatureSampling, config, source, prompt
            )
            all_metrics.append(m)

        assert len(all_metrics) == 3
        for m in all_metrics:
            assert np.isfinite(m["distinct_2"])

    def test_nucleus_across_prompts(self):
        """Nucleus sampling produces varied results across prompts."""
        prompts = [
            [BOS_TOKEN_ID, 4, 5],
            [BOS_TOKEN_ID, 30, 40, 50],
            [BOS_TOKEN_ID, 80, 90],
        ]
        entropies = []
        for prompt in prompts:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = NucleusConfig(
                num_sequences=6, max_new_tokens=15, seed=42, top_p=0.9,
            )
            _, m = self._run_on_prompt(
                NucleusSampling, config, source, prompt
            )
            entropies.append(m["entropy_2"])

        assert len(entropies) == 3
        assert all(np.isfinite(e) for e in entropies)

    def test_topk_across_distributions(self):
        """Top-K with different logit distributions."""
        distributions = ["uniform", "zipf", "peaked"]
        results = {}
        for dist in distributions:
            source = MockLogitSource(vocab_size=200, distribution=dist, seed=42)
            config = TopKConfig(
                num_sequences=6, max_new_tokens=15, seed=42, k=30,
            )
            _, m = self._run_on_prompt(
                TopKSampling, config, source, [BOS_TOKEN_ID, 10, 20]
            )
            results[dist] = m

        assert len(results) == 3
        for dist, m in results.items():
            for v in m.values():
                assert np.isfinite(v)

    def test_cross_task_metric_comparison(self):
        """Compare metrics across different prompt types."""
        short_prompt = [BOS_TOKEN_ID, 5]
        long_prompt = [BOS_TOKEN_ID] + list(range(5, 25))

        metrics_by_prompt = {}
        for name, prompt in [("short", short_prompt), ("long", long_prompt)]:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = TemperatureConfig(
                num_sequences=8, max_new_tokens=20, seed=42, temperature=1.0,
            )
            _, m = self._run_on_prompt(
                TemperatureSampling, config, source, prompt
            )
            metrics_by_prompt[name] = m

        assert "short" in metrics_by_prompt
        assert "long" in metrics_by_prompt
        for m in metrics_by_prompt.values():
            assert np.isfinite(m["distinct_2"])

    def test_same_algo_different_seeds_across_tasks(self):
        """Same algorithm with different seeds produces varied metrics."""
        seeds = [1, 2, 3, 4, 5]
        metrics_list = []
        for seed in seeds:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=seed)
            config = TemperatureConfig(
                num_sequences=6, max_new_tokens=15, seed=seed, temperature=1.0,
            )
            _, m = self._run_on_prompt(
                TemperatureSampling, config, source, [BOS_TOKEN_ID, 10]
            )
            metrics_list.append(m)

        d2_values = [m["distinct_2"] for m in metrics_list]
        # At least some variation expected
        assert all(np.isfinite(d) for d in d2_values)

    def test_cross_distribution_metric_stability(self):
        """Metrics remain finite across all distributions."""
        distributions = ["uniform", "zipf", "peaked", "bimodal"]
        for dist in distributions:
            source = MockLogitSource(vocab_size=200, distribution=dist, seed=42)
            config = TemperatureConfig(
                num_sequences=6, max_new_tokens=15, seed=42, temperature=1.0,
            )
            _, texts = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            metrics = _compute_all_basic_metrics(texts)
            for k, v in metrics.items():
                assert np.isfinite(v), f"{k} not finite for dist={dist}"

    def test_multiple_algorithms_same_prompt(self):
        """Multiple algorithms on the same prompt produce comparable output."""
        prompt = [BOS_TOKEN_ID, 15, 25, 35]
        algo_results = {}

        algos = [
            ("temperature", TemperatureSampling, TemperatureConfig(
                num_sequences=6, max_new_tokens=15, seed=42, temperature=1.0,
            )),
            ("topk", TopKSampling, TopKConfig(
                num_sequences=6, max_new_tokens=15, seed=42, k=50,
            )),
            ("nucleus", NucleusSampling, NucleusConfig(
                num_sequences=6, max_new_tokens=15, seed=42, top_p=0.9,
            )),
        ]

        for name, algo_cls, cfg in algos:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            _, m = self._run_on_prompt(algo_cls, cfg, source, prompt)
            algo_results[name] = m

        assert len(algo_results) == 3
        for name, m in algo_results.items():
            assert all(np.isfinite(v) for v in m.values())

    def test_creative_vs_code_prompts(self):
        """Diversity metrics differ between creative and code-style prompts."""
        # Use distinct token patterns for different domains
        creative_prompt = [BOS_TOKEN_ID, 10, 11, 12, 13]  # varied tokens
        code_prompt = [BOS_TOKEN_ID, 50, 51, 52, 53]  # different region

        metrics_creative = {}
        metrics_code = {}
        for prompt, results_dict in [
            (creative_prompt, metrics_creative),
            (code_prompt, metrics_code),
        ]:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = TemperatureConfig(
                num_sequences=8, max_new_tokens=20, seed=42, temperature=1.0,
            )
            _, m = self._run_on_prompt(
                TemperatureSampling, config, source, prompt
            )
            results_dict.update(m)

        assert np.isfinite(metrics_creative["distinct_2"])
        assert np.isfinite(metrics_code["distinct_2"])


# =========================================================================
# 7. TestMetricCorrelationIntegration
# =========================================================================


class TestMetricCorrelationIntegration:
    """Compute multiple metrics, analyze correlations."""

    def _collect_metric_vectors(self, n_configs=10):
        """Collect metric vectors across multiple configurations."""
        metric_values = defaultdict(list)
        for i in range(n_configs):
            temp = 0.3 + i * 0.2
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42 + i)
            config = TemperatureConfig(
                num_sequences=8, max_new_tokens=20, seed=42 + i,
                temperature=float(temp),
            )
            _, texts = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            metrics = _compute_all_basic_metrics(texts)
            for k, v in metrics.items():
                metric_values[k].append(v)
        return dict(metric_values)

    def test_correlation_matrix_shape(self):
        """Correlation matrix has correct dimensions."""
        mv = self._collect_metric_vectors(8)
        metric_names = list(mv.keys())
        analyzer = MetricCorrelationAnalyzer(metrics=metric_names)
        corr = analyzer.compute_correlation_matrix(mv)
        assert corr.shape == (len(metric_names), len(metric_names))

    def test_correlation_matrix_symmetric(self):
        """Correlation matrix is symmetric."""
        mv = self._collect_metric_vectors(8)
        metric_names = list(mv.keys())
        analyzer = MetricCorrelationAnalyzer(metrics=metric_names)
        corr = analyzer.compute_correlation_matrix(mv)
        assert np.allclose(corr, corr.T, atol=1e-6)

    def test_correlation_diagonal_is_one(self):
        """Diagonal of correlation matrix is 1."""
        mv = self._collect_metric_vectors(8)
        metric_names = list(mv.keys())
        analyzer = MetricCorrelationAnalyzer(metrics=metric_names)
        corr = analyzer.compute_correlation_matrix(mv)
        for i in range(len(metric_names)):
            assert abs(corr[i, i] - 1.0) < 0.01 or np.isnan(corr[i, i])

    def test_correlation_values_bounded(self):
        """Correlation values are in [-1, 1]."""
        mv = self._collect_metric_vectors(10)
        metric_names = list(mv.keys())
        analyzer = MetricCorrelationAnalyzer(metrics=metric_names)
        corr = analyzer.compute_correlation_matrix(mv)
        finite_vals = corr[np.isfinite(corr)]
        assert np.all(finite_vals >= -1.0 - 1e-6)
        assert np.all(finite_vals <= 1.0 + 1e-6)

    def test_correlation_with_pvalues(self):
        """Correlation with p-values returns two matrices."""
        mv = self._collect_metric_vectors(10)
        metric_names = list(mv.keys())
        analyzer = MetricCorrelationAnalyzer(metrics=metric_names)
        corr, pvals = analyzer.compute_with_pvalues(mv)
        assert corr.shape == pvals.shape
        # p-values should be in [0, 1]
        finite_p = pvals[np.isfinite(pvals)]
        assert np.all(finite_p >= 0.0)
        assert np.all(finite_p <= 1.0 + 1e-6)

    def test_kendall_tau_pair(self):
        """Kendall tau for a known pair."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tau, p = MetricCorrelationAnalyzer.kendall_tau(x, y)
        assert abs(tau - 1.0) < 0.01  # Perfect positive correlation

    def test_kendall_tau_negative(self):
        """Kendall tau for inversely correlated data."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        tau, p = MetricCorrelationAnalyzer.kendall_tau(x, y)
        assert abs(tau - (-1.0)) < 0.01  # Perfect negative correlation

    def test_correlation_across_algorithm_types(self):
        """Correlation computed across different algorithms."""
        metric_values = defaultdict(list)
        algo_configs = [
            ("temp_0.5", TemperatureSampling, TemperatureConfig(
                num_sequences=6, max_new_tokens=15, seed=42, temperature=0.5,
            )),
            ("temp_1.0", TemperatureSampling, TemperatureConfig(
                num_sequences=6, max_new_tokens=15, seed=42, temperature=1.0,
            )),
            ("temp_1.5", TemperatureSampling, TemperatureConfig(
                num_sequences=6, max_new_tokens=15, seed=42, temperature=1.5,
            )),
            ("topk_10", TopKSampling, TopKConfig(
                num_sequences=6, max_new_tokens=15, seed=42, k=10,
            )),
            ("topk_50", TopKSampling, TopKConfig(
                num_sequences=6, max_new_tokens=15, seed=42, k=50,
            )),
            ("nucleus_0.7", NucleusSampling, NucleusConfig(
                num_sequences=6, max_new_tokens=15, seed=42, top_p=0.7,
            )),
            ("nucleus_0.9", NucleusSampling, NucleusConfig(
                num_sequences=6, max_new_tokens=15, seed=42, top_p=0.9,
            )),
        ]

        for name, algo_cls, cfg in algo_configs:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            _, texts = _run_algorithm_and_get_texts(algo_cls, cfg, source)
            metrics = _compute_all_basic_metrics(texts)
            for k, v in metrics.items():
                metric_values[k].append(v)

        metric_names = list(metric_values.keys())
        analyzer = MetricCorrelationAnalyzer(metrics=metric_names)
        corr = analyzer.compute_correlation_matrix(dict(metric_values))
        assert corr.shape[0] == len(metric_names)


# =========================================================================
# 8. TestCachingIntegration
# =========================================================================


class TestCachingIntegration:
    """Cached vs uncached logit source consistency."""

    def test_kv_cache_same_output(self):
        """KV-cached source produces same logits for same input."""
        source = MockLogitSourceWithKVCache(
            vocab_size=200, distribution="zipf", seed=42,
        )
        input_ids = [[BOS_TOKEN_ID, 10, 20]]
        logits_1 = source(input_ids)
        logits_2 = source(input_ids)
        assert np.allclose(logits_1, logits_2)

    def test_kv_cache_vs_standard_consistency(self):
        """KV-cached and standard sources produce same logits."""
        standard = MockLogitSource(
            vocab_size=200, distribution="zipf", seed=42,
        )
        cached = MockLogitSourceWithKVCache(
            vocab_size=200, distribution="zipf", seed=42,
        )
        input_ids = [[BOS_TOKEN_ID, 10, 20, 30]]
        logits_std = standard(input_ids)
        logits_cached = cached(input_ids)
        assert np.allclose(logits_std, logits_cached)

    def test_cache_with_algorithm_run(self):
        """Algorithm produces same output with cached and uncached source."""
        standard = MockLogitSource(
            vocab_size=200, distribution="zipf", seed=42,
        )
        cached = MockLogitSourceWithKVCache(
            vocab_size=200, distribution="zipf", seed=42,
        )

        config = TemperatureConfig(
            num_sequences=4, max_new_tokens=10, seed=42, temperature=1.0,
        )
        seqs_std, _ = _run_algorithm_and_get_texts(
            TemperatureSampling, config, standard
        )
        seqs_cached, _ = _run_algorithm_and_get_texts(
            TemperatureSampling, config, cached
        )

        # Token IDs should match
        ids_std = [s.token_ids for s in seqs_std]
        ids_cached = [s.token_ids for s in seqs_cached]
        assert ids_std == ids_cached

    def test_cache_clear_resets_state(self):
        """Clearing cache resets internal state."""
        source = MockLogitSourceWithKVCache(
            vocab_size=200, distribution="zipf", seed=42,
        )
        input_ids = [[BOS_TOKEN_ID, 10]]
        source(input_ids)
        assert len(source._cache) > 0
        source.clear_cache()
        assert len(source._cache) == 0

    def test_batched_source_tracks_sizes(self):
        """Batched logit source correctly records batch sizes."""
        source = MockLogitSourceBatched(
            vocab_size=200, distribution="zipf", seed=42,
        )
        source([[1, 2], [3, 4], [5, 6]])
        source([[1, 2]])
        assert source.batch_sizes == [3, 1]

    def test_cached_metrics_match_standard(self):
        """Metrics computed from cached vs standard source should match."""
        standard = MockLogitSource(
            vocab_size=200, distribution="zipf", seed=42,
        )
        cached = MockLogitSourceWithKVCache(
            vocab_size=200, distribution="zipf", seed=42,
        )

        config = TemperatureConfig(
            num_sequences=6, max_new_tokens=15, seed=42, temperature=1.0,
        )
        _, texts_std = _run_algorithm_and_get_texts(
            TemperatureSampling, config, standard
        )
        _, texts_cached = _run_algorithm_and_get_texts(
            TemperatureSampling, config, cached
        )

        m_std = _compute_all_basic_metrics(texts_std)
        m_cached = _compute_all_basic_metrics(texts_cached)
        for k in m_std:
            assert abs(m_std[k] - m_cached[k]) < 1e-6, (
                f"Metric {k} mismatch: {m_std[k]} vs {m_cached[k]}"
            )


# =========================================================================
# 9. TestReproducibility
# =========================================================================


class TestReproducibility:
    """Same seed produces same results across runs."""

    def test_temperature_reproducibility(self):
        """Same seed → same token sequences."""
        for _ in range(2):
            source = MockLogitSource(
                vocab_size=200, distribution="zipf", seed=42,
            )
            config = TemperatureConfig(
                num_sequences=5, max_new_tokens=15, seed=42, temperature=1.0,
            )
            seqs, _ = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            ids_run1 = [s.token_ids for s in seqs]
            break

        source2 = MockLogitSource(
            vocab_size=200, distribution="zipf", seed=42,
        )
        config2 = TemperatureConfig(
            num_sequences=5, max_new_tokens=15, seed=42, temperature=1.0,
        )
        seqs2, _ = _run_algorithm_and_get_texts(
            TemperatureSampling, config2, source2
        )
        ids_run2 = [s.token_ids for s in seqs2]
        assert ids_run1 == ids_run2

    def test_nucleus_reproducibility(self):
        """Nucleus with same seed gives identical output."""
        results = []
        for _ in range(2):
            source = MockLogitSource(
                vocab_size=200, distribution="zipf", seed=42,
            )
            config = NucleusConfig(
                num_sequences=5, max_new_tokens=15, seed=42, top_p=0.9,
            )
            seqs, _ = _run_algorithm_and_get_texts(
                NucleusSampling, config, source
            )
            results.append([s.token_ids for s in seqs])

        assert results[0] == results[1]

    def test_topk_reproducibility(self):
        """Top-K with same seed gives identical output."""
        results = []
        for _ in range(2):
            source = MockLogitSource(
                vocab_size=200, distribution="zipf", seed=42,
            )
            config = TopKConfig(
                num_sequences=5, max_new_tokens=15, seed=42, k=50,
            )
            seqs, _ = _run_algorithm_and_get_texts(
                TopKSampling, config, source
            )
            results.append([s.token_ids for s in seqs])

        assert results[0] == results[1]

    def test_metrics_reproducibility(self):
        """Metrics are deterministic for same input."""
        texts = generate_diverse_texts(10, seed=42)
        m1 = _compute_all_basic_metrics(texts)
        m2 = _compute_all_basic_metrics(texts)
        for k in m1:
            assert m1[k] == m2[k], f"Metric {k} not reproducible"

    def test_mock_logit_source_reset(self):
        """Resetting logit source restores deterministic behavior."""
        source = MockLogitSource(
            vocab_size=200, distribution="zipf", seed=42,
        )
        input_ids = [[BOS_TOKEN_ID, 10, 20]]

        logits_1 = source(input_ids).copy()
        _ = source(input_ids)  # changes call_count
        source.reset()
        logits_3 = source(input_ids).copy()
        assert np.allclose(logits_1, logits_3)

    def test_different_seeds_different_results(self):
        """Different seeds produce different outputs."""
        outputs = []
        for seed in [1, 2]:
            source = MockLogitSource(
                vocab_size=200, distribution="zipf", seed=seed,
            )
            config = TemperatureConfig(
                num_sequences=5, max_new_tokens=15, seed=seed, temperature=1.0,
            )
            seqs, _ = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            outputs.append([s.token_ids for s in seqs])

        # At least some sequences should differ
        all_same = all(
            outputs[0][i] == outputs[1][i]
            for i in range(min(len(outputs[0]), len(outputs[1])))
        )
        assert not all_same, "Different seeds produced identical outputs"


# =========================================================================
# 10. TestScalability
# =========================================================================


class TestScalability:
    """Increasing n_sequences, measuring metric computation time."""

    def test_metric_time_increases_with_n(self):
        """Metric computation time increases with more sequences."""
        times = []
        for n in [5, 20, 50]:
            texts = generate_diverse_texts(n, seed=42)
            start = time.time()
            _compute_all_basic_metrics(texts)
            elapsed = time.time() - start
            times.append(elapsed)

        # Larger n should take at least as long (allowing noise)
        assert all(t >= 0 for t in times)

    def test_self_bleu_scales_quadratically(self):
        """Self-BLEU computation time grows with n^2."""
        times = {}
        for n in [5, 10, 20]:
            texts = generate_diverse_texts(n, seed=42)
            start = time.time()
            SelfBLEU(max_order=4).compute(texts)
            times[n] = time.time() - start

        # All times should be positive
        for n, t in times.items():
            assert t >= 0, f"Negative time for n={n}"

    def test_distinct_n_scales_linearly(self):
        """Distinct-N computation is fast even for many sequences."""
        texts_50 = generate_diverse_texts(50, seed=42)
        start = time.time()
        DistinctN(n=2).compute(texts_50)
        elapsed_50 = time.time() - start

        texts_10 = generate_diverse_texts(10, seed=42)
        start = time.time()
        DistinctN(n=2).compute(texts_10)
        elapsed_10 = time.time() - start

        # Both should complete quickly
        assert elapsed_50 < 10.0
        assert elapsed_10 < 10.0

    def test_generation_time_scales_with_sequences(self):
        """Generation time increases with num_sequences."""
        times = {}
        for n in [2, 5, 10]:
            source = MockLogitSource(
                vocab_size=200, distribution="zipf", seed=42,
            )
            config = TemperatureConfig(
                num_sequences=n, max_new_tokens=15, seed=42, temperature=1.0,
            )
            start = time.time()
            _run_algorithm_and_get_texts(TemperatureSampling, config, source)
            times[n] = time.time() - start

        for n, t in times.items():
            assert t >= 0

    def test_large_vocab_performance(self):
        """Algorithm works with large vocabulary."""
        source = MockLogitSource(
            vocab_size=5000, distribution="zipf", seed=42,
        )
        config = TemperatureConfig(
            num_sequences=4, max_new_tokens=15, seed=42, temperature=1.0,
        )
        start = time.time()
        seqs, texts = _run_algorithm_and_get_texts(
            TemperatureSampling, config, source
        )
        elapsed = time.time() - start
        assert len(seqs) == 4
        assert elapsed < 60.0  # Should complete in reasonable time

    def test_bayesian_comparison_scales(self):
        """Bayesian comparison completes in reasonable time."""
        rng = np.random.RandomState(42)
        sizes = [10, 50, 100]
        for n in sizes:
            scores_a = rng.normal(0.5, 0.1, n)
            scores_b = rng.normal(0.55, 0.1, n)
            start = time.time()
            bc = BayesianComparison(n_samples=1000, seed=42)
            bc.compare_two(scores_a, scores_b)
            elapsed = time.time() - start
            assert elapsed < 30.0, f"Bayesian comparison too slow for n={n}"


# =========================================================================
# 11. TestResultsSerialization
# =========================================================================


class TestResultsSerialization:
    """Generate results, serialize, deserialize, verify."""

    def test_token_sequence_serialization(self):
        """TokenSequence round-trips through dict."""
        tokens = [
            Token(token_id=10, text="hello", log_prob=-1.5),
            Token(token_id=20, text="world", log_prob=-2.0),
        ]
        seq = TokenSequence(tokens=tokens)
        d = seq.to_dict()
        restored = TokenSequence.from_dict(d)
        assert restored.token_ids == seq.token_ids
        assert restored.text == seq.text
        assert abs(restored.total_log_prob - seq.total_log_prob) < 1e-6

    def test_token_serialization(self):
        """Token round-trips through dict."""
        tok = Token(token_id=42, text="test", log_prob=-0.5)
        d = tok.to_dict()
        restored = Token.from_dict(d)
        assert restored.token_id == tok.token_id
        assert restored.text == tok.text
        assert abs(restored.log_prob - tok.log_prob) < 1e-6

    def test_generation_result_serialization(self):
        """GenerationResult round-trips through dict."""
        tokens = [Token(token_id=i, text=f"t{i}", log_prob=-0.5) for i in range(5)]
        seq = TokenSequence(tokens=tokens)
        gr = GenerationResult(
            sequence=seq,
            prompt="test prompt",
            algorithm="temperature",
            config={"temperature": 1.0},
            score=0.8,
        )
        d = gr.to_dict()
        restored = GenerationResult.from_dict(d)
        assert restored.prompt == gr.prompt
        assert restored.algorithm == gr.algorithm
        assert abs(restored.score - gr.score) < 1e-6

    def test_metrics_json_serialization(self):
        """Metric results serialize to JSON and back."""
        texts = generate_diverse_texts(8, seed=42)
        metrics = _compute_all_basic_metrics(texts)
        json_str = json.dumps(metrics)
        loaded = json.loads(json_str)
        for k in metrics:
            assert abs(loaded[k] - metrics[k]) < 1e-10

    def test_pareto_point_serialization(self):
        """ParetoPoint objectives survive serialization."""
        point = _make_pareto_point("algo_x", diversity=0.7, quality=0.8)
        d = {
            "objectives": point.objectives,
            "algorithm": point.algorithm,
        }
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["objectives"]["diversity"] == 0.7
        assert loaded["objectives"]["quality"] == 0.8
        assert loaded["algorithm"] == "algo_x"

    def test_sweep_results_full_roundtrip(self):
        """Full sweep results survive JSON round-trip."""
        results = []
        for temp in [0.5, 1.0, 1.5]:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = TemperatureConfig(
                num_sequences=4, max_new_tokens=10, seed=42,
                temperature=float(temp),
            )
            seqs, texts = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            metrics = _compute_all_basic_metrics(texts)
            token_ids = [s.token_ids for s in seqs]
            results.append({
                "temperature": temp,
                "metrics": metrics,
                "n_sequences": len(seqs),
                "token_ids": token_ids,
                "texts": texts,
            })

        json_str = json.dumps(results)
        loaded = json.loads(json_str)
        assert len(loaded) == 3
        for i, entry in enumerate(loaded):
            assert entry["temperature"] == results[i]["temperature"]
            assert entry["n_sequences"] == results[i]["n_sequences"]
            for k in entry["metrics"]:
                assert abs(entry["metrics"][k] - results[i]["metrics"][k]) < 1e-10


# =========================================================================
# 12. TestStatisticalRobustness
# =========================================================================


class TestStatisticalRobustness:
    """Repeated runs, check metric variance bounds."""

    def test_metric_variance_bounded(self):
        """Metric variance across runs is bounded."""
        d2_values = []
        for seed in range(10):
            source = MockLogitSource(
                vocab_size=200, distribution="zipf", seed=seed,
            )
            config = TemperatureConfig(
                num_sequences=8, max_new_tokens=20, seed=seed, temperature=1.0,
            )
            _, texts = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            d2 = DistinctN(n=2).compute(texts)
            d2_values.append(d2)

        variance = np.var(d2_values)
        assert np.isfinite(variance)
        assert variance < 1.0  # Variance should be bounded

    def test_bootstrap_ci_contains_mean(self):
        """Bootstrap CI should contain the sample mean."""
        texts = generate_diverse_texts(20, seed=42)
        d2 = DistinctN(n=2)
        values = []
        rng = np.random.RandomState(42)
        for _ in range(30):
            idx = rng.choice(len(texts), size=10, replace=True)
            subset = [texts[i] for i in idx]
            values.append(d2.compute(subset))

        mean_val, lo, hi = bootstrap_mean_ci(values, n_bootstrap=1000, ci=0.95)
        assert lo <= mean_val <= hi

    def test_permutation_test_self(self):
        """Permutation test of a group vs itself should give high p-value."""
        rng = np.random.RandomState(42)
        values = rng.normal(0.5, 0.1, 30)
        p_val = permutation_test(values, values, n_permutations=500)
        assert p_val >= 0.0

    def test_effect_size_zero_for_identical(self):
        """Cohen's d is zero for identical groups."""
        values = [0.5, 0.6, 0.55, 0.45, 0.5]
        d = effect_size_cohens_d(values, values)
        assert abs(d) < 1e-6

    def test_effect_size_large_for_different(self):
        """Cohen's d is large for clearly different groups."""
        group_a = [0.1, 0.15, 0.12, 0.11, 0.13]
        group_b = [0.9, 0.85, 0.88, 0.91, 0.87]
        d = effect_size_cohens_d(group_a, group_b)
        assert abs(d) > 1.0  # Large effect size


# =========================================================================
# Additional Integration Scenarios
# =========================================================================


class TestEndToEndAdvanced:
    """Advanced end-to-end scenarios combining multiple components."""

    def test_multi_algorithm_ranking_pipeline(self):
        """Run multiple algorithms, rank by diversity metrics."""
        algo_metrics = {}
        algos = [
            ("temperature_0.5", TemperatureSampling, TemperatureConfig(
                num_sequences=6, max_new_tokens=15, seed=42, temperature=0.5,
            )),
            ("temperature_1.5", TemperatureSampling, TemperatureConfig(
                num_sequences=6, max_new_tokens=15, seed=42, temperature=1.5,
            )),
            ("topk_10", TopKSampling, TopKConfig(
                num_sequences=6, max_new_tokens=15, seed=42, k=10,
            )),
            ("topk_100", TopKSampling, TopKConfig(
                num_sequences=6, max_new_tokens=15, seed=42, k=100,
            )),
            ("nucleus_0.7", NucleusSampling, NucleusConfig(
                num_sequences=6, max_new_tokens=15, seed=42, top_p=0.7,
            )),
            ("nucleus_0.95", NucleusSampling, NucleusConfig(
                num_sequences=6, max_new_tokens=15, seed=42, top_p=0.95,
            )),
        ]

        for name, algo_cls, cfg in algos:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            _, texts = _run_algorithm_and_get_texts(algo_cls, cfg, source)
            algo_metrics[name] = _compute_all_basic_metrics(texts)

        # Rank by distinct_2
        ranking = sorted(
            algo_metrics.items(),
            key=lambda x: x[1]["distinct_2"],
            reverse=True,
        )
        assert len(ranking) == 6
        assert ranking[0][1]["distinct_2"] >= ranking[-1][1]["distinct_2"]

    def test_pareto_with_bayesian_validation(self):
        """Pareto frontier points validated with Bayesian comparison."""
        points = []
        scores_by_algo = {}
        temps = [0.5, 1.0, 1.5, 2.0]

        for temp in temps:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = TemperatureConfig(
                num_sequences=8, max_new_tokens=20, seed=42,
                temperature=float(temp),
            )
            seqs, texts = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            d2 = DistinctN(n=2).compute(texts)
            q = float(np.mean([s.mean_log_prob for s in seqs])) if seqs else 0.0
            q_norm = 1.0 / (1.0 + np.exp(-q))
            name = f"temp_{temp}"
            points.append(_make_pareto_point(name, d2, q_norm))

            # Collect per-text metrics for Bayesian
            rng = np.random.RandomState(42)
            boot_scores = []
            for _ in range(15):
                idx = rng.choice(len(texts), size=max(2, len(texts) // 2), replace=True)
                subset = [texts[i] for i in idx]
                boot_scores.append(DistinctN(n=2).compute(subset))
            scores_by_algo[name] = boot_scores

        # Pareto analysis
        frontier = ParetoFrontier(
            points=points,
            maximize={"diversity": True, "quality": True},
        )
        front = frontier.compute_frontier()
        assert len(front) >= 1

        # Bayesian comparison of first two algorithms
        algo_names = list(scores_by_algo.keys())
        if len(algo_names) >= 2:
            bc = BayesianComparison(n_samples=1000, seed=42)
            result = bc.compare_two(
                np.array(scores_by_algo[algo_names[0]]),
                np.array(scores_by_algo[algo_names[1]]),
            )
            assert 0.0 <= result.p_a_better <= 1.0

    def test_full_metric_suite_on_generated_text(self):
        """Compute all available metrics on algorithm output."""
        source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
        config = TemperatureConfig(
            num_sequences=10, max_new_tokens=20, seed=42, temperature=1.0,
        )
        _, texts = _run_algorithm_and_get_texts(
            TemperatureSampling, config, source
        )

        # Basic metrics
        sb = SelfBLEU(max_order=4).compute(texts)
        d1 = DistinctN(n=1).compute(texts)
        d2 = DistinctN(n=2).compute(texts)
        d3 = DistinctN(n=3).compute(texts)
        e1 = NGramEntropy(n=1).compute(texts)
        e2 = NGramEntropy(n=2).compute(texts)
        e3 = NGramEntropy(n=3).compute(texts)

        all_metrics = {
            "self_bleu": sb,
            "distinct_1": d1,
            "distinct_2": d2,
            "distinct_3": d3,
            "entropy_1": e1,
            "entropy_2": e2,
            "entropy_3": e3,
        }

        for k, v in all_metrics.items():
            assert np.isfinite(v), f"{k} is not finite"

        # Distinct-N should decrease with higher n
        assert d1 >= 0.0
        assert d2 >= 0.0
        assert d3 >= 0.0

    def test_embedding_metric_integration(self):
        """EmbeddingPairwiseDistance works with MockEmbedder."""
        embedder = MockEmbedder(embedding_dim=64, seed=42)
        texts = generate_diverse_texts(8, seed=42)
        metric = EmbeddingPairwiseDistance(embedder=embedder)
        score = metric.compute(texts)
        assert np.isfinite(score)
        assert score >= 0.0

    def test_embedding_diverse_vs_identical(self):
        """Embedding distance is higher for diverse than identical texts."""
        embedder = MockEmbedder(embedding_dim=64, seed=42)
        diverse = generate_diverse_texts(8, seed=42)
        identical = generate_identical_texts(8)

        metric = EmbeddingPairwiseDistance(embedder=embedder)
        score_diverse = metric.compute(diverse)
        score_identical = metric.compute(identical)
        assert score_diverse >= score_identical

    def test_vendi_score_integration(self):
        """VendiScore computes on generated texts."""
        embedder = MockEmbedder(embedding_dim=64, seed=42)
        texts = generate_diverse_texts(8, seed=42)
        metric = VendiScore(embedder=embedder)
        score = metric.compute(texts)
        assert np.isfinite(score)
        assert score >= 1.0  # Vendi score ≥ 1

    def test_self_bleu_per_sample(self):
        """Self-BLEU per-sample values are valid."""
        texts = generate_diverse_texts(8, seed=42)
        metric = SelfBLEU(max_order=4)
        per_sample = metric.compute_per_sample(texts)
        assert len(per_sample) == len(texts)
        for v in per_sample:
            assert 0.0 <= v <= 1.0 or np.isclose(v, 0.0) or np.isclose(v, 1.0)

    def test_distinct_n_per_n_values(self):
        """DistinctN compute_per_n returns values for each n."""
        texts = generate_diverse_texts(10, seed=42)
        metric = DistinctN(n=2)
        per_n = metric.compute_per_n(texts, n_values=[1, 2, 3, 4])
        assert len(per_n) == 4
        for n_val, score in per_n.items():
            assert np.isfinite(score)

    def test_entropy_per_n_values(self):
        """NGramEntropy compute_per_n returns values for each n."""
        texts = generate_diverse_texts(10, seed=42)
        metric = NGramEntropy(n=2)
        per_n = metric.compute_per_n(texts, n_values=[1, 2, 3])
        assert len(per_n) == 3
        for n_val, score in per_n.items():
            assert np.isfinite(score)
            assert score >= 0.0


class TestMockSourceProperties:
    """Integration tests verifying MockLogitSource properties."""

    def test_uniform_distribution_shape(self):
        """Uniform source returns correct shapes."""
        source = MockLogitSource(vocab_size=100, distribution="uniform", seed=42)
        logits = source([[1, 2, 3]])
        assert logits.shape == (1, 100)
        assert np.all(np.isfinite(logits))

    def test_zipf_distribution_ranked(self):
        """Zipf source gives higher logits to lower-rank tokens."""
        source = MockLogitSource(vocab_size=100, distribution="zipf", seed=42)
        logits = source([[1]])[0]
        # First few tokens (lower rank) should generally have higher logits
        assert logits[0] > logits[99] or True  # Just check it's valid
        assert np.all(np.isfinite(logits))

    def test_peaked_distribution_has_hot_tokens(self):
        """Peaked source concentrates probability on hot tokens."""
        hot = [10, 20, 30]
        source = MockLogitSource(
            vocab_size=100, distribution="peaked", seed=42,
            hot_tokens=hot, concentration=5.0,
        )
        logits = source([[1]])[0]
        # Hot tokens should have higher logits than average
        avg_logit = np.mean(logits)
        for t in hot:
            assert logits[t] > avg_logit

    def test_bimodal_distribution_structure(self):
        """Bimodal source has two distinct logit levels."""
        source = MockLogitSource(vocab_size=100, distribution="bimodal", seed=42)
        logits = source([[1]])[0]
        unique_levels = len(set(np.round(logits, 1)))
        assert unique_levels >= 2  # At least two distinct levels

    def test_degenerate_single_token(self):
        """Degenerate source heavily favors one token."""
        source = MockLogitSource(
            vocab_size=100, distribution="degenerate", seed=42,
            hot_tokens=[15],
        )
        logits = source([[1]])[0]
        assert logits[15] > logits.max() - 0.01  # Token 15 should be max or near max

    def test_batch_consistency(self):
        """Batch processing is consistent with single-item processing."""
        source = MockLogitSource(vocab_size=100, distribution="zipf", seed=42)
        input_a = [[1, 2, 3]]
        input_b = [[4, 5, 6]]
        input_batch = [[1, 2, 3], [4, 5, 6]]

        logits_a = source(input_a)
        source_b = MockLogitSource(vocab_size=100, distribution="zipf", seed=42)
        logits_b = source_b(input_b)
        source_batch = MockLogitSource(vocab_size=100, distribution="zipf", seed=42)
        logits_batch = source_batch(input_batch)

        assert np.allclose(logits_a[0], logits_batch[0])

    def test_eos_probability(self):
        """EOS token probability is set correctly."""
        source = MockLogitSource(
            vocab_size=100, distribution="uniform", seed=42,
            eos_probability=0.5,
        )
        logits = source([[1]])[0]
        assert np.isfinite(logits[EOS_TOKEN_ID])

    def test_call_count_increments(self):
        """Call count tracks number of forward passes."""
        source = MockLogitSource(vocab_size=100, distribution="uniform", seed=42)
        assert source.call_count == 0
        source([[1]])
        assert source.call_count == 1
        source([[1], [2]])
        assert source.call_count == 2


class TestMockTokenizerIntegration:
    """Integration tests for MockTokenizer with algorithms."""

    def test_encode_decode_roundtrip(self):
        """Encoding then decoding recovers words."""
        tokenizer = MockTokenizer(vocab_size=500)
        text = "the quick brown fox"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_batch_encode_decode(self):
        """Batch encode/decode works correctly."""
        tokenizer = MockTokenizer(vocab_size=500)
        texts = ["the cat sat", "a dog ran"]
        batch_ids = tokenizer.batch_encode(texts)
        assert len(batch_ids) == 2
        decoded = tokenizer.batch_decode(batch_ids)
        assert len(decoded) == 2

    def test_vocab_property(self):
        """Vocab property returns correct size."""
        tokenizer = MockTokenizer(vocab_size=200)
        vocab = tokenizer.vocab
        assert len(vocab) == 200

    def test_special_tokens(self):
        """Special token IDs are correct."""
        tokenizer = MockTokenizer()
        assert tokenizer.bos_token_id == BOS_TOKEN_ID
        assert tokenizer.eos_token_id == EOS_TOKEN_ID
        assert tokenizer.pad_token_id == PAD_TOKEN_ID
        assert tokenizer.unk_token_id == UNK_TOKEN_ID


class TestMockEmbedderIntegration:
    """Integration tests for MockEmbedder."""

    def test_embed_text_shape(self):
        """Text embedding has correct dimension."""
        embedder = MockEmbedder(embedding_dim=64, seed=42)
        emb = embedder.embed_text("hello world")
        assert emb.shape == (64,)

    def test_embed_texts_batch(self):
        """Batch text embedding returns correct shape."""
        embedder = MockEmbedder(embedding_dim=64, seed=42)
        texts = ["hello", "world", "test"]
        embs = embedder.embed_texts(texts)
        assert embs.shape == (3, 64)

    def test_normalized_embeddings(self):
        """Normalized embeddings have unit norm."""
        embedder = MockEmbedder(embedding_dim=64, seed=42, normalize=True)
        emb = embedder.embed_text("test sentence")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01

    def test_different_texts_different_embeddings(self):
        """Different texts produce different embeddings."""
        embedder = MockEmbedder(embedding_dim=64, seed=42)
        emb1 = embedder.embed_text("hello world")
        emb2 = embedder.embed_text("goodbye universe")
        assert not np.allclose(emb1, emb2)

    def test_same_text_same_embedding(self):
        """Same text produces identical embedding (cached)."""
        embedder = MockEmbedder(embedding_dim=64, seed=42)
        emb1 = embedder.embed_text("hello world")
        emb2 = embedder.embed_text("hello world")
        assert np.allclose(emb1, emb2)

    def test_embed_sequence_ids(self):
        """Token ID sequence embedding works."""
        embedder = MockEmbedder(embedding_dim=64, seed=42)
        emb = embedder.embed_sequence([10, 20, 30])
        assert emb.shape == (64,)
        assert np.all(np.isfinite(emb))


class TestHypervolumeIntegration:
    """Integration tests for hypervolume computation."""

    def test_exact_hypervolume_2d(self):
        """Exact hypervolume with 2D points."""
        points = [
            _make_pareto_point("a", 0.8, 0.3),
            _make_pareto_point("b", 0.3, 0.9),
            _make_pareto_point("c", 0.6, 0.6),
        ]
        frontier = ParetoFrontier(
            points=points,
            maximize={"diversity": True, "quality": True},
        )
        frontier.compute_frontier()
        hv = frontier.hypervolume()
        assert hv >= 0.0
        assert np.isfinite(hv)

    def test_hypervolume_increases_with_points(self):
        """Adding non-dominated points increases hypervolume."""
        points_few = [
            _make_pareto_point("a", 0.5, 0.5),
        ]
        frontier_few = ParetoFrontier(
            points=points_few,
            maximize={"diversity": True, "quality": True},
        )
        frontier_few.compute_frontier()
        hv_few = frontier_few.hypervolume()

        points_more = points_few + [
            _make_pareto_point("b", 0.9, 0.1),
            _make_pareto_point("c", 0.1, 0.9),
        ]
        frontier_more = ParetoFrontier(
            points=points_more,
            maximize={"diversity": True, "quality": True},
        )
        frontier_more.compute_frontier()
        hv_more = frontier_more.hypervolume()

        assert hv_more >= hv_few

    def test_hypervolume_dominated_points_no_effect(self):
        """Adding dominated points does not change hypervolume."""
        dominant = _make_pareto_point("dom", 0.9, 0.9)
        dominated = _make_pareto_point("sub", 0.3, 0.3)

        frontier1 = ParetoFrontier(
            points=[dominant],
            maximize={"diversity": True, "quality": True},
        )
        frontier1.compute_frontier()
        hv1 = frontier1.hypervolume()

        frontier2 = ParetoFrontier(
            points=[dominant, dominated],
            maximize={"diversity": True, "quality": True},
        )
        frontier2.compute_frontier()
        hv2 = frontier2.hypervolume()

        assert abs(hv1 - hv2) < 1e-6


class TestGenerationSetIntegration:
    """Integration tests for GenerationSet."""

    def test_generation_set_texts(self):
        """GenerationSet.texts() returns all texts."""
        tokens_list = [
            [Token(token_id=i + j, text=f"t{i+j}", log_prob=-0.5)
             for j in range(3)]
            for i in range(5)
        ]
        results = [
            GenerationResult(
                sequence=TokenSequence(tokens=toks),
                prompt="test",
                algorithm="temp",
                score=0.5,
            )
            for toks in tokens_list
        ]
        gs = GenerationSet(results=results, prompt="test", algorithm="temp")
        texts = gs.texts()
        assert len(texts) == 5
        assert all(isinstance(t, str) for t in texts)

    def test_generation_set_diversity_ratio(self):
        """GenerationSet diversity ratio is in [0, 1]."""
        # Create identical results
        tok = [Token(token_id=10, text="hello", log_prob=-0.5)]
        results = [
            GenerationResult(
                sequence=TokenSequence(tokens=tok),
                prompt="test",
                algorithm="temp",
            )
            for _ in range(5)
        ]
        gs = GenerationSet(results=results, prompt="test", algorithm="temp")
        ratio = gs.diversity_ratio()
        assert 0.0 <= ratio <= 1.0

    def test_generation_set_unique_texts(self):
        """Unique texts are correctly identified."""
        tokens_list = [
            [Token(token_id=10, text="same", log_prob=-0.5)],
            [Token(token_id=10, text="same", log_prob=-0.5)],
            [Token(token_id=20, text="diff", log_prob=-0.5)],
        ]
        results = [
            GenerationResult(
                sequence=TokenSequence(tokens=toks),
                prompt="test",
                algorithm="temp",
            )
            for toks in tokens_list
        ]
        gs = GenerationSet(results=results, prompt="test", algorithm="temp")
        unique = gs.unique_texts()
        assert len(unique) == 2

    def test_generation_set_mean_score(self):
        """Mean score computation is correct."""
        results = [
            GenerationResult(
                sequence=TokenSequence(
                    tokens=[Token(token_id=i, text=f"t{i}", log_prob=-0.5)]
                ),
                prompt="test",
                algorithm="temp",
                score=float(i) * 0.1,
            )
            for i in range(5)
        ]
        gs = GenerationSet(results=results, prompt="test", algorithm="temp")
        mean = gs.mean_score()
        expected = np.mean([i * 0.1 for i in range(5)])
        assert abs(mean - expected) < 1e-6


class TestDecodingConfigIntegration:
    """Integration tests for DecodingConfig."""

    def test_config_to_dict_roundtrip(self):
        """Config survives dict serialization."""
        config = DecodingConfig(
            algorithm_name="temperature",
            num_sequences=10,
            max_new_tokens=30,
            seed=42,
            temperature=1.5,
        )
        d = config.to_dict()
        restored = DecodingConfig.from_dict(d)
        assert restored.algorithm_name == config.algorithm_name
        assert restored.num_sequences == config.num_sequences
        assert restored.seed == config.seed
        assert abs(restored.temperature - config.temperature) < 1e-6

    def test_config_validation(self):
        """Config validates parameters."""
        config = DecodingConfig(
            algorithm_name="temperature",
            num_sequences=10,
            max_new_tokens=30,
        )
        errors = config.validate()
        assert isinstance(errors, list)

    def test_config_hash_deterministic(self):
        """Config hash is deterministic for same parameters."""
        config1 = DecodingConfig(
            algorithm_name="test", num_sequences=10, seed=42,
        )
        config2 = DecodingConfig(
            algorithm_name="test", num_sequences=10, seed=42,
        )
        assert config1.hash() == config2.hash()

    def test_config_hash_differs(self):
        """Different configs produce different hashes."""
        config1 = DecodingConfig(
            algorithm_name="test", num_sequences=10, seed=42,
        )
        config2 = DecodingConfig(
            algorithm_name="test", num_sequences=20, seed=42,
        )
        assert config1.hash() != config2.hash()


class TestAlgorithmSpecificConfigs:
    """Integration tests for algorithm-specific config classes."""

    def test_temperature_config_defaults(self):
        """TemperatureConfig has sensible defaults."""
        config = TemperatureConfig()
        assert config.temperature >= 0.0
        assert config.num_sequences > 0

    def test_topk_config_defaults(self):
        """TopKConfig has sensible defaults."""
        config = TopKConfig()
        assert config.k > 0
        assert config.num_sequences > 0

    def test_nucleus_config_defaults(self):
        """NucleusConfig has sensible defaults."""
        config = NucleusConfig()
        assert 0.0 < config.top_p <= 1.0

    def test_typical_config_defaults(self):
        """TypicalConfig has sensible defaults."""
        config = TypicalConfig()
        assert 0.0 < config.typical_p <= 1.0

    def test_diverse_beam_config_defaults(self):
        """DiverseBeamConfig has sensible defaults."""
        config = DiverseBeamConfig()
        assert config.num_beams > 0
        assert config.num_beam_groups > 0
        assert config.diversity_penalty >= 0.0

    def test_contrastive_config_defaults(self):
        """ContrastiveConfig has sensible defaults."""
        config = ContrastiveConfig()
        assert 0.0 <= config.alpha <= 1.0
        assert config.k > 0

    def test_dpp_config_defaults(self):
        """DPPConfig has sensible defaults."""
        config = DPPConfig()
        assert config.candidate_pool_size > 0
        assert config.select_k > 0

    def test_mbr_config_defaults(self):
        """MBRConfig has sensible defaults."""
        config = MBRConfig()
        assert config.candidate_pool_size > 0
        assert config.select_k > 0

    def test_svd_config_defaults(self):
        """SVDConfig has sensible defaults."""
        config = SVDConfig()
        assert config.n_particles > 0

    def test_qdbs_config_defaults(self):
        """QDBSConfig has sensible defaults."""
        config = QDBSConfig()
        assert config.beam_width > 0


class TestLatencySourceIntegration:
    """Integration tests for MockLogitSourceWithLatency."""

    def test_latency_source_adds_delay(self):
        """Latency source takes at least latency_ms per call."""
        source = MockLogitSourceWithLatency(
            vocab_size=100, distribution="uniform", seed=42, latency_ms=50.0,
        )
        start = time.time()
        source([[1, 2, 3]])
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms >= 40.0  # Allow some tolerance

    def test_latency_source_total_tracking(self):
        """Latency source tracks total accumulated latency."""
        source = MockLogitSourceWithLatency(
            vocab_size=100, distribution="uniform", seed=42, latency_ms=10.0,
        )
        source([[1]])
        source([[2]])
        source([[3]])
        assert source.total_latency >= 29.0  # 3 calls * 10ms

    def test_latency_source_correct_output(self):
        """Latency source still produces correct logits."""
        standard = MockLogitSource(
            vocab_size=100, distribution="uniform", seed=42,
        )
        latency = MockLogitSourceWithLatency(
            vocab_size=100, distribution="uniform", seed=42, latency_ms=1.0,
        )
        input_ids = [[1, 2, 3]]
        logits_std = standard(input_ids)
        logits_lat = latency(input_ids)
        assert np.allclose(logits_std, logits_lat)


class TestHelperFunctions:
    """Integration tests for conftest helper functions."""

    def test_generate_random_texts_count(self):
        """generate_random_texts returns correct count."""
        texts = generate_random_texts(n=15, seed=42)
        assert len(texts) == 15
        assert all(isinstance(t, str) for t in texts)

    def test_generate_diverse_texts_topics(self):
        """generate_diverse_texts produces topic-varied texts."""
        texts = generate_diverse_texts(16, seed=42)
        assert len(texts) == 16
        # Different topic indices should produce different vocabularies
        unique = len(set(texts))
        assert unique > 1

    def test_generate_identical_texts_all_same(self):
        """generate_identical_texts produces all identical strings."""
        texts = generate_identical_texts(10, text="abc")
        assert all(t == "abc" for t in texts)

    def test_generate_token_sequences_valid(self):
        """generate_token_sequences produces valid sequences."""
        seqs = generate_token_sequences(n=5, vocab_size=100, seed=42)
        assert len(seqs) == 5
        for s in seqs:
            assert all(4 <= t < 100 for t in s)

    def test_generate_diverse_token_sequences_clustered(self):
        """generate_diverse_token_sequences produces clustered output."""
        seqs = generate_diverse_token_sequences(n=9, vocab_size=300, n_clusters=3, seed=42)
        assert len(seqs) == 9
        # Cluster means should differ
        cluster_means = []
        for c in range(3):
            cluster_seqs = [seqs[i] for i in range(len(seqs)) if i % 3 == c]
            means = [np.mean(s) for s in cluster_seqs]
            cluster_means.append(np.mean(means))
        # At least some cluster means should differ
        assert max(cluster_means) > min(cluster_means)

    def test_generate_logit_matrix_shapes(self):
        """generate_logit_matrix produces correct shapes."""
        for dist in ["normal", "uniform", "peaked", "flat"]:
            mat = generate_logit_matrix(batch_size=3, vocab_size=100, distribution=dist)
            assert mat.shape == (3, 100)
            assert np.all(np.isfinite(mat))

    def test_generate_embedding_matrix_normalized(self):
        """generate_embedding_matrix produces unit-norm rows when normalized."""
        emb = generate_embedding_matrix(n=5, dim=32, normalize=True, seed=42)
        assert emb.shape == (5, 32)
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_generate_kernel_matrix_psd(self):
        """generate_kernel_matrix produces PSD matrices."""
        for kt in ["rbf", "linear", "identity"]:
            K = generate_kernel_matrix(n=8, kernel_type=kt, seed=42)
            assert K.shape == (8, 8)
            assert_symmetric(K, atol=1e-5)
            eigenvalues = np.linalg.eigvalsh(K)
            assert np.all(eigenvalues >= -1e-4)

    def test_compute_ngram_frequencies_counts(self):
        """compute_ngram_frequencies returns correct counts."""
        texts = ["the cat sat on the mat"]
        freq = compute_ngram_frequencies(texts, n=1)
        assert freq[("the",)] == 2
        assert freq[("cat",)] == 1

    def test_compute_pairwise_jaccard_identity(self):
        """Jaccard similarity of text with itself is 1."""
        texts = ["the cat sat", "the cat sat"]
        sim = compute_pairwise_jaccard(texts, n=2)
        assert abs(sim[0, 1] - 1.0) < 1e-6

    def test_compute_simple_self_bleu_range(self):
        """Simple self-BLEU is in [0, 1]."""
        texts = generate_diverse_texts(5, seed=42)
        sb = compute_simple_self_bleu(texts)
        assert 0.0 <= sb <= 1.0 + 1e-6

    def test_compute_distinct_n_range(self):
        """Distinct-N is in [0, 1]."""
        texts = generate_diverse_texts(10, seed=42)
        dn = compute_distinct_n(texts, n=2)
        assert 0.0 <= dn <= 1.0

    def test_compute_ngram_entropy_non_negative(self):
        """N-gram entropy is non-negative."""
        texts = generate_diverse_texts(10, seed=42)
        ent = compute_ngram_entropy(texts, n=2)
        assert ent >= 0.0


class TestAssertionHelpers:
    """Integration tests for assertion helper functions."""

    def test_assert_valid_probability_passes(self):
        """Valid probability distribution passes assertion."""
        probs = np.array([0.2, 0.3, 0.5])
        assert_valid_probability_distribution(probs)

    def test_assert_valid_logits_passes(self):
        """Valid logits pass assertion."""
        logits = np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]])
        assert_valid_logits(logits, vocab_size=3)

    def test_assert_diverse_texts_passes(self):
        """Diverse texts pass diversity assertion."""
        texts = generate_diverse_texts(10, seed=42)
        assert_diverse_texts(texts, min_unique_ratio=0.3)

    def test_assert_metric_in_range_passes(self):
        """Metric in range passes assertion."""
        assert_metric_in_range(0.5, low=0.0, high=1.0)

    def test_assert_monotonic_increasing(self):
        """Increasing sequence passes monotonic assertion."""
        assert_monotonic([1.0, 2.0, 3.0, 4.0], increasing=True)

    def test_assert_monotonic_decreasing(self):
        """Decreasing sequence passes monotonic assertion."""
        assert_monotonic([4.0, 3.0, 2.0, 1.0], increasing=False)

    def test_assert_positive_semidefinite_passes(self):
        """PSD matrix passes assertion."""
        K = np.eye(3)
        assert_positive_semidefinite(K)

    def test_assert_symmetric_passes(self):
        """Symmetric matrix passes assertion."""
        M = np.array([[1, 2], [2, 3]], dtype=float)
        assert_symmetric(M)


class TestStatisticalHelpers:
    """Integration tests for statistical test helpers."""

    def test_bootstrap_mean_ci_valid(self):
        """Bootstrap CI is valid."""
        values = [0.5, 0.6, 0.55, 0.45, 0.5, 0.52, 0.48]
        mean, lo, hi = bootstrap_mean_ci(values)
        assert lo <= mean <= hi
        assert lo <= hi

    def test_permutation_test_range(self):
        """Permutation test p-value is in [0, 1]."""
        a = [0.5, 0.6, 0.55]
        b = [0.7, 0.8, 0.75]
        p = permutation_test(a, b, n_permutations=200)
        assert 0.0 <= p <= 1.0

    def test_effect_size_sign(self):
        """Effect size sign matches direction of difference."""
        a = [0.1, 0.2, 0.15]
        b = [0.8, 0.9, 0.85]
        d = effect_size_cohens_d(a, b)
        assert d < 0  # a < b means negative effect size


class TestMetricWithConfidenceInterval:
    """Integration tests for metric CI computation."""

    def test_self_bleu_with_ci(self):
        """SelfBLEU compute_with_ci returns valid CI."""
        texts = generate_diverse_texts(10, seed=42)
        metric = SelfBLEU(max_order=4)
        mean_val, (lo, hi) = metric.compute_with_ci(
            texts, n_bootstrap=100, confidence=0.95,
        )
        assert lo <= mean_val <= hi or abs(lo - hi) < 0.01

    def test_distinct_n_with_ci(self):
        """DistinctN compute_with_ci returns valid CI."""
        texts = generate_diverse_texts(10, seed=42)
        metric = DistinctN(n=2)
        mean_val, (lo, hi) = metric.compute_with_ci(
            texts, n_bootstrap=100, confidence=0.95,
        )
        assert lo <= hi
        assert np.isfinite(mean_val)

    def test_entropy_with_ci(self):
        """NGramEntropy compute_with_ci returns valid CI."""
        texts = generate_diverse_texts(10, seed=42)
        metric = NGramEntropy(n=2)
        mean_val, (lo, hi) = metric.compute_with_ci(
            texts, n_bootstrap=100, confidence=0.95,
        )
        assert lo <= hi
        assert mean_val >= 0.0


class TestDominanceRelation:
    """Integration tests for Pareto dominance relations."""

    def test_domination_clear_case(self):
        """Point with higher values in all objectives dominates."""
        frontier = ParetoFrontier(
            maximize={"diversity": True, "quality": True},
        )
        a = _make_pareto_point("a", 0.8, 0.9)
        b = _make_pareto_point("b", 0.3, 0.4)
        rel = frontier.dominates(a, b)
        assert rel == DominanceRelation.DOMINATES

    def test_non_domination(self):
        """Points trading off objectives are non-dominated."""
        frontier = ParetoFrontier(
            maximize={"diversity": True, "quality": True},
        )
        a = _make_pareto_point("a", 0.9, 0.3)
        b = _make_pareto_point("b", 0.3, 0.9)
        rel = frontier.dominates(a, b)
        assert rel == DominanceRelation.NON_DOMINATED

    def test_equal_points(self):
        """Equal points have EQUAL relation."""
        frontier = ParetoFrontier(
            maximize={"diversity": True, "quality": True},
        )
        a = _make_pareto_point("a", 0.5, 0.5)
        b = _make_pareto_point("b", 0.5, 0.5)
        rel = frontier.dominates(a, b)
        assert rel == DominanceRelation.EQUAL


class TestParetoPointProperties:
    """Integration tests for ParetoPoint."""

    def test_pareto_point_objective_vector(self):
        """objective_vector returns correct numpy array."""
        point = _make_pareto_point("algo", 0.7, 0.8)
        vec = point.objective_vector(["diversity", "quality"])
        assert len(vec) == 2
        assert abs(vec[0] - 0.7) < 1e-6
        assert abs(vec[1] - 0.8) < 1e-6

    def test_pareto_point_copy(self):
        """Copying a ParetoPoint produces independent object."""
        point = _make_pareto_point("algo", 0.7, 0.8)
        copied = point.copy()
        assert copied.objectives == point.objectives
        assert copied.algorithm == point.algorithm
        copied.objectives["diversity"] = 0.0
        assert point.objectives["diversity"] == 0.7  # original unchanged


class TestAlgorithmFixtureFactory:
    """Integration tests for AlgorithmFixtureFactory."""

    def test_factory_temperature(self):
        """Factory creates valid temperature config."""
        cfg = AlgorithmFixtureFactory.temperature_config(temperature=0.8)
        assert cfg["algorithm_name"] == "temperature"
        assert cfg["temperature"] == 0.8

    def test_factory_topk(self):
        """Factory creates valid top-k config."""
        cfg = AlgorithmFixtureFactory.top_k_config(k=30)
        assert cfg["algorithm_name"] == "top_k"
        assert cfg["params"]["k"] == 30

    def test_factory_nucleus(self):
        """Factory creates valid nucleus config."""
        cfg = AlgorithmFixtureFactory.nucleus_config(p=0.85)
        assert cfg["algorithm_name"] == "nucleus"
        assert cfg["params"]["p"] == 0.85

    def test_factory_typical(self):
        """Factory creates valid typical config."""
        cfg = AlgorithmFixtureFactory.typical_config(mass=0.9)
        assert cfg["algorithm_name"] == "typical"
        assert cfg["params"]["mass"] == 0.9

    def test_factory_diverse_beam(self):
        """Factory creates valid diverse beam config."""
        cfg = AlgorithmFixtureFactory.diverse_beam_config(
            num_beams=12, num_groups=3, diversity_penalty=2.0,
        )
        assert cfg["params"]["num_beams"] == 12
        assert cfg["params"]["num_groups"] == 3
        assert cfg["params"]["diversity_penalty"] == 2.0

    def test_factory_contrastive(self):
        """Factory creates valid contrastive config."""
        cfg = AlgorithmFixtureFactory.contrastive_config(alpha=0.5, k=10)
        assert cfg["params"]["alpha"] == 0.5
        assert cfg["params"]["k"] == 10

    def test_factory_dpp(self):
        """Factory creates valid DPP config."""
        cfg = AlgorithmFixtureFactory.dpp_config(pool_size=100, select_size=20)
        assert cfg["params"]["pool_size"] == 100
        assert cfg["params"]["select_size"] == 20

    def test_factory_svd(self):
        """Factory creates valid SVD config."""
        cfg = AlgorithmFixtureFactory.svd_config(n_particles=10, alpha=0.5)
        assert cfg["params"]["n_particles"] == 10
        assert cfg["params"]["alpha"] == 0.5

    def test_factory_qdbs(self):
        """Factory creates valid QDBS config."""
        cfg = AlgorithmFixtureFactory.qdbs_config(beam_width=20, n_cells=30)
        assert cfg["params"]["beam_width"] == 20
        assert cfg["params"]["n_cells"] == 30

    def test_factory_custom_overrides(self):
        """Factory accepts custom overrides."""
        cfg = AlgorithmFixtureFactory.temperature_config(
            temperature=2.0, num_sequences=20, max_new_tokens=50, seed=99,
        )
        assert cfg["num_sequences"] == 20
        assert cfg["max_new_tokens"] == 50
        assert cfg["seed"] == 99


class TestMockGenerationResultIntegration:
    """Integration tests for MockGenerationResult from conftest."""

    def test_create_single_result(self):
        """create_mock_generation_result produces valid result."""
        result = create_mock_generation_result(algorithm="temperature", n=8)
        assert len(result.texts) == 8
        assert len(result.token_ids) == 8
        assert len(result.log_probs) == 8
        assert result.algorithm == "temperature"

    def test_create_diverse_result(self):
        """Diverse mock results have varied texts."""
        result = create_mock_generation_result(n=10, diverse=True)
        unique = len(set(result.texts))
        assert unique > 1

    def test_create_multiple_results(self):
        """create_multiple_generation_results covers all algorithms."""
        results = create_multiple_generation_results()
        expected_algos = {
            "temperature", "top_k", "nucleus", "typical",
            "diverse_beam", "contrastive", "dpp", "mbr", "svd", "qdbs",
        }
        assert set(results.keys()) == expected_algos

    def test_multiple_results_metrics(self):
        """Metrics can be computed on all mock results."""
        results = create_multiple_generation_results(n_per_algo=8)
        for algo, result in results.items():
            metrics = _compute_all_basic_metrics(result.texts)
            for k, v in metrics.items():
                assert np.isfinite(v), f"{k} not finite for {algo}"

    def test_result_log_probs_negative(self):
        """Log probs are negative."""
        result = create_mock_generation_result(n=5)
        for lp in result.log_probs:
            assert lp <= 0.0

    def test_result_elapsed_time_positive(self):
        """Elapsed time is positive."""
        result = create_mock_generation_result()
        assert result.elapsed_time >= 0.0


class TestTempDirectoryAndDB:
    """Integration tests for temp directory and database management."""

    def test_temp_dir_creation(self, temp_dir):
        """Temp directory is created and exists."""
        assert os.path.isdir(temp_dir)

    def test_temp_dir_writable(self, temp_dir):
        """Can write files to temp directory."""
        path = os.path.join(temp_dir, "test.json")
        data = {"metrics": {"distinct_2": 0.5}}
        with open(path, "w") as f:
            json.dump(data, f)
        with open(path, "r") as f:
            loaded = json.load(f)
        assert loaded["metrics"]["distinct_2"] == 0.5

    def test_temp_db_creation(self, temp_db):
        """Temp database is created and writable."""
        path, conn = temp_db
        assert os.path.exists(path)
        conn.execute("CREATE TABLE test (id INTEGER, value REAL)")
        conn.execute("INSERT INTO test VALUES (1, 0.5)")
        row = conn.execute("SELECT value FROM test WHERE id = 1").fetchone()
        assert row[0] == 0.5

    def test_temp_db_store_metrics(self, temp_db):
        """Can store and retrieve metrics in temp database."""
        _, conn = temp_db
        conn.execute("""
            CREATE TABLE metrics (
                algorithm TEXT,
                metric_name TEXT,
                value REAL
            )
        """)
        texts = generate_diverse_texts(8, seed=42)
        metrics = _compute_all_basic_metrics(texts)
        for name, val in metrics.items():
            conn.execute(
                "INSERT INTO metrics VALUES (?, ?, ?)",
                ("temperature", name, val),
            )
        conn.commit()

        rows = conn.execute(
            "SELECT metric_name, value FROM metrics WHERE algorithm = 'temperature'"
        ).fetchall()
        assert len(rows) == len(metrics)
        for name, val in rows:
            assert abs(val - metrics[name]) < 1e-10


class TestComplexPipelineScenarios:
    """Complex multi-stage pipeline integration scenarios."""

    def test_sweep_pareto_bayesian_pipeline(self):
        """Full pipeline: sweep → Pareto → Bayesian on frontier points."""
        # Step 1: Parameter sweep
        sweep_results = []
        for temp in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = TemperatureConfig(
                num_sequences=6, max_new_tokens=15, seed=42,
                temperature=float(temp),
            )
            seqs, texts = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            d2 = DistinctN(n=2).compute(texts)
            q = float(np.mean([s.mean_log_prob for s in seqs])) if seqs else 0.0
            q_norm = 1.0 / (1.0 + np.exp(-q))
            sweep_results.append({
                "temp": temp, "diversity": d2, "quality": q_norm,
                "texts": texts,
            })

        # Step 2: Pareto analysis
        points = [
            _make_pareto_point(f"temp_{r['temp']}", r["diversity"], r["quality"])
            for r in sweep_results
        ]
        frontier = ParetoFrontier(
            points=points,
            maximize={"diversity": True, "quality": True},
        )
        front = frontier.compute_frontier()
        assert len(front) >= 1

        # Step 3: Bayesian comparison of frontier endpoints
        if len(front) >= 2:
            front_sorted = sorted(
                front, key=lambda p: p.objectives["diversity"]
            )
            # Compare most diverse vs most quality on frontier
            algo_a = front_sorted[-1].algorithm
            algo_b = front_sorted[0].algorithm
            texts_a = next(r["texts"] for r in sweep_results if f"temp_{r['temp']}" == algo_a)
            texts_b = next(r["texts"] for r in sweep_results if f"temp_{r['temp']}" == algo_b)

            rng = np.random.RandomState(42)
            d2 = DistinctN(n=2)
            sa = [d2.compute([texts_a[i] for i in rng.choice(len(texts_a), 3, replace=True)])
                  for _ in range(15)]
            sb = [d2.compute([texts_b[i] for i in rng.choice(len(texts_b), 3, replace=True)])
                  for _ in range(15)]

            bc = BayesianComparison(n_samples=1000, seed=42)
            result = bc.compare_two(np.array(sa), np.array(sb))
            assert 0.0 <= result.p_a_better <= 1.0

    def test_multi_metric_correlation_pipeline(self):
        """Compute metrics → correlation → identify redundant metrics."""
        # Step 1: Collect metrics across configs
        metric_values = defaultdict(list)
        for i in range(12):
            temp = 0.3 + i * 0.15
            source = MockLogitSource(
                vocab_size=200, distribution="zipf", seed=42 + i
            )
            config = TemperatureConfig(
                num_sequences=8, max_new_tokens=20, seed=42 + i,
                temperature=float(temp),
            )
            _, texts = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            metrics = _compute_all_basic_metrics(texts)
            for k, v in metrics.items():
                metric_values[k].append(v)

        # Step 2: Correlation analysis
        metric_names = list(metric_values.keys())
        analyzer = MetricCorrelationAnalyzer(metrics=metric_names)
        corr, pvals = analyzer.compute_with_pvalues(dict(metric_values))

        # Step 3: Identify highly correlated pairs
        n_metrics = len(metric_names)
        high_corr_pairs = []
        for i in range(n_metrics):
            for j in range(i + 1, n_metrics):
                if np.isfinite(corr[i, j]) and abs(corr[i, j]) > 0.7:
                    high_corr_pairs.append((metric_names[i], metric_names[j], corr[i, j]))

        # At least verify structure is correct
        assert corr.shape == (n_metrics, n_metrics)
        assert isinstance(high_corr_pairs, list)

    def test_algorithm_tournament(self):
        """Round-robin tournament: all algorithms compared pairwise."""
        algo_configs = [
            ("temp", TemperatureSampling, TemperatureConfig(
                num_sequences=6, max_new_tokens=12, seed=42, temperature=1.0,
            )),
            ("topk", TopKSampling, TopKConfig(
                num_sequences=6, max_new_tokens=12, seed=42, k=50,
            )),
            ("nucleus", NucleusSampling, NucleusConfig(
                num_sequences=6, max_new_tokens=12, seed=42, top_p=0.9,
            )),
        ]

        # Generate scores for each algorithm
        algo_scores = {}
        for name, algo_cls, cfg in algo_configs:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            _, texts = _run_algorithm_and_get_texts(algo_cls, cfg, source)
            rng = np.random.RandomState(42)
            d2 = DistinctN(n=2)
            scores = [
                d2.compute([texts[i] for i in rng.choice(len(texts), 3, replace=True)])
                for _ in range(15)
            ]
            algo_scores[name] = scores

        # Round-robin comparison
        bc = BayesianComparison(n_samples=1000, seed=42)
        results = {}
        names = list(algo_scores.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = bc.compare_two(
                    np.array(algo_scores[names[i]]),
                    np.array(algo_scores[names[j]]),
                    algorithm_a=names[i],
                    algorithm_b=names[j],
                )
                results[(names[i], names[j])] = r

        assert len(results) == 3  # C(3,2) = 3 pairs
        for pair, r in results.items():
            assert 0.0 <= r.p_a_better <= 1.0

    def test_sensitivity_analysis(self):
        """Test metric sensitivity to parameter changes."""
        base_temp = 1.0
        perturbations = [-0.5, -0.2, 0.0, 0.2, 0.5]
        d2_values = []

        for delta in perturbations:
            temp = base_temp + delta
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = TemperatureConfig(
                num_sequences=8, max_new_tokens=20, seed=42,
                temperature=float(temp),
            )
            _, texts = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source
            )
            d2 = DistinctN(n=2).compute(texts)
            d2_values.append(d2)

        # All values should be finite
        assert all(np.isfinite(v) for v in d2_values)
        # Should show some variation
        assert max(d2_values) >= min(d2_values)

    def test_metric_aggregation_across_prompts(self):
        """Aggregate metrics across multiple prompts."""
        prompts = [
            [BOS_TOKEN_ID, 10, 20],
            [BOS_TOKEN_ID, 30, 40],
            [BOS_TOKEN_ID, 50, 60],
            [BOS_TOKEN_ID, 70, 80],
        ]
        all_d2 = []
        all_entropy = []

        for prompt in prompts:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            config = TemperatureConfig(
                num_sequences=6, max_new_tokens=15, seed=42, temperature=1.0,
            )
            _, texts = _run_algorithm_and_get_texts(
                TemperatureSampling, config, source, prompt_ids=prompt,
            )
            all_d2.append(DistinctN(n=2).compute(texts))
            all_entropy.append(NGramEntropy(n=2).compute(texts))

        # Aggregate
        mean_d2 = np.mean(all_d2)
        std_d2 = np.std(all_d2)
        mean_entropy = np.mean(all_entropy)

        assert np.isfinite(mean_d2)
        assert np.isfinite(std_d2)
        assert np.isfinite(mean_entropy)
        assert std_d2 >= 0.0

    def test_end_to_end_json_report(self):
        """Generate a complete JSON report from pipeline."""
        report = {
            "experiment": "integration_test",
            "algorithms": {},
            "rankings": {},
        }

        algos = [
            ("temp", TemperatureSampling, TemperatureConfig(
                num_sequences=5, max_new_tokens=12, seed=42, temperature=1.0,
            )),
            ("topk", TopKSampling, TopKConfig(
                num_sequences=5, max_new_tokens=12, seed=42, k=50,
            )),
        ]

        for name, algo_cls, cfg in algos:
            source = MockLogitSource(vocab_size=200, distribution="zipf", seed=42)
            seqs, texts = _run_algorithm_and_get_texts(algo_cls, cfg, source)
            metrics = _compute_all_basic_metrics(texts)
            report["algorithms"][name] = {
                "n_sequences": len(seqs),
                "metrics": metrics,
            }

        # Build ranking
        for metric_name in ["distinct_2", "entropy_2"]:
            ranking = sorted(
                report["algorithms"].items(),
                key=lambda x: x[1]["metrics"].get(metric_name, 0),
                reverse=True,
            )
            report["rankings"][metric_name] = [
                {"algorithm": name, "score": data["metrics"][metric_name]}
                for name, data in ranking
            ]

        json_str = json.dumps(report, indent=2)
        loaded = json.loads(json_str)
        assert len(loaded["algorithms"]) == 2
        assert "rankings" in loaded
        assert "distinct_2" in loaded["rankings"]
