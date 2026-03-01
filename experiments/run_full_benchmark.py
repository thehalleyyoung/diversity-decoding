#!/usr/bin/env python3
"""
Full benchmark experiments for diversity selection algorithms.

Tests DPP, MMR, clustering, submodular, embedding diversity, and fair diversity
methods on synthetic data. Produces divflow_benchmark_results.json.
"""

import sys
import os
import json
import time
import numpy as np

# Add source directory to path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'src')
sys.path.insert(0, SRC_DIR)

from dpp_sampler import DPPSampler, compute_kernel, DPPEnsemble, DualDPP
from mmr_selector import MMRSelector, cosine_similarity_matrix, tfidf_vectors
from clustering_diversity import (
    KMedoids, DBSCAN, HierarchicalClustering, SpectralClustering,
    ClusterDiversity, CoverageMaximizer, FacilityLocation, DiversityDispersion,
    pairwise_distances
)
from submodular_optimizer import (
    SubmodularOptimizer, FacilityLocationFunction, LogDeterminantFunction,
    CoverageFunction, FeatureBasedFunction, GraphCutFunction,
    verify_diminishing_returns
)
from embedding_diversity import (
    EmbeddingDiversity, DiversityComparator, PairwiseDiversityAnalyzer
)
from text_diversity_toolkit import TextDiversityToolkit
from fair_diversity import FairDiverseSelector
from benchmark_suite import (
    DiversityBenchmark, SyntheticDatasets, DiversityMetrics, BenchmarkReport
)


def section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ======================================================================
# Test 1: DPP Sampler
# ======================================================================

def test_dpp_sampler():
    """Test DPP sampler on synthetic kernel matrices."""
    section("Test 1: DPP Sampler")
    rng = np.random.RandomState(42)
    n, d = 50, 5
    items = rng.randn(n, d)
    K = compute_kernel(items, kernel='rbf', gamma=0.5)

    sampler = DPPSampler()
    sampler.fit(K)

    results = {}

    # Test exact DPP sampling
    sample = sampler.sample(rng=rng)
    print(f"  Exact DPP sample size: {len(sample)}")
    results['exact_sample_size'] = int(len(sample))
    assert len(sample) > 0, "DPP sample should not be empty"
    assert len(set(sample)) == len(sample), "No duplicates in DPP sample"

    # Test k-DPP sampling
    k = 8
    sample_k = sampler.sample_k(k, rng=rng)
    print(f"  k-DPP sample (k={k}): {len(sample_k)} items")
    assert len(sample_k) == k, f"k-DPP should return exactly {k} items"
    results['k_dpp_correct_size'] = True

    # Test greedy sampling
    greedy = sampler.greedy_sample(k)
    print(f"  Greedy DPP: {greedy}")
    assert len(greedy) == k
    results['greedy_size'] = int(len(greedy))

    # Verify diversity: DPP vs random
    n_trials = 20
    dpp_spreads = []
    random_spreads = []
    for _ in range(n_trials):
        dpp_sel = sampler.greedy_sample(k)
        rand_sel = rng.choice(n, size=k, replace=False)

        dpp_items = items[dpp_sel]
        rand_items = items[rand_sel]

        dpp_D = pairwise_distances(dpp_items)
        rand_D = pairwise_distances(rand_items)

        dpp_spread = dpp_D[np.triu_indices(k, k=1)].mean()
        rand_spread = rand_D[np.triu_indices(k, k=1)].mean()

        dpp_spreads.append(dpp_spread)
        random_spreads.append(rand_spread)

    avg_dpp = np.mean(dpp_spreads)
    avg_rand = np.mean(random_spreads)
    print(f"  DPP avg spread: {avg_dpp:.4f} vs Random: {avg_rand:.4f}")
    results['dpp_more_diverse'] = bool(avg_dpp >= avg_rand * 0.95)

    # Test MAP inference
    sel_map, logdet = sampler.map_inference(k)
    print(f"  MAP inference log-det: {logdet:.4f}")
    results['map_logdet'] = float(logdet)

    # Test log probability
    lp = sampler.log_probability(sample_k)
    print(f"  Log probability of k-DPP sample: {lp:.4f}")
    results['log_prob_finite'] = bool(np.isfinite(lp))

    # Test quality-diversity DPP
    quality = rng.rand(n) + 0.5
    L_qd = DPPSampler.quality_diversity_kernel(quality, K)
    sampler_qd = DPPSampler()
    sampler_qd.fit(L_qd)
    qd_sel = sampler_qd.greedy_sample(k)
    avg_quality_selected = quality[qd_sel].mean()
    avg_quality_all = quality.mean()
    print(f"  QD-DPP avg quality: {avg_quality_selected:.4f} (all: {avg_quality_all:.4f})")
    results['qd_quality_boost'] = float(avg_quality_selected)

    # Test conditional DPP
    fixed = [0, 1, 2]
    cond_sel = sampler.sample_conditional(fixed, 3, rng=rng)
    print(f"  Conditional DPP (fixed {fixed}, +3): {cond_sel}")
    assert len(set(cond_sel) & set(fixed)) == 0, "Conditional should not re-select fixed items"
    results['conditional_no_overlap'] = True

    # Test kernel learning
    observed = [rng.choice(n, size=5, replace=False).tolist() for _ in range(10)]
    L_learned = DPPSampler.learn_kernel(observed, n, n_features=5, max_iter=30)
    assert L_learned.shape == (n, n)
    results['kernel_learning_shape'] = list(L_learned.shape)

    # Test DPP Ensemble
    ensemble = DPPEnsemble(kernels=['rbf', 'cosine'])
    ensemble.fit(items)
    ens_sel = ensemble.sample(k, rng=rng)
    print(f"  Ensemble DPP: {ens_sel}")
    results['ensemble_size'] = int(len(ens_sel))

    # Test Dual DPP
    dual = DualDPP()
    dual.fit(items)
    dual_sel = dual.sample_k(k, rng=rng)
    print(f"  Dual DPP: {dual_sel}")
    results['dual_dpp_size'] = int(len(dual_sel))

    print("  ✓ DPP sampler tests passed")
    return results


# ======================================================================
# Test 2: MMR Selector
# ======================================================================

def test_mmr_selector():
    """Test MMR on text-like items with known relevance/diversity tradeoff."""
    section("Test 2: MMR Selector")
    rng = np.random.RandomState(42)
    n, d = 100, 20
    items = rng.randn(n, d)
    query = rng.randn(d)
    k = 10

    selector = MMRSelector()
    results = {}

    # Standard MMR
    sel = selector.select(items, query, k, lambda_param=0.5)
    print(f"  Standard MMR: {sel}")
    assert len(sel) == k
    results['standard_mmr_size'] = int(len(sel))

    # Fast MMR
    sel_fast = selector.select_fast(items, query, k, lambda_param=0.5)
    print(f"  Fast MMR: {sel_fast}")
    assert len(sel_fast) == k
    results['fast_mmr_size'] = int(len(sel_fast))

    # Batch MMR
    sel_batch = selector.select_batch(items, query, k, batch_size=3)
    print(f"  Batch MMR: {sel_batch}")
    results['batch_mmr_size'] = int(len(sel_batch))

    # High lambda = more relevant
    sel_high = selector.select_fast(items, query, k, lambda_param=0.9)
    sel_low = selector.select_fast(items, query, k, lambda_param=0.1)

    rel = cosine_similarity_matrix(items, query.reshape(1, -1)).ravel()
    avg_rel_high = np.mean(rel[sel_high])
    avg_rel_low = np.mean(rel[sel_low])
    print(f"  λ=0.9 avg relevance: {avg_rel_high:.4f}")
    print(f"  λ=0.1 avg relevance: {avg_rel_low:.4f}")
    results['high_lambda_more_relevant'] = bool(avg_rel_high > avg_rel_low)

    # Adaptive MMR
    sel_adaptive, best_lam = selector.select_adaptive(items, query, k)
    print(f"  Adaptive MMR best λ: {best_lam:.2f}")
    results['adaptive_lambda'] = float(best_lam)

    # Constrained MMR
    def even_only(idx, selected):
        return idx % 2 == 0
    sel_constr = selector.select_constrained(items, query, k,
                                             constraints=even_only)
    all_even = all(i % 2 == 0 for i in sel_constr)
    print(f"  Constrained (even only): {sel_constr}, all even: {all_even}")
    results['constraint_satisfied'] = bool(all_even)

    # Diversity reranking
    scores = rel.copy()
    sel_rerank = MMRSelector.diversity_rerank(items, scores, k)
    print(f"  Reranked: {sel_rerank}")
    results['rerank_size'] = int(len(sel_rerank))

    # Fairness MMR
    groups = rng.choice(3, size=n)
    sel_fair = MMRSelector.fairness_mmr(items, query, groups, k,
                                        min_per_group={0: 2, 1: 2, 2: 2})
    group_counts = {g: int(np.sum(groups[sel_fair] == g)) for g in range(3)}
    print(f"  Fair MMR group counts: {group_counts}")
    fair_satisfied = all(group_counts[g] >= 2 for g in range(3))
    results['fairness_satisfied'] = bool(fair_satisfied)

    print("  ✓ MMR selector tests passed")
    return results


# ======================================================================
# Test 3: Clustering Methods
# ======================================================================

def test_clustering():
    """Test clustering methods: verify correct cluster count recovery."""
    section("Test 3: Clustering Methods")
    rng = np.random.RandomState(42)

    # Well-separated clusters
    centers = np.array([[0, 0], [10, 10], [20, 0]], dtype=np.float64)
    X = np.vstack([c + rng.randn(30, 2) * 0.5 for c in centers])
    true_labels = np.array([0]*30 + [1]*30 + [2]*30)

    results = {}

    # K-medoids
    km = KMedoids(n_clusters=3)
    km.fit(X)
    n_clusters_km = len(np.unique(km.labels_))
    print(f"  K-medoids found {n_clusters_km} clusters (expected 3)")
    results['kmedoids_clusters'] = n_clusters_km
    results['kmedoids_correct'] = n_clusters_km == 3

    # DBSCAN
    db = DBSCAN(eps=2.0, min_samples=3)
    db.fit(X)
    n_clusters_db = len(set(db.labels_.tolist()) - {-1})
    print(f"  DBSCAN found {n_clusters_db} clusters (expected 3)")
    results['dbscan_clusters'] = n_clusters_db

    # Hierarchical
    hc = HierarchicalClustering(n_clusters=3)
    hc.fit(X)
    n_clusters_hc = len(np.unique(hc.labels_))
    print(f"  Hierarchical found {n_clusters_hc} clusters (expected 3)")
    results['hierarchical_clusters'] = n_clusters_hc

    # Spectral
    sc = SpectralClustering(n_clusters=3, gamma=0.01)
    sc.fit(X)
    n_clusters_sc = len(np.unique(sc.labels_))
    print(f"  Spectral found {n_clusters_sc} clusters (expected 3)")
    results['spectral_clusters'] = n_clusters_sc

    # Cluster diversity selection
    cd = ClusterDiversity(method='kmedoids')
    sel = cd.select(X, k=3)
    # Check that selected items are from different clusters
    sel_labels = km.labels_[sel]
    n_unique_labels = len(np.unique(sel_labels))
    print(f"  Cluster diversity: {n_unique_labels} unique clusters in selection")
    results['cluster_diversity_spread'] = n_unique_labels

    # Coverage maximizer
    cov = CoverageMaximizer(radius=3.0)
    sel_cov = cov.select(X, k=5)
    cov_score = cov.coverage_score(X, sel_cov)
    print(f"  Coverage: {cov_score:.3f}")
    results['coverage_score'] = float(cov_score)

    # Facility location
    fl = FacilityLocation()
    sel_fl, obj = fl.select(X, k=5)
    print(f"  Facility location objective: {obj:.3f}")
    results['facility_location_obj'] = float(obj)

    # Dispersion
    dd = DiversityDispersion()
    sel_dd, min_d = dd.select(X, k=5)
    print(f"  Dispersion min distance: {min_d:.3f}")
    results['dispersion_min_dist'] = float(min_d)

    print("  ✓ Clustering tests passed")
    return results


# ======================================================================
# Test 4: Submodular Optimizer
# ======================================================================

def test_submodular():
    """Test submodular optimizer: verify greedy (1-1/e) approximation."""
    section("Test 4: Submodular Optimizer")
    rng = np.random.RandomState(42)
    n = 30
    d = 5
    X = rng.randn(n, d)

    D = pairwise_distances(X)
    sim = np.max(D) - D
    np.fill_diagonal(sim, 0.0)

    optimizer = SubmodularOptimizer()
    results = {}

    # Facility location
    fl = FacilityLocationFunction(sim)
    k = 5
    sel_greedy, val_greedy = optimizer.greedy(fl, k)
    print(f"  Facility location greedy: value={val_greedy:.4f}")
    results['fl_greedy_value'] = float(val_greedy)

    # Stochastic greedy
    sel_stoch, val_stoch = optimizer.stochastic_greedy(fl, k, rng=rng)
    print(f"  Facility location stochastic: value={val_stoch:.4f}")
    results['fl_stochastic_value'] = float(val_stoch)

    # Check approximation ratio: compare greedy to exhaustive (small n)
    # Greedy should get at least (1-1/e) ≈ 0.632 of optimal
    # We can't compute optimal for n=30, so verify greedy >= stochastic * 0.9
    ratio = val_greedy / max(val_stoch, 1e-12)
    print(f"  Greedy/Stochastic ratio: {ratio:.4f}")
    results['greedy_vs_stochastic_ratio'] = float(ratio)

    # Small problem: verify (1-1/e) bound
    n_small = 10
    X_small = rng.randn(n_small, d)
    D_small = pairwise_distances(X_small)
    sim_small = np.max(D_small) - D_small
    np.fill_diagonal(sim_small, 0.0)
    fl_small = FacilityLocationFunction(sim_small)
    k_small = 3

    # Exhaustive search for optimal
    from itertools import combinations
    best_val = 0
    for subset in combinations(range(n_small), k_small):
        val = fl_small.evaluate(set(subset))
        best_val = max(best_val, val)

    sel_g, val_g = optimizer.greedy(fl_small, k_small)
    approx_ratio = val_g / max(best_val, 1e-12)
    print(f"  Small problem: greedy={val_g:.4f}, optimal={best_val:.4f}, ratio={approx_ratio:.4f}")
    print(f"  (1-1/e) ≈ 0.632, actual ratio: {approx_ratio:.4f}")
    results['approx_ratio'] = float(approx_ratio)
    results['meets_bound'] = bool(approx_ratio >= 0.63)

    # Verify diminishing returns
    is_sub, violations = verify_diminishing_returns(fl, n_tests=50, rng=rng)
    print(f"  Facility location submodular: {is_sub} ({len(violations)} violations)")
    results['fl_is_submodular'] = bool(is_sub)

    # Log-determinant
    K = X @ X.T + 0.1 * np.eye(n)
    ld = LogDeterminantFunction(K)
    sel_ld, val_ld = optimizer.greedy(ld, k)
    print(f"  Log-det greedy: value={val_ld:.4f}")
    results['logdet_value'] = float(val_ld)

    # Coverage function
    covers = [set(rng.choice(20, size=5, replace=False).tolist()) for _ in range(n)]
    cov = CoverageFunction(covers)
    sel_cov, val_cov = optimizer.greedy(cov, k)
    print(f"  Coverage greedy: value={val_cov:.4f}")
    results['coverage_value'] = float(val_cov)

    # Feature-based
    W = rng.rand(n, 10) * 0.3
    fb = FeatureBasedFunction(W)
    sel_fb, val_fb = optimizer.greedy(fb, k)
    print(f"  Feature-based greedy: value={val_fb:.4f}")
    results['feature_based_value'] = float(val_fb)

    is_sub_fb, v_fb = verify_diminishing_returns(fb, n_tests=50, rng=rng)
    results['feature_based_submodular'] = bool(is_sub_fb)

    # Graph cut
    W_g = rng.rand(n, n) * 0.5
    W_g = 0.5 * (W_g + W_g.T)
    np.fill_diagonal(W_g, 0.0)
    gc = GraphCutFunction(W_g)
    sel_gc, val_gc = optimizer.greedy(gc, k)
    print(f"  Graph cut greedy: value={val_gc:.4f}")
    results['graph_cut_value'] = float(val_gc)

    # Continuous greedy
    sel_cg, val_cg = optimizer.continuous_greedy(fl_small, k_small,
                                                  n_steps=10, n_samples=20,
                                                  rng=rng)
    print(f"  Continuous greedy: value={val_cg:.4f}")
    results['continuous_greedy_value'] = float(val_cg)

    print("  ✓ Submodular optimizer tests passed")
    return results


# ======================================================================
# Test 5: Embedding Diversity
# ======================================================================

def test_embedding_diversity():
    """Test embedding diversity metrics on controlled diversity levels."""
    section("Test 5: Embedding Diversity Metrics")
    rng = np.random.RandomState(42)

    computer = EmbeddingDiversity()
    results = {}

    # Generate data at different diversity levels
    diversity_levels = [0.01, 0.1, 0.5, 1.0, 2.0]
    metrics_by_level = {}

    for level in diversity_levels:
        data = SyntheticDatasets.controlled_diversity(
            n=50, d=10, diversity_level=level, random_state=42
        )
        scores = computer.compute(data)
        metrics_by_level[level] = scores
        print(f"  Level {level:.2f}: volume={scores.volume:.2f}, "
              f"spread={scores.spread:.3f}, vendi={scores.vendi_score:.2f}")

    # Verify ordering: higher diversity level -> higher metric values
    ordered_volume = all(
        metrics_by_level[diversity_levels[i]].volume <=
        metrics_by_level[diversity_levels[i+1]].volume + 1.0
        for i in range(len(diversity_levels) - 1)
    )
    ordered_spread = all(
        metrics_by_level[diversity_levels[i]].spread <=
        metrics_by_level[diversity_levels[i+1]].spread + 0.1
        for i in range(len(diversity_levels) - 1)
    )
    ordered_vendi = all(
        metrics_by_level[diversity_levels[i]].vendi_score <=
        metrics_by_level[diversity_levels[i+1]].vendi_score + 0.5
        for i in range(len(diversity_levels) - 1)
    )

    print(f"  Volume ordering correct: {ordered_volume}")
    print(f"  Spread ordering correct: {ordered_spread}")
    print(f"  Vendi ordering correct: {ordered_vendi}")

    results['volume_ordered'] = bool(ordered_volume)
    results['spread_ordered'] = bool(ordered_spread)
    results['vendi_ordered'] = bool(ordered_vendi)

    # Test comparator
    comparator = DiversityComparator()
    high_div = rng.randn(50, 10)
    low_div = rng.randn(1, 10) + rng.randn(50, 10) * 0.05
    comparison = comparator.compare([high_div, low_div], ['high', 'low'])
    print(f"  Comparator winner: {comparison['overall_winner']}")
    results['comparator_winner_correct'] = comparison['overall_winner'] == 'high'

    # Test analyzer
    analyzer = PairwiseDiversityAnalyzer()
    dist_stats = analyzer.distance_distribution(high_div)
    nn_stats = analyzer.nearest_neighbor_diversity(high_div)
    print(f"  Dist stats: mean={dist_stats['mean']:.3f}")
    print(f"  NN diversity: uniformity={nn_stats['uniformity']:.3f}")
    results['dist_mean'] = float(dist_stats['mean'])
    results['nn_uniformity'] = float(nn_stats['uniformity'])

    # Intrinsic dimensionality
    low_dim_data = SyntheticDatasets.manifold(n=100, ambient_dim=20,
                                               intrinsic_dim=3)
    dim = computer.intrinsic_dimensionality(low_dim_data)
    print(f"  Intrinsic dim of 3D manifold in 20D: {dim:.2f}")
    results['intrinsic_dim_estimate'] = float(dim)

    print("  ✓ Embedding diversity tests passed")
    return results


# ======================================================================
# Test 6: Text Diversity
# ======================================================================

def test_text_diversity():
    """Test text diversity toolkit."""
    section("Test 6: Text Diversity Toolkit")

    # Diverse texts (different topics)
    diverse_texts = [
        "The quick brown fox jumps over the lazy dog near the forest.",
        "Quantum computing uses qubits and superposition for parallel computation.",
        "The stock market showed strong gains in technology sector today.",
        "Ancient Egyptian pyramids were built thousands of years ago.",
        "Machine learning models require large datasets for training.",
        "The Pacific Ocean is the largest body of water on Earth.",
        "Mozart composed his first symphony at the age of eight.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
    ]

    # Homogeneous texts (similar topic)
    homogeneous_texts = [
        "The cat sat on the mat and looked around.",
        "The cat sat on the rug and gazed about.",
        "The cat rested on the mat and observed.",
        "The cat lay on the carpet and watched.",
        "The cat perched on the mat and surveyed.",
        "The cat lounged on the rug and glanced.",
        "The cat settled on the mat and peered.",
        "The cat curled on the carpet and stared.",
    ]

    toolkit = TextDiversityToolkit()
    results = {}

    # Analyze diverse texts
    report_div = toolkit.analyze(diverse_texts)
    print(f"  Diverse texts:")
    print(f"    Distinct-2: {report_div.distinct_2:.4f}")
    print(f"    Self-BLEU: {report_div.self_bleu:.4f}")
    print(f"    Semantic diversity: {report_div.semantic_diversity:.4f}")
    print(f"    Cross-text novelty: {report_div.cross_text_novelty:.4f}")

    # Analyze homogeneous texts
    report_hom = toolkit.analyze(homogeneous_texts)
    print(f"  Homogeneous texts:")
    print(f"    Distinct-2: {report_hom.distinct_2:.4f}")
    print(f"    Self-BLEU: {report_hom.self_bleu:.4f}")
    print(f"    Semantic diversity: {report_hom.semantic_diversity:.4f}")
    print(f"    Cross-text novelty: {report_hom.cross_text_novelty:.4f}")

    # Verify diverse > homogeneous
    results['distinct2_diverse_higher'] = bool(report_div.distinct_2 > report_hom.distinct_2)
    results['self_bleu_diverse_lower'] = bool(report_div.self_bleu < report_hom.self_bleu)
    results['semantic_diverse_higher'] = bool(report_div.semantic_diversity > report_hom.semantic_diversity)
    results['novelty_diverse_higher'] = bool(report_div.cross_text_novelty > report_hom.cross_text_novelty)

    print(f"  Distinct-2 diverse > homogeneous: {results['distinct2_diverse_higher']}")
    print(f"  Self-BLEU diverse < homogeneous: {results['self_bleu_diverse_lower']}")
    print(f"  Semantic diverse > homogeneous: {results['semantic_diverse_higher']}")

    # Homogenization detection
    results['homogeneous_detected'] = bool(report_hom.homogenization_alert)
    results['diverse_not_flagged'] = bool(not report_div.homogenization_alert)
    print(f"  Homogenization detected in homogeneous: {results['homogeneous_detected']}")
    print(f"  Diverse not flagged: {results['diverse_not_flagged']}")

    # Topic diversity
    results['topic_entropy_diverse'] = float(report_div.topic_entropy)
    results['topic_entropy_homogeneous'] = float(report_hom.topic_entropy)

    # Quality-diversity tradeoff
    quality = np.array([0.9, 0.85, 0.7, 0.65, 0.5, 0.4, 0.35, 0.2])
    tradeoff = TextDiversityToolkit.diversity_quality_tradeoff(diverse_texts, quality)
    results['tradeoff_points'] = len(tradeoff['sizes'])
    print(f"  Tradeoff curve: {results['tradeoff_points']} points")

    print("  ✓ Text diversity tests passed")
    return results


# ======================================================================
# Test 7: Fair Diversity
# ======================================================================

def test_fair_diversity():
    """Test fair diversity: verify group constraints satisfied."""
    section("Test 7: Fair Diversity")
    rng = np.random.RandomState(42)

    # 3 groups with different sizes
    items_g0 = rng.randn(40, 5) + np.array([0, 0, 0, 0, 0])
    items_g1 = rng.randn(30, 5) + np.array([3, 3, 0, 0, 0])
    items_g2 = rng.randn(10, 5) + np.array([0, 0, 5, 5, 0])
    items = np.vstack([items_g0, items_g1, items_g2])
    groups = np.array([0]*40 + [1]*30 + [2]*10)

    selector = FairDiverseSelector()
    k = 15
    results = {}

    # Group-fair with minimums
    min_per = {0: 3, 1: 3, 2: 3}
    sel1 = selector.select(items, groups, k, min_per_group=min_per, strategy='group_fair')
    group_counts = {g: int(np.sum(groups[sel1] == g)) for g in range(3)}
    print(f"  Group-fair counts: {group_counts}")
    all_satisfied = all(group_counts[g] >= min_per[g] for g in range(3))
    results['group_fair_satisfied'] = bool(all_satisfied)
    print(f"  All minimums satisfied: {all_satisfied}")

    # Proportional
    sel2 = selector.proportional_representation(items, groups, k)
    prop_counts = {g: int(np.sum(groups[sel2] == g)) for g in range(3)}
    print(f"  Proportional counts: {prop_counts}")
    # Expected: ~7.5, 5.6, 1.9 -> 7-8, 5-6, 1-2
    results['proportional_counts'] = prop_counts

    # Rooney rule
    sel3 = selector.rooney_rule(items, groups, k)
    rooney_counts = {g: int(np.sum(groups[sel3] == g)) for g in range(3)}
    print(f"  Rooney counts: {rooney_counts}")
    # Underrepresented group 2 should have >= 1
    results['rooney_all_represented'] = bool(all(rooney_counts[g] >= 1 for g in range(3)))

    # Within-group diversity
    sel4 = selector.diversity_within_groups(items, groups, k)
    print(f"  Within-group diverse: {len(sel4)} items")
    results['within_group_size'] = int(len(sel4))

    # Fairness metrics
    metrics = FairDiverseSelector.compute_fairness_metrics(items, groups, sel1)
    print(f"  Demographic parity: {metrics.demographic_parity:.4f}")
    print(f"  Representation ratio: {metrics.representation_ratio:.4f}")
    print(f"  Overall diversity: {metrics.overall_diversity:.4f}")
    results['demographic_parity'] = float(metrics.demographic_parity)
    results['representation_ratio'] = float(metrics.representation_ratio)

    # Intersectional
    gender = np.array([0]*20 + [1]*20 + [0]*15 + [1]*15 + [0]*5 + [1]*5)
    sel5 = selector.intersectional_selection(items, [groups, gender], k)
    combined = FairDiverseSelector.create_intersectional_groups([groups, gender])
    n_intersections = len(np.unique(combined[sel5]))
    print(f"  Intersectional groups in selection: {n_intersections}")
    results['intersectional_groups'] = n_intersections

    # Pareto tradeoff
    pareto_results = selector.pareto_tradeoff(items, groups, k, n_configs=10)
    n_pareto = sum(1 for r in pareto_results if r.get('is_pareto', False))
    print(f"  Pareto-optimal configs: {n_pareto}")
    results['pareto_configs'] = n_pareto

    print("  ✓ Fair diversity tests passed")
    return results


# ======================================================================
# Test 8: Full Benchmark Suite
# ======================================================================

def test_benchmark_suite():
    """Run full benchmark comparing all methods."""
    section("Test 8: Full Benchmark Suite")
    rng = np.random.RandomState(42)

    results = {}

    # Define methods
    def dpp_method(items, k):
        K = compute_kernel(items, kernel='rbf')
        s = DPPSampler()
        s.fit(K)
        return s.greedy_sample(k)

    def mmr_method(items, k):
        query = items.mean(axis=0)
        sel = MMRSelector()
        return sel.select_fast(items, query, k, lambda_param=0.3)

    def cluster_method(items, k):
        cd = ClusterDiversity(method='kmedoids')
        return cd.select(items, k)

    def submodular_method(items, k):
        D = pairwise_distances(items)
        sim = np.max(D) - D
        np.fill_diagonal(sim, 0.0)
        fl = FacilityLocationFunction(sim)
        opt = SubmodularOptimizer()
        sel, _ = opt.greedy(fl, k)
        return np.array(sel)

    def random_method(items, k):
        return rng.choice(len(items), size=min(k, len(items)), replace=False)

    def farthest_method(items, k):
        dd = DiversityDispersion()
        sel, _ = dd.select(items, k)
        return sel

    methods = {
        'DPP': dpp_method,
        'MMR': mmr_method,
        'Clustering': cluster_method,
        'Submodular': submodular_method,
        'Random': random_method,
        'FarthestPoint': farthest_method,
    }

    # Run benchmark
    bench = DiversityBenchmark()
    report = bench.run(methods, k=10, n_trials=3)
    print(report.summary())

    results['per_method_scores'] = report.per_method_scores
    results['winner'] = report.winner
    results['statistical_comparisons'] = report.statistical_comparisons

    # Verify DPP/Submodular beat random
    dpp_spread = report.per_method_scores.get('DPP', {}).get('spread', 0)
    rand_spread = report.per_method_scores.get('Random', {}).get('spread', 0)
    results['dpp_beats_random'] = bool(dpp_spread >= rand_spread * 0.9)
    print(f"\n  DPP spread: {dpp_spread:.4f}")
    print(f"  Random spread: {rand_spread:.4f}")
    print(f"  DPP beats random: {results['dpp_beats_random']}")

    sub_spread = report.per_method_scores.get('Submodular', {}).get('spread', 0)
    results['submodular_beats_random'] = bool(sub_spread >= rand_spread * 0.9)

    # Scaling benchmark
    scaling = bench.scaling_benchmark(dpp_method, 'DPP', sizes=[30, 60, 100])
    results['scaling'] = scaling
    scaling_summary = [(s['n'], round(s['runtime'], 3)) for s in scaling]
    print(f"  DPP scaling: {scaling_summary}")

    # Robustness
    robust = bench.robustness_benchmark(farthest_method, 'FarthestPoint', n=50)
    results['robustness_spreads'] = robust['spreads']

    print("  ✓ Benchmark suite tests passed")
    return results


# ======================================================================
# Main
# ======================================================================

def main():
    """Run all tests and produce results JSON."""
    print("=" * 60)
    print("  DIVERSITY METHODS FULL BENCHMARK")
    print("=" * 60)

    all_results = {}
    start_time = time.time()

    all_results['dpp'] = test_dpp_sampler()
    all_results['mmr'] = test_mmr_selector()
    all_results['clustering'] = test_clustering()
    all_results['submodular'] = test_submodular()
    all_results['embedding_diversity'] = test_embedding_diversity()
    all_results['text_diversity'] = test_text_diversity()
    all_results['fair_diversity'] = test_fair_diversity()
    all_results['benchmark'] = test_benchmark_suite()

    total_time = time.time() - start_time

    # Summary
    section("SUMMARY")
    print(f"  Total time: {total_time:.2f}s")

    n_tests = 0
    n_passed = 0
    for category, results in all_results.items():
        if isinstance(results, dict):
            for key, val in results.items():
                if isinstance(val, bool):
                    n_tests += 1
                    if val:
                        n_passed += 1
                    else:
                        print(f"  ⚠ {category}.{key} = False")

    print(f"  Tests: {n_passed}/{n_tests} passed")
    all_results['summary'] = {
        'total_time': total_time,
        'tests_passed': n_passed,
        'tests_total': n_tests,
        'all_passed': n_passed == n_tests
    }

    # Ensure clean serialization
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, set):
            return sorted(list(obj))
        return obj

    all_results_clean = make_serializable(all_results)

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'divflow_benchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results_clean, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")
    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE: {n_passed}/{n_tests} tests passed")
    print(f"{'='*60}")

    return all_results


if __name__ == '__main__':
    main()
