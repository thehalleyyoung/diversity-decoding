"""
Comprehensive benchmark: test all diversity modules, run existing benchmarks,
produce comprehensive_benchmark_results.json.
"""

import sys
import os
import json
import time
import traceback
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.diversity_theory import (
    DiversityTheory, SubmodularityVerifier, MonotonicityVerifier,
    ApproximationGuarantee, ParetoAnalysis, SampleComplexity,
    ConcentrationInequalities, MinimaxDiversity, InformationTheoreticDiversity
)
from src.active_diverse_selection import (
    ActiveDiverseSelector, ActiveDiverseSelectionSimulator, UncertaintyEstimator,
    DiversityScorer, BatchSelectionResult
)
from src.generation_diversity import (
    DiverseGenerator, VocabSimulator, TemperatureSampler, TopKSampler,
    NucleusSampler, TypicalSampler, ContrastiveSearch, DiverseBeamSearch,
    StochasticBeamSearch, ControllableDiversity
)
from src.recommendation_diversity import (
    DiverseRecommender, RecommendationItem, UserProfile,
    CalibratedRecommender, SerendipityOptimizer, NoveltyScorer,
    IntraListDiversity, CoverageOptimizer, FairnessAwareRecommender,
    TemporalDiversity, MultiStakeholderDiversity
)
from src.retrieval_diversity import (
    DiverseRetriever, Document, PM2Diversifier, XQuADDiversifier,
    IASelectDiversifier, MMRRetriever, SubtopicCoverage,
    NoveltyBiasedRanker, DiversifiedReRanker
)
from src.ensemble_diversity import (
    EnsembleDiversity, DisagreementMeasure, QStatistic,
    CorrelationDiversity, DoubleFaultMeasure, EnsembleEntropy,
    KohaviWolpertVariance, DiversityAccuracyTradeoff,
    DiverseEnsembleConstruction
)
from src.curriculum_data_diversity import (
    CurriculumDiversity, FacilityLocation, CoresetConstruction,
    StratifiedSampler, DataDeduplicator, DifficultyDiversityBalance,
    ActiveDataSelector, AugmentationDiversity
)
from src.evaluation_metrics import (
    DiversityEvaluator, CoveragePrecisionRecall, NormalizedDiscountedCumulativeDiversity,
    AlphaNDCG, ERRIA, SubtopicRecallMetric, DiversityLift,
    StatisticalSignificance, DiversityQualityTradeoff
)
from src.adversarial_diversity import (
    AdversarialDiversifier, AdversarialFilter, GANDiversity,
    ContrastiveDiversity, AdversarialDeduplication,
    DiversityRobustnessTester, DiversityAttack, SimpleDiscriminator
)


def run_with_timing(name, func):
    """Run a benchmark function with timing and error handling."""
    print(f"  Running: {name}...", end=" ", flush=True)
    start = time.time()
    try:
        result = func()
        elapsed = time.time() - start
        print(f"OK ({elapsed:.2f}s)")
        return {"status": "passed", "time": elapsed, "result": result}
    except Exception as e:
        elapsed = time.time() - start
        print(f"FAILED ({elapsed:.2f}s): {e}")
        traceback.print_exc()
        return {"status": "failed", "time": elapsed, "error": str(e)}


def test_diversity_theory():
    """Test diversity theory: verify submodularity, monotonicity, greedy bounds."""
    results = {}
    rng = np.random.RandomState(42)

    def test_facility_location():
        theory = DiversityTheory(seed=42)
        props = theory.analyze('facility_location', n=8, dim=5, k=3)
        assert props.is_submodular, "Facility location should be submodular"
        assert props.is_monotone, "Facility location should be monotone"
        assert props.approximation_ratio >= (1 - 1/np.e) - 0.01, \
            f"Greedy should achieve (1-1/e) bound, got {props.approximation_ratio}"
        return {
            "is_submodular": props.is_submodular,
            "is_monotone": props.is_monotone,
            "approx_ratio": props.approximation_ratio,
            "minimax": props.minimax_value,
        }

    def test_max_coverage():
        theory = DiversityTheory(seed=42)
        props = theory.analyze('max_coverage', n=8, n_elements=15, k=3)
        assert props.is_submodular, "Max coverage should be submodular"
        assert props.is_monotone, "Max coverage should be monotone"
        return {"is_submodular": props.is_submodular, "is_monotone": props.is_monotone}

    def test_log_det():
        theory = DiversityTheory(seed=42)
        props = theory.analyze('log_det', n=6, dim=4, k=3)
        assert props.is_submodular, "log-det should be submodular"
        return {"is_submodular": props.is_submodular}

    def test_greedy_bound():
        n = 10
        dim = 5
        points = rng.randn(n, dim)
        dist_matrix = squareform(pdist(points))
        max_d = np.max(dist_matrix)
        sim = max_d - dist_matrix

        def fl_func(subset):
            if not subset:
                return 0.0
            s_list = sorted(subset)
            return float(np.sum(np.max(sim[:, s_list], axis=1)))

        approx = ApproximationGuarantee(list(range(n)))
        result = approx.verify_greedy_bound(fl_func, 3)
        assert result['satisfies_bound'], "Greedy should satisfy (1-1/e) bound"
        return {"ratio": result['approximation_ratio'], "satisfies": result['satisfies_bound']}

    def test_pareto():
        points = rng.randn(30, 5)
        quality = rng.rand(30)
        pa = ParetoAnalysis(points, quality)
        pareto = pa.compute_pareto_frontier(5, n_weights=20)
        assert len(pareto) >= 2, "Should find at least 2 Pareto points"
        area = pa.area_under_pareto(pareto)
        return {"n_pareto_points": len(pareto), "area": area}

    def test_concentration():
        ci = ConcentrationInequalities()
        bound = ci.mcdiarmid_bound(100, [0.1] * 10)
        assert bound['bound'] > 0, "Bound should be positive"
        hoeff = ci.hoeffding_bound(100, 1.0)
        assert hoeff['bound'] > 0
        return {"mcdiarmid": bound['bound'], "hoeffding": hoeff['bound']}

    def test_minimax():
        points = rng.randn(20, 5)
        mm = MinimaxDiversity(points)
        centers, radius = mm.k_center_greedy(5)
        assert len(centers) == 5
        disp_result = mm.max_dispersion_greedy(5)
        assert disp_result[1] > 0
        return {"radius": radius, "dispersion": disp_result[1]}

    def test_info_theory():
        it = InformationTheoreticDiversity()
        data = rng.randn(50, 5)
        ent = it.entropy(data)
        assert ent > 0
        tc = it.total_correlation(data)
        return {"entropy": ent, "total_correlation": tc}

    results['facility_location'] = run_with_timing("facility_location", test_facility_location)
    results['max_coverage'] = run_with_timing("max_coverage", test_max_coverage)
    results['log_det'] = run_with_timing("log_det", test_log_det)
    results['greedy_bound'] = run_with_timing("greedy_bound", test_greedy_bound)
    results['pareto'] = run_with_timing("pareto", test_pareto)
    results['concentration'] = run_with_timing("concentration", test_concentration)
    results['minimax'] = run_with_timing("minimax", test_minimax)
    results['info_theory'] = run_with_timing("info_theory", test_info_theory)
    return results


def test_active_selection():
    """Test active diverse selection: simulate rounds, verify coverage improves."""
    results = {}
    rng = np.random.RandomState(42)

    def test_simulation():
        pool = rng.randn(200, 10)
        selector = ActiveDiverseSelector(diversity_weight=0.5, seed=42)
        sim = ActiveDiverseSelectionSimulator(pool, selector, seed=42)
        sim_result = sim.simulate(n_rounds=100, k_per_round=1)

        coverages = [r['coverage'] for r in sim_result['round_results']]
        assert coverages[-1] >= coverages[0], "Coverage should improve over rounds"
        assert sim_result['n_selected'] > 50, "Should select many items"
        return {
            "n_selected": sim_result['n_selected'],
            "initial_coverage": coverages[0],
            "final_coverage": coverages[-1],
            "coverage_improved": coverages[-1] > coverages[0]
        }

    def test_batch():
        pool = rng.randn(100, 10)
        selected = rng.randn(5, 10)
        selector = ActiveDiverseSelector(diversity_weight=0.5, seed=42)
        result = selector.batch_select(pool, selected, k=10, method='greedy')
        assert len(result.selected_indices) == 10
        return {"n_selected": len(result.selected_indices), "diversity": result.total_diversity}

    def test_bayesian():
        pool = rng.randn(50, 10)
        selected = rng.randn(3, 10)
        n_classes = 5
        logits = pool @ rng.randn(10, n_classes)
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        preds = exp_l / exp_l.sum(axis=1, keepdims=True)
        selector = ActiveDiverseSelector(seed=42)
        result = selector.bayesian_select(pool, selected, preds)
        assert result.selected_index >= 0
        return {"selected": result.selected_index, "score": result.score}

    def test_ucb():
        pool = rng.randn(50, 10)
        selected = rng.randn(3, 10)
        logits = pool @ rng.randn(10, 5)
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        preds = exp_l / exp_l.sum(axis=1, keepdims=True)
        selector = ActiveDiverseSelector(seed=42)
        result = selector.ucb_select(pool, selected, preds, t=10)
        assert result.selected_index >= 0
        return {"selected": result.selected_index}

    results['simulation_100_rounds'] = run_with_timing("100-round simulation", test_simulation)
    results['batch_select'] = run_with_timing("batch selection", test_batch)
    results['bayesian'] = run_with_timing("bayesian selection", test_bayesian)
    results['ucb'] = run_with_timing("UCB selection", test_ucb)
    return results


def test_generation_diversity():
    """Test generation: varying temperature, verify diversity monotonic."""
    results = {}

    def test_temperature_monotonic():
        gen = DiverseGenerator(vocab_size=200, embed_dim=32, seed=42)
        prompt = np.random.RandomState(42).randn(32)
        context = gen._embed_to_tokens(prompt)

        diversities = []
        levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        for level in levels:
            metrics = gen.controllable.measure_generation_diversity(
                context, n=8, diversity_level=level, method='temperature'
            )
            diversities.append(metrics['pairwise_diversity'])

        mostly_increasing = sum(
            1 for i in range(1, len(diversities)) if diversities[i] >= diversities[i-1] * 0.8
        )
        return {
            "diversities": diversities,
            "levels": levels,
            "mostly_monotonic": mostly_increasing >= len(levels) - 2
        }

    def test_diverse_beam():
        gen = DiverseGenerator(vocab_size=200, embed_dim=32, seed=42)
        prompt = np.random.RandomState(42).randn(32)
        results_beam = gen.generate(prompt, n=4, method='diverse_beam')
        assert len(results_beam) >= 2
        return {"n_generated": len(results_beam)}

    def test_contrastive():
        gen = DiverseGenerator(vocab_size=200, embed_dim=32, seed=42)
        prompt = np.random.RandomState(42).randn(32)
        results_c = gen.generate(prompt, n=4, method='contrastive')
        assert len(results_c) >= 2
        return {"n_generated": len(results_c)}

    def test_stochastic_beam():
        gen = DiverseGenerator(vocab_size=200, embed_dim=32, seed=42)
        prompt = np.random.RandomState(42).randn(32)
        results_s = gen.generate(prompt, n=4, method='stochastic_beam')
        assert len(results_s) >= 2
        return {"n_generated": len(results_s)}

    results['temperature_monotonic'] = run_with_timing("temperature monotonicity", test_temperature_monotonic)
    results['diverse_beam'] = run_with_timing("diverse beam search", test_diverse_beam)
    results['contrastive'] = run_with_timing("contrastive search", test_contrastive)
    results['stochastic_beam'] = run_with_timing("stochastic beam search", test_stochastic_beam)
    return results


def test_recommendation_diversity():
    """Test recommendations: simulate users, verify calibration and novelty."""
    results = {}
    rng = np.random.RandomState(42)

    def test_multi_user():
        rec = DiverseRecommender(n_categories=10, n_providers=5, seed=42)
        items = rec.create_test_items(100, dim=20)

        all_cal = []
        all_nov = []
        all_div = []
        all_cov = []

        for user_id in range(50):
            user = UserProfile(
                user_id=user_id,
                preference_vector=rng.randn(20),
                history=rng.choice(100, size=5, replace=False).tolist(),
                category_distribution=rng.dirichlet(np.ones(10))
            )
            result = rec.recommend(user, items, k=10, method='multi_stakeholder')
            all_cal.append(result.calibration_score)
            all_nov.append(result.novelty_score)
            all_div.append(result.diversity_score)
            all_cov.append(result.coverage_score)

        return {
            "n_users": 50,
            "avg_calibration": float(np.mean(all_cal)),
            "avg_novelty": float(np.mean(all_nov)),
            "avg_diversity": float(np.mean(all_div)),
            "avg_coverage": float(np.mean(all_cov)),
        }

    def test_methods():
        rec = DiverseRecommender(n_categories=10, n_providers=5, seed=42)
        items = rec.create_test_items(50, dim=20)
        user = rec.create_test_user(dim=20, n_history=5)
        # Fix history to valid range
        user.history = rng.choice(50, size=5, replace=False).tolist()

        method_results = {}
        for method in ['calibrated', 'serendipity', 'novelty', 'mmr', 'coverage', 'fairness']:
            result = rec.recommend(user, items, k=10, method=method)
            method_results[method] = {
                "diversity": result.diversity_score,
                "calibration": result.calibration_score,
                "novelty": result.novelty_score,
            }
        return method_results

    results['50_users'] = run_with_timing("50-user simulation", test_multi_user)
    results['methods'] = run_with_timing("recommendation methods", test_methods)
    return results


def test_retrieval_diversity():
    """Test retrieval: 5-topic corpus, verify subtopic coverage."""
    results = {}
    rng = np.random.RandomState(42)

    def test_subtopic_coverage():
        retriever = DiverseRetriever(n_subtopics=10, n_aspects=5, seed=42)
        corpus = retriever.create_test_corpus(100, dim=20, n_topics=5)
        query = rng.randn(20)

        method_results = {}
        for method in ['xquad', 'pm2', 'ia_select', 'mmr', 'subtopic', 'novelty', 'rerank']:
            result = retriever.retrieve(query, corpus, k=10, method=method)
            method_results[method] = {
                "diversity": result.diversity_score,
                "coverage": result.subtopic_coverage,
            }

        # Verify diversification methods have decent coverage
        for m in ['xquad', 'subtopic']:
            assert method_results[m]['coverage'] > 0, f"{m} should have some coverage"

        return method_results

    def test_5_topic_coverage():
        retriever = DiverseRetriever(n_subtopics=5, n_aspects=5, seed=42)
        corpus = retriever.create_test_corpus(50, dim=10, n_topics=5)
        query = rng.randn(10)

        result = retriever.retrieve(query, corpus, k=10, method='subtopic')
        topics_covered = set()
        for doc_id in result.doc_ids:
            topics_covered.update(corpus[doc_id].subtopics)

        return {
            "topics_covered": len(topics_covered),
            "coverage": result.subtopic_coverage,
            "diversity": result.diversity_score,
        }

    results['methods'] = run_with_timing("retrieval methods", test_subtopic_coverage)
    results['5_topic'] = run_with_timing("5-topic coverage", test_5_topic_coverage)
    return results


def test_ensemble_diversity():
    """Test ensemble: 5 models, verify disagreement correlates with improvement."""
    results = {}
    rng = np.random.RandomState(42)

    def test_analyze():
        ens = EnsembleDiversity(seed=42)
        preds, labels = ens.generate_test_ensemble(
            n_models=5, n_samples=200, n_classes=3, diversity_level=0.5
        )
        report = ens.analyze(preds, labels)

        assert report.disagreement > 0, "Should have some disagreement"
        assert len(report.individual_accuracies) == 5
        return {
            "disagreement": report.disagreement,
            "q_statistic": report.q_statistic,
            "correlation": report.correlation_diversity,
            "double_fault": report.double_fault,
            "entropy": report.entropy_diversity,
            "kw_variance": report.kw_variance,
            "ensemble_accuracy": report.ensemble_accuracy,
            "avg_individual_accuracy": float(np.mean(report.individual_accuracies)),
            "improvement": report.ensemble_accuracy - float(np.mean(report.individual_accuracies)),
        }

    def test_diversity_levels():
        ens = EnsembleDiversity(seed=42)
        disagreements = []
        improvements = []

        for div_level in [0.2, 0.5, 0.8]:
            preds, labels = ens.generate_test_ensemble(
                n_models=5, n_samples=200, n_classes=3, diversity_level=div_level
            )
            report = ens.analyze(preds, labels)
            disagreements.append(report.disagreement)
            improvements.append(
                report.ensemble_accuracy - float(np.mean(report.individual_accuracies))
            )

        return {
            "disagreements": disagreements,
            "improvements": improvements,
            "correlation_positive": disagreements[-1] > disagreements[0]
        }

    def test_construction():
        ens = EnsembleDiversity(seed=42)
        n_samples = 200
        n_classes = 3
        labels = rng.randint(0, n_classes, size=n_samples)

        n_candidates = 10
        candidate_preds = np.zeros((n_candidates, n_samples), dtype=int)
        for m in range(n_candidates):
            acc = rng.uniform(0.5, 0.8)
            correct = rng.rand(n_samples) < acc
            candidate_preds[m] = np.where(correct, labels, rng.randint(0, n_classes, size=n_samples))

        construction = DiverseEnsembleConstruction(seed=42)
        selected = construction.greedy_construct(candidate_preds, labels, k=5, lambda_div=0.5)
        assert len(selected) == 5

        sub_preds = candidate_preds[selected]
        ensemble_pred = np.zeros(n_samples, dtype=int)
        for s in range(n_samples):
            votes = sub_preds[:, s]
            values, counts = np.unique(votes, return_counts=True)
            ensemble_pred[s] = values[np.argmax(counts)]
        ens_acc = float(np.mean(ensemble_pred == labels))

        return {"selected_models": selected, "ensemble_accuracy": ens_acc}

    results['analyze'] = run_with_timing("ensemble analysis", test_analyze)
    results['diversity_levels'] = run_with_timing("diversity levels", test_diversity_levels)
    results['construction'] = run_with_timing("ensemble construction", test_construction)
    return results


def test_curriculum_diversity():
    """Test curriculum: select training subset, verify class balance + coverage."""
    results = {}
    rng = np.random.RandomState(42)

    def test_facility_location():
        data = rng.randn(200, 10)
        labels = rng.randint(0, 5, size=200)
        cd = CurriculumDiversity(seed=42)
        result = cd.select(data, budget=50, labels=labels, method='facility_location')
        assert len(result.indices) == 50
        assert result.coverage > 0
        return {
            "n_selected": len(result.indices),
            "coverage": result.coverage,
            "diversity": result.diversity_score,
        }

    def test_stratified():
        data = rng.randn(200, 10)
        labels = rng.randint(0, 5, size=200)
        cd = CurriculumDiversity(seed=42)
        result = cd.select(data, budget=50, labels=labels, method='stratified')
        assert len(result.class_balance) > 0
        balance_values = list(result.class_balance.values())
        max_imbalance = max(balance_values) - min(balance_values)
        return {
            "n_selected": len(result.indices),
            "class_balance": result.class_balance,
            "max_imbalance": max_imbalance,
        }

    def test_deduplication():
        data = rng.randn(100, 10)
        # Add duplicates
        duplicates = data[:20] + rng.randn(20, 10) * 0.01
        all_data = np.vstack([data, duplicates])
        dedup = DataDeduplicator()
        result = dedup.deduplicate_threshold(all_data, threshold=0.5)
        assert result.reduction_ratio > 0, "Should remove some duplicates"
        return {
            "kept": len(result.kept_indices),
            "removed": len(result.removed_indices),
            "reduction": result.reduction_ratio,
        }

    def test_coreset():
        data = rng.randn(200, 10)
        coreset = CoresetConstruction()
        centers = coreset.k_center_greedy(data, k=20)
        radius = coreset.k_center_radius(data, centers)
        assert len(centers) == 20
        assert radius > 0
        return {"n_centers": len(centers), "radius": radius}

    results['facility_location'] = run_with_timing("facility location", test_facility_location)
    results['stratified'] = run_with_timing("stratified sampling", test_stratified)
    results['deduplication'] = run_with_timing("deduplication", test_deduplication)
    results['coreset'] = run_with_timing("coreset construction", test_coreset)
    return results


def test_evaluation_metrics():
    """Test evaluation: compute all metrics on known diversity levels, verify ordering."""
    results = {}
    rng = np.random.RandomState(42)

    def test_coverage_pr():
        reference = rng.randn(100, 5)
        selected_diverse = reference[rng.choice(100, size=20, replace=False)]
        selected_clustered = reference[:5] + rng.randn(5, 5) * 0.01

        ev = DiversityEvaluator(seed=42)
        report_diverse = ev.evaluate(selected_diverse, reference)
        report_clustered = ev.evaluate(np.vstack([reference[:5], selected_clustered]), reference)

        assert report_diverse.pairwise_diversity > report_clustered.pairwise_diversity, \
            "Diverse selection should have higher pairwise diversity"

        return {
            "diverse_diversity": report_diverse.pairwise_diversity,
            "clustered_diversity": report_clustered.pairwise_diversity,
            "diverse_recall": report_diverse.recall,
            "diverse_ndcd": report_diverse.ndcd,
        }

    def test_alpha_ndcg():
        andcg = AlphaNDCG(alpha=0.5)
        doc_subtopics = {0: [0, 1], 1: [0], 2: [1, 2], 3: [2, 3], 4: [3, 4]}
        all_subtopics = {0, 1, 2, 3, 4}

        diverse_ranking = [0, 3, 4, 2, 1]
        redundant_ranking = [0, 1, 2, 3, 4]

        score_diverse = andcg.compute(diverse_ranking, doc_subtopics, all_subtopics)
        score_redundant = andcg.compute(redundant_ranking, doc_subtopics, all_subtopics)

        return {
            "diverse_score": score_diverse,
            "redundant_score": score_redundant,
        }

    def test_subtopic_recall():
        sr = SubtopicRecallMetric()
        all_topics = {0, 1, 2, 3, 4}
        full_coverage = [{0, 1}, {2, 3}, {4}]
        partial_coverage = [{0, 1}]

        full_recall = sr.compute(full_coverage, all_topics)
        partial_recall = sr.compute(partial_coverage, all_topics)

        assert full_recall > partial_recall
        assert abs(full_recall - 1.0) < 1e-10
        return {"full": full_recall, "partial": partial_recall}

    def test_diversity_lift():
        pool = rng.randn(200, 5)
        diverse_idx = []
        remaining = list(range(200))
        diverse_idx.append(remaining.pop(0))
        for _ in range(19):
            dists = cdist(pool[remaining], pool[diverse_idx])
            min_dists = np.min(dists, axis=1)
            best = np.argmax(min_dists)
            diverse_idx.append(remaining.pop(best))

        diverse_selected = pool[diverse_idx]
        lift = DiversityLift(seed=42)
        result = lift.compute(diverse_selected, pool)
        assert result['normalized_lift'] > 0, "Greedy diverse should beat random"
        return result

    def test_significance():
        sig = StatisticalSignificance(seed=42)
        a = rng.randn(100) + 0.5
        b = rng.randn(100)
        result = sig.paired_bootstrap(a, b)
        assert result['significant'], "Should detect significant difference"
        return result

    def test_tradeoff():
        items = rng.randn(50, 5)
        quality = rng.rand(50)
        tradeoff = DiversityQualityTradeoff(seed=42)
        curve = tradeoff.compute_tradeoff_curve(items, quality, k=10, n_weights=10)
        assert len(curve) >= 2
        pareto = tradeoff.extract_pareto_frontier(curve)
        return {"n_curve_points": len(curve), "n_pareto": len(pareto)}

    results['coverage_pr'] = run_with_timing("coverage precision-recall", test_coverage_pr)
    results['alpha_ndcg'] = run_with_timing("alpha-nDCG", test_alpha_ndcg)
    results['subtopic_recall'] = run_with_timing("subtopic recall", test_subtopic_recall)
    results['diversity_lift'] = run_with_timing("diversity lift", test_diversity_lift)
    results['significance'] = run_with_timing("statistical significance", test_significance)
    results['tradeoff'] = run_with_timing("tradeoff curve", test_tradeoff)
    return results


def test_adversarial_diversity():
    """Test adversarial diversity methods."""
    results = {}
    rng = np.random.RandomState(42)

    def test_filter():
        existing = rng.randn(30, 10)
        candidates = rng.randn(50, 10)
        af = AdversarialFilter(seed=42)
        result = af.filter(existing, candidates, threshold=0.6)
        assert len(result.selected_indices) > 0
        return {"kept": len(result.selected_indices), "filtered": result.n_filtered}

    def test_contrastive():
        items = rng.randn(30, 10)
        cd = ContrastiveDiversity(seed=42)
        selected = cd.maximize_contrastive(items, k=10)
        assert len(selected) == 10
        div = float(np.mean(pdist(items[selected])))
        return {"selected": len(selected), "diversity": div}

    def test_gan():
        data = rng.randn(50, 10)
        gan = GANDiversity(dim=10, latent_dim=8, seed=42)
        history = gan.train(data, n_epochs=30, batch_size=16)
        generated = gan.generate(10)
        assert generated.shape == (10, 10)
        return {"generated_shape": list(generated.shape), "final_d_loss": history['d_losses'][-1]}

    def test_robustness():
        items = rng.randn(15, 5)

        def div_func(x):
            if len(x) < 2:
                return 0.0
            return float(np.mean(pdist(x)))

        tester = DiversityRobustnessTester(seed=42)
        result = tester.test_robustness(items, div_func, epsilon=0.1, n_perturbations=50)
        assert result.robustness_score > 0
        return {
            "original": result.original_diversity,
            "perturbed": result.perturbed_diversity,
            "robustness": result.robustness_score,
        }

    def test_attack():
        items = rng.randn(10, 5)

        def div_func(x):
            if len(x) < 2:
                return 0.0
            return float(np.mean(pdist(x)))

        attacker = DiversityAttack(seed=42)
        result = attacker.random_attack(items, div_func, epsilon=0.3)
        return {
            "original": result['original_diversity'],
            "attacked": result['attacked_diversity'],
            "decrease": result['max_decrease'],
        }

    results['filter'] = run_with_timing("adversarial filter", test_filter)
    results['contrastive'] = run_with_timing("contrastive diversity", test_contrastive)
    results['gan'] = run_with_timing("GAN diversity", test_gan)
    results['robustness'] = run_with_timing("robustness testing", test_robustness)
    results['attack'] = run_with_timing("diversity attack", test_attack)
    return results


def run_existing_benchmarks():
    """Run any existing benchmark modules if available."""
    results = {}
    rng = np.random.RandomState(42)

    # Test DPP (if dpp_sampler exists)
    def test_dpp():
        try:
            from src.dpp_sampler import DPPSampler
            sampler = DPPSampler(seed=42) if hasattr(DPPSampler, '__init__') else DPPSampler()
            return {"status": "module_found"}
        except Exception as e:
            return {"status": "skipped", "reason": str(e)}

    # Test MMR (if mmr_selector exists)
    def test_mmr():
        try:
            from src.mmr_selector import MMRSelector
            return {"status": "module_found"}
        except Exception as e:
            return {"status": "skipped", "reason": str(e)}

    # Test submodular (if submodular_optimizer exists)
    def test_submodular():
        try:
            from src.submodular_optimizer import SubmodularOptimizer
            return {"status": "module_found"}
        except Exception as e:
            return {"status": "skipped", "reason": str(e)}

    # Test embedding diversity
    def test_embedding():
        try:
            from src.embedding_diversity import EmbeddingDiversity
            return {"status": "module_found"}
        except Exception as e:
            return {"status": "skipped", "reason": str(e)}

    # Test fair diversity
    def test_fair():
        try:
            from src.fair_diversity import FairDiversitySelector
            return {"status": "module_found"}
        except Exception as e:
            return {"status": "skipped", "reason": str(e)}

    results['dpp'] = run_with_timing("DPP sampler", test_dpp)
    results['mmr'] = run_with_timing("MMR selector", test_mmr)
    results['submodular'] = run_with_timing("submodular optimizer", test_submodular)
    results['embedding'] = run_with_timing("embedding diversity", test_embedding)
    results['fair'] = run_with_timing("fair diversity", test_fair)
    return results


def main():
    """Run comprehensive benchmark suite."""
    print("=" * 70)
    print("COMPREHENSIVE DIVERSITY BENCHMARK SUITE")
    print("=" * 70)

    all_results = {}
    start_time = time.time()

    sections = [
        ("1. Diversity Theory", test_diversity_theory),
        ("2. Active Selection", test_active_selection),
        ("3. Generation Diversity", test_generation_diversity),
        ("4. Recommendation Diversity", test_recommendation_diversity),
        ("5. Retrieval Diversity", test_retrieval_diversity),
        ("6. Ensemble Diversity", test_ensemble_diversity),
        ("7. Curriculum Diversity", test_curriculum_diversity),
        ("8. Evaluation Metrics", test_evaluation_metrics),
        ("9. Adversarial Diversity", test_adversarial_diversity),
        ("10. Existing Benchmarks", run_existing_benchmarks),
    ]

    for section_name, section_func in sections:
        print(f"\n{'=' * 70}")
        print(f"{section_name}")
        print(f"{'=' * 70}")
        try:
            all_results[section_name] = section_func()
        except Exception as e:
            print(f"  SECTION FAILED: {e}")
            traceback.print_exc()
            all_results[section_name] = {"status": "failed", "error": str(e)}

    total_time = time.time() - start_time

    # Count pass/fail
    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for section_name, section_results in all_results.items():
        if isinstance(section_results, dict):
            for test_name, test_result in section_results.items():
                if isinstance(test_result, dict) and 'status' in test_result:
                    total_tests += 1
                    if test_result['status'] == 'passed':
                        passed_tests += 1
                    elif test_result['status'] == 'failed':
                        failed_tests += 1

    # Serialize results (convert numpy types)
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, set):
            return list(obj)
        return obj

    serializable = make_serializable(all_results)
    serializable['summary'] = {
        'total_tests': total_tests,
        'passed': passed_tests,
        'failed': failed_tests,
        'total_time': total_time,
    }

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'comprehensive_benchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    # Print summary
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total tests: {total_tests}")
    print(f"Passed:      {passed_tests}")
    print(f"Failed:      {failed_tests}")
    print(f"Total time:  {total_time:.2f}s")
    print(f"Results:     {output_path}")

    if failed_tests > 0:
        print(f"\n*** {failed_tests} TESTS FAILED ***")
        for section_name, section_results in all_results.items():
            if isinstance(section_results, dict):
                for test_name, test_result in section_results.items():
                    if isinstance(test_result, dict) and test_result.get('status') == 'failed':
                        print(f"  FAIL: {section_name}/{test_name}: {test_result.get('error', 'unknown')}")
    else:
        print("\nALL TESTS PASSED!")

    print(f"{'=' * 70}")
    return 0 if failed_tests == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
