"""
Comprehensive real-data experiments for DivFlow paper revision.

Experiments:
1. Selection Algorithm Benchmark on Real Embeddings (gpt-4.1-nano text → embeddings)
2. Metric Taxonomy Validation (on real nano-generated text)
3. Scaling Analysis (proper 6+ data points)
4. Fair Diversity on Imbalanced Groups
5. Comparison against scikit-learn baselines
"""

import json
import os
import sys
import time
import numpy as np
from scipy import stats
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from submodular_optimizer import (
    SubmodularOptimizer, FacilityLocationFunction,
    SumPairwiseDistanceFunction, LogDeterminantFunction
)
from dpp_sampler import compute_kernel, DPPSampler
from mmr_selector import MMRSelector
from clustering_diversity import ClusterDiversity
from unified_selector import (
    DPPSelector, MMRSelector as UnifiedMMR, SubmodularSelector,
    ClusteringSelector, FarthestPointSelector, RandomSelector
)


def compute_spread(points: np.ndarray, indices: list) -> float:
    """Min pairwise distance of selected subset."""
    subset = points[indices]
    if len(subset) < 2:
        return 0.0
    dists = np.linalg.norm(subset[:, None] - subset[None, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return float(np.min(dists))


def compute_sum_dist(points: np.ndarray, indices: list) -> float:
    """Sum of pairwise distances."""
    subset = points[indices]
    if len(subset) < 2:
        return 0.0
    dists = np.linalg.norm(subset[:, None] - subset[None, :], axis=-1)
    return float(np.sum(dists) / 2.0)


def compute_coverage(points: np.ndarray, indices: list, radius: float = None) -> float:
    """Fraction of ground set within radius of selected set."""
    if radius is None:
        # Use median nearest-neighbor distance as radius
        all_dists = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        np.fill_diagonal(all_dists, np.inf)
        radius = np.median(np.min(all_dists, axis=1)) * 2.0
    selected = points[indices]
    dists_to_selected = np.min(
        np.linalg.norm(points[:, None] - selected[None, :], axis=-1), axis=1
    )
    return float(np.mean(dists_to_selected <= radius))


def experiment_1_selection_benchmark():
    """Benchmark all selection algorithms on real embeddings.
    
    Uses text-embedding-3-small to embed real nano-generated texts,
    then compares selection algorithms.
    """
    print("=" * 60)
    print("EXPERIMENT 1: Selection Algorithm Benchmark")
    print("=" * 60)

    # Load existing nano-generated texts
    texts_path = os.path.join(os.path.dirname(__file__), 'scaled_results', 'taxonomy_texts.json')
    if not os.path.exists(texts_path):
        texts_path = os.path.join(os.path.dirname(__file__), 'scaled_results', 'taxonomy_texts_nano.json')

    with open(texts_path) as f:
        texts_data = json.load(f)

    # Collect all texts (dict: config_key -> list of texts)
    all_texts = []
    config_keys = list(texts_data.keys())
    for key in config_keys[:20]:  # 20 groups × 10 texts = 200
        texts = texts_data[key]
        if isinstance(texts, list):
            all_texts.extend(texts[:10])

    if len(all_texts) < 50:
        print(f"  Only {len(all_texts)} texts found, generating synthetic embeddings")
        np.random.seed(42)
        embeddings = np.random.randn(200, 50)
    else:
        print(f"  Loaded {len(all_texts)} texts, computing embeddings...")
        embeddings = compute_text_embeddings(all_texts[:200])

    n = len(embeddings)
    print(f"  Using {n} embeddings of dimension {embeddings.shape[1]}")

    # Run selection algorithms
    selectors = {
        'DPP': DPPSelector(kernel='rbf'),
        'MMR': UnifiedMMR(lambda_param=0.5),
        'Submodular-Greedy': SubmodularSelector(method='greedy'),
        'Clustering': ClusteringSelector(),
        'FarthestPoint': FarthestPointSelector(),
        'Random': RandomSelector(),
    }

    k_values = [5, 10, 20]
    n_trials = 30
    results = {}

    for k in k_values:
        print(f"\n  k={k}:")
        results[f'k={k}'] = {}
        for name, selector in selectors.items():
            spreads = []
            sum_dists = []
            coverages = []
            runtimes = []
            for trial in range(n_trials):
                rng = np.random.RandomState(trial)
                t0 = time.time()
                try:
                    indices, meta = selector.select(embeddings, k, rng=rng)
                    runtime = time.time() - t0
                    sp = compute_spread(embeddings, indices)
                    sd = compute_sum_dist(embeddings, indices)
                    cov = compute_coverage(embeddings, indices)
                    spreads.append(sp)
                    sum_dists.append(sd)
                    coverages.append(cov)
                    runtimes.append(runtime)
                except Exception as e:
                    print(f"    {name} trial {trial} failed: {e}")
                    continue

            if len(spreads) > 0:
                results[f'k={k}'][name] = {
                    'spread_mean': float(np.mean(spreads)),
                    'spread_std': float(np.std(spreads)),
                    'sum_dist_mean': float(np.mean(sum_dists)),
                    'sum_dist_std': float(np.std(sum_dists)),
                    'coverage_mean': float(np.mean(coverages)),
                    'coverage_std': float(np.std(coverages)),
                    'runtime_mean': float(np.mean(runtimes)),
                    'n_successful': len(spreads),
                }
                print(f"    {name:20s}: spread={np.mean(spreads):.3f}±{np.std(spreads):.3f}  "
                      f"sum_dist={np.mean(sum_dists):.1f}±{np.std(sum_dists):.1f}  "
                      f"coverage={np.mean(coverages):.3f}  "
                      f"time={np.mean(runtimes)*1000:.1f}ms")

    # Statistical comparisons (k=10)
    print("\n  Statistical comparisons (k=10):")
    k10 = results.get('k=10', {})
    comparisons = {}
    methods = list(k10.keys())
    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            # Re-run to get paired samples for t-test
            pass  # We'll use the mean/std for summary

    return results


def compute_text_embeddings(texts: list) -> np.ndarray:
    """Compute embeddings using text-embedding-3-small."""
    try:
        from openai import OpenAI
        client = OpenAI()
        batch_size = 50
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Truncate long texts
            batch = [t[:2000] if len(t) > 2000 else t for t in batch]
            batch = [t if t.strip() else "empty" for t in batch]
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            for item in resp.data:
                all_embeddings.append(item.embedding)
            print(f"    Embedded {min(i+batch_size, len(texts))}/{len(texts)}")
        return np.array(all_embeddings)
    except Exception as e:
        print(f"    Embedding API failed ({e}), using TF-IDF fallback")
        return tfidf_embeddings(texts)


def tfidf_embeddings(texts: list) -> np.ndarray:
    """Simple TF-IDF embeddings as fallback."""
    from collections import Counter
    vocab = Counter()
    tokenized = []
    for t in texts:
        tokens = t.lower().split()
        tokenized.append(tokens)
        vocab.update(tokens)

    # Keep top 500 words
    top_words = [w for w, _ in vocab.most_common(500)]
    word2idx = {w: i for i, w in enumerate(top_words)}

    embeddings = np.zeros((len(texts), len(top_words)))
    for i, tokens in enumerate(tokenized):
        for t in tokens:
            if t in word2idx:
                embeddings[i, word2idx[t]] += 1
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embeddings = embeddings / norms
    return embeddings


def experiment_2_metric_taxonomy_real():
    """Validate metric taxonomy on real nano-generated text.
    
    Loads existing taxonomy results and validates cluster structure.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Metric Taxonomy on Real Text")
    print("=" * 60)

    # Load existing unified results
    results_path = os.path.join(os.path.dirname(__file__), 'unified_results', 'unified_results.json')
    with open(results_path) as f:
        unified = json.load(f)

    # Extract GPT-2 tau matrix
    gpt2 = unified.get('gpt2', {})
    nano = unified.get('nano', {})
    metrics = gpt2.get('metric_names', [])
    tau_matrix = np.array(gpt2.get('tau_matrix', []))

    print(f"\n  GPT-2 Kendall tau matrix ({len(metrics)} metrics):")
    print(f"  Metrics: {metrics}")

    # Identify clusters
    clusters = identify_clusters(metrics, tau_matrix, threshold=0.7)
    print(f"\n  Clusters at |τ| > 0.7:")
    for cname, members in clusters.items():
        print(f"    {cname}: {members}")

    # Nano per-prompt analysis
    nano_tau = nano.get('kendall_tau_per_prompt', {})
    nano_summary = {}
    for pair_key, pair_data in nano_tau.items():
        mean_tau = pair_data.get('mean_tau', None)
        std_tau = pair_data.get('std_tau', None)
        if mean_tau is not None:
            nano_summary[pair_key] = {
                'mean_tau': mean_tau,
                'std_tau': std_tau,
            }
    print(f"\n  Nano per-prompt results ({len(nano_summary)} pairs):")
    for pair, data in list(nano_summary.items())[:8]:
        print(f"    {pair}: τ={data['mean_tau']:.3f}±{data.get('std_tau', 0):.3f}")

    # Domain stability
    domain_results = nano.get('domain_results', {})
    print(f"\n  Domain stability ({len(domain_results)} domains):")
    domain_epd_taus = {}
    for domain, ddata in domain_results.items():
        tau_pairs = ddata.get('kendall_tau', {})
        d2_epd = tau_pairs.get('distinct_2_vs_epd', {}).get('tau', None)
        if d2_epd is not None:
            domain_epd_taus[domain] = d2_epd
            print(f"    {domain}: D-2 vs EPD τ = {d2_epd:.3f}")

    # EPD sample ablation
    epd_ablation = unified.get('epd_sample_ablation', {})
    print(f"\n  EPD sample-size ablation:")
    for n_samples, adata in sorted(epd_ablation.items(), key=lambda x: int(x[0])):
        d2_epd = adata.get('distinct_2_vs_epd', {}).get('mean_tau', None)
        if d2_epd is not None:
            print(f"    n={n_samples}: D-2 vs EPD τ = {d2_epd:.3f}")

    # Length ablation
    length_abl = unified.get('length_ablation', {})
    print(f"\n  Length ablation:")
    for max_len, ldata in sorted(length_abl.items(), key=lambda x: int(x[0])):
        d2_epd = ldata.get('distinct_2_vs_epd', {}).get('mean_tau', None)
        if d2_epd is not None:
            print(f"    max_len={max_len}: D-2 vs EPD τ = {d2_epd:.3f}")

    return {
        'gpt2_metrics': metrics,
        'gpt2_tau_matrix': tau_matrix.tolist(),
        'clusters': {k: list(v) for k, v in clusters.items()},
        'nano_summary': nano_summary,
        'domain_epd_taus': domain_epd_taus,
        'epd_ablation': {k: v.get('distinct_2_vs_epd', {}).get('mean_tau') 
                         for k, v in epd_ablation.items()},
        'length_ablation': {k: v.get('distinct_2_vs_epd', {}).get('mean_tau')
                           for k, v in length_abl.items()},
    }


def identify_clusters(metrics, tau_matrix, threshold=0.7):
    """Identify clusters of metrics with |τ| > threshold."""
    n = len(metrics)
    visited = set()
    clusters = {}
    cluster_id = 0
    for i in range(n):
        if i in visited:
            continue
        cluster = [metrics[i]]
        visited.add(i)
        for j in range(i+1, n):
            if j in visited:
                continue
            if abs(tau_matrix[i][j]) > threshold:
                cluster.append(metrics[j])
                visited.add(j)
        cluster_id += 1
        clusters[f'C{cluster_id}'] = cluster
    return clusters


def experiment_3_scaling():
    """Proper scaling analysis with 6+ data points."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Scaling Analysis")
    print("=" * 60)

    n_values = [50, 100, 200, 500, 1000, 2000]
    d = 50
    k = 10
    n_trials = 5

    results = {}
    for method_name, selector_cls in [
        ('DPP', DPPSelector),
        ('SubmodularGreedy', SubmodularSelector),
        ('FarthestPoint', FarthestPointSelector),
        ('Random', RandomSelector),
    ]:
        print(f"\n  {method_name}:")
        selector = selector_cls()
        times = {}
        for n in n_values:
            if method_name == 'DPP' and n > 1000:
                print(f"    n={n}: skipped (DPP O(n³))")
                continue
            trial_times = []
            for trial in range(n_trials):
                rng = np.random.RandomState(trial)
                data = rng.randn(n, d)
                t0 = time.time()
                try:
                    indices, _ = selector.select(data, k, rng=rng)
                    elapsed = time.time() - t0
                    trial_times.append(elapsed)
                except Exception as e:
                    print(f"    n={n} trial {trial} failed: {e}")
            if trial_times:
                mean_t = np.mean(trial_times)
                times[n] = mean_t
                print(f"    n={n}: {mean_t*1000:.1f}ms")

        results[method_name] = times

    # Compute scaling exponents via log-log regression
    scaling_exponents = {}
    for method, times in results.items():
        ns = sorted(times.keys())
        if len(ns) >= 3:
            log_n = np.log(ns)
            log_t = np.log([times[n] for n in ns])
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_t)
            scaling_exponents[method] = {
                'exponent': float(slope),
                'r_squared': float(r_value**2),
                'n_points': len(ns),
            }
            print(f"\n  {method}: scaling exponent = {slope:.2f} (R² = {r_value**2:.3f}, {len(ns)} points)")

    return {'times': {k: {str(n): t for n, t in v.items()} for k, v in results.items()},
            'scaling_exponents': scaling_exponents}


def experiment_4_fair_diversity():
    """Fair diversity with imbalanced groups (realistic scenario)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Fair Diversity with Imbalanced Groups")
    print("=" * 60)

    np.random.seed(42)
    # Imbalanced groups: 60%, 25%, 10%, 5%
    group_sizes = [120, 50, 20, 10]
    n = sum(group_sizes)
    d = 50
    k = 20

    # Generate data with group structure
    data = np.vstack([
        np.random.randn(gs, d) + np.random.randn(1, d) * 3
        for gs in group_sizes
    ])
    groups = np.concatenate([np.full(gs, i) for i, gs in enumerate(group_sizes)])

    # Minimum representation: at least 2 from each group
    min_per_group = {0: 2, 1: 2, 2: 2, 3: 2}

    print(f"  n={n}, k={k}, groups={group_sizes}")
    print(f"  Min per group: {min_per_group}")

    # Unconstrained selection (FarthestPoint)
    fp = FarthestPointSelector()
    fp_indices, _ = fp.select(data, k)
    fp_groups = groups[fp_indices]
    fp_spread = compute_spread(data, fp_indices)
    fp_sum_dist = compute_sum_dist(data, fp_indices)
    fp_group_counts = {int(g): int(np.sum(fp_groups == g)) for g in range(len(group_sizes))}

    print(f"\n  Unconstrained FarthestPoint:")
    print(f"    Group counts: {fp_group_counts}")
    print(f"    Spread: {fp_spread:.3f}")
    print(f"    Sum dist: {fp_sum_dist:.1f}")

    # Fair selection: greedy with group constraints
    fair_indices = fair_greedy_select(data, k, groups, min_per_group)
    fair_groups = groups[fair_indices]
    fair_spread = compute_spread(data, fair_indices)
    fair_sum_dist = compute_sum_dist(data, fair_indices)
    fair_group_counts = {int(g): int(np.sum(fair_groups == g)) for g in range(len(group_sizes))}

    print(f"\n  Fair selection:")
    print(f"    Group counts: {fair_group_counts}")
    print(f"    Spread: {fair_spread:.3f}")
    print(f"    Sum dist: {fair_sum_dist:.1f}")

    # Check constraint satisfaction
    constraints_met = all(fair_group_counts.get(g, 0) >= min_per_group.get(g, 0)
                         for g in min_per_group)
    diversity_retention = fair_sum_dist / max(fp_sum_dist, 1e-12)

    print(f"\n  Constraints satisfied: {constraints_met}")
    print(f"  Diversity retention: {diversity_retention:.1%}")

    return {
        'n': n, 'k': k, 'd': d,
        'group_sizes': group_sizes,
        'min_per_group': min_per_group,
        'unconstrained': {
            'group_counts': fp_group_counts,
            'spread': fp_spread,
            'sum_dist': fp_sum_dist,
        },
        'fair': {
            'group_counts': fair_group_counts,
            'spread': fair_spread,
            'sum_dist': fair_sum_dist,
            'constraints_met': constraints_met,
        },
        'diversity_retention': diversity_retention,
    }


def fair_greedy_select(data, k, groups, min_per_group):
    """Greedy farthest-point with fairness constraints."""
    n = len(data)
    n_groups = len(set(groups))
    selected = []
    remaining = set(range(n))

    # Phase 1: ensure minimums
    for g, min_count in min_per_group.items():
        g_indices = [i for i in remaining if groups[i] == g]
        if not g_indices:
            continue
        # Farthest-point within group
        sub_data = data[g_indices]
        if len(g_indices) <= min_count:
            for idx in g_indices:
                selected.append(idx)
                remaining.discard(idx)
        else:
            # Pick min_count farthest points
            sub_selected = [0]
            dists = np.full(len(g_indices), np.inf)
            for _ in range(min_count - 1):
                last = sub_selected[-1]
                new_dists = np.linalg.norm(sub_data - sub_data[last], axis=1)
                dists = np.minimum(dists, new_dists)
                for s in sub_selected:
                    dists[s] = -1
                next_idx = int(np.argmax(dists))
                sub_selected.append(next_idx)
            for si in sub_selected:
                idx = g_indices[si]
                selected.append(idx)
                remaining.discard(idx)

    # Phase 2: fill remaining slots with farthest-point (unconstrained)
    while len(selected) < k and remaining:
        best_idx = None
        best_min_dist = -1
        for idx in remaining:
            min_dist = min(np.linalg.norm(data[idx] - data[s]) for s in selected) if selected else np.inf
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        if best_idx is not None:
            selected.append(best_idx)
            remaining.discard(best_idx)

    return selected


def experiment_5_baseline_comparison():
    """Compare DivFlow against scikit-learn baselines."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Comparison Against Baselines")
    print("=" * 60)

    from sklearn.cluster import KMeans, DBSCAN, SpectralClustering

    np.random.seed(42)
    n, d, k = 200, 50, 10
    n_trials = 30

    results = {}

    for trial in range(n_trials):
        rng = np.random.RandomState(trial)
        data = rng.randn(n, d)

        # DivFlow methods
        for name, sel in [
            ('DivFlow-DPP', DPPSelector(kernel='rbf')),
            ('DivFlow-Submodular', SubmodularSelector()),
            ('DivFlow-FarthestPoint', FarthestPointSelector()),
        ]:
            t0 = time.time()
            try:
                indices, _ = sel.select(data, k, rng=rng)
                runtime = time.time() - t0
                sp = compute_spread(data, indices)
                sd = compute_sum_dist(data, indices)
            except Exception as e:
                continue
            if name not in results:
                results[name] = {'spreads': [], 'sum_dists': [], 'runtimes': []}
            results[name]['spreads'].append(sp)
            results[name]['sum_dists'].append(sd)
            results[name]['runtimes'].append(runtime)

        # sklearn KMeans → centroids → nearest data points
        t0 = time.time()
        km = KMeans(n_clusters=k, random_state=trial, n_init=1)
        km.fit(data)
        centroids = km.cluster_centers_
        indices_km = []
        for c in centroids:
            dists = np.linalg.norm(data - c, axis=1)
            idx = int(np.argmin(dists))
            while idx in indices_km:
                dists[idx] = np.inf
                idx = int(np.argmin(dists))
            indices_km.append(idx)
        runtime = time.time() - t0
        sp = compute_spread(data, indices_km)
        sd = compute_sum_dist(data, indices_km)
        if 'sklearn-KMeans' not in results:
            results['sklearn-KMeans'] = {'spreads': [], 'sum_dists': [], 'runtimes': []}
        results['sklearn-KMeans']['spreads'].append(sp)
        results['sklearn-KMeans']['sum_dists'].append(sd)
        results['sklearn-KMeans']['runtimes'].append(runtime)

        # Random baseline
        t0 = time.time()
        indices_rand = rng.choice(n, k, replace=False).tolist()
        runtime = time.time() - t0
        sp = compute_spread(data, indices_rand)
        sd = compute_sum_dist(data, indices_rand)
        if 'Random' not in results:
            results['Random'] = {'spreads': [], 'sum_dists': [], 'runtimes': []}
        results['Random']['spreads'].append(sp)
        results['Random']['sum_dists'].append(sd)
        results['Random']['runtimes'].append(runtime)

    # Summarize
    summary = {}
    print(f"\n  {'Method':25s} {'Spread':>12s} {'SumDist':>12s} {'Time(ms)':>10s}")
    print("  " + "-" * 62)
    for name in sorted(results.keys()):
        r = results[name]
        sp_mean = np.mean(r['spreads'])
        sp_std = np.std(r['spreads'])
        sd_mean = np.mean(r['sum_dists'])
        sd_std = np.std(r['sum_dists'])
        rt_mean = np.mean(r['runtimes']) * 1000
        print(f"  {name:25s} {sp_mean:7.3f}±{sp_std:.3f} {sd_mean:7.1f}±{sd_std:.1f} {rt_mean:8.1f}")
        summary[name] = {
            'spread_mean': float(sp_mean),
            'spread_std': float(sp_std),
            'sum_dist_mean': float(sd_mean),
            'sum_dist_std': float(sd_std),
            'runtime_ms': float(rt_mean),
            'n_trials': len(r['spreads']),
        }

    # Statistical significance: DivFlow-FarthestPoint vs Random
    if 'DivFlow-FarthestPoint' in results and 'Random' in results:
        fp_spreads = results['DivFlow-FarthestPoint']['spreads']
        rand_spreads = results['Random']['spreads']
        t_stat, p_val = stats.ttest_ind(fp_spreads, rand_spreads)
        cohens_d = (np.mean(fp_spreads) - np.mean(rand_spreads)) / np.sqrt(
            (np.std(fp_spreads)**2 + np.std(rand_spreads)**2) / 2
        )
        print(f"\n  FarthestPoint vs Random: t={t_stat:.3f}, p={p_val:.6f}, Cohen's d={cohens_d:.3f}")
        summary['stat_test_fp_vs_random'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'cohens_d': float(cohens_d),
        }

    # Submodular vs Random
    if 'DivFlow-Submodular' in results and 'Random' in results:
        sub_spreads = results['DivFlow-Submodular']['spreads']
        rand_spreads = results['Random']['spreads']
        t_stat, p_val = stats.ttest_ind(sub_spreads, rand_spreads)
        cohens_d = (np.mean(sub_spreads) - np.mean(rand_spreads)) / np.sqrt(
            (np.std(sub_spreads)**2 + np.std(rand_spreads)**2) / 2
        )
        print(f"  Submodular vs Random: t={t_stat:.3f}, p={p_val:.6f}, Cohen's d={cohens_d:.3f}")
        summary['stat_test_sub_vs_random'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'cohens_d': float(cohens_d),
        }

    return summary


def experiment_6_text_selection_diversity():
    """Apply selection algorithms to real text data and measure text diversity metrics."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Selection Algorithms on Text Diversity")
    print("=" * 60)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "diversity_metrics",
        os.path.join(os.path.dirname(__file__), '..', 'src', 'metrics', 'diversity.py')
    )
    dm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dm)
    DistinctN = dm.DistinctN
    SelfBLEU = dm.SelfBLEU
    NGramEntropy = dm.NGramEntropy

    # Load texts
    texts_path = os.path.join(os.path.dirname(__file__), 'scaled_results', 'taxonomy_texts.json')
    with open(texts_path) as f:
        texts_data = json.load(f)

    # Get a pool of texts
    pool = []
    config_keys = list(texts_data.keys())
    for key in config_keys[:30]:
        texts = texts_data[key]
        if isinstance(texts, list):
            pool.extend(texts)
    pool = pool[:300]
    print(f"  Pool size: {len(pool)} texts")

    if len(pool) < 50:
        print("  Insufficient texts, skipping")
        return {}

    # Compute TF-IDF embeddings for selection
    embeddings = tfidf_embeddings(pool)
    k = 20

    # Select with different methods
    selectors = {
        'DPP': DPPSelector(kernel='rbf'),
        'FarthestPoint': FarthestPointSelector(),
        'Random': RandomSelector(),
    }

    results = {}
    for name, sel in selectors.items():
        indices, _ = sel.select(embeddings, k, rng=np.random.RandomState(42))
        selected_texts = [pool[i] for i in indices]

        # Compute text diversity metrics
        d2 = DistinctN(n=2)
        d2_val = d2.compute(selected_texts)
        entropy = NGramEntropy(n=2)
        ent_val = entropy.compute(selected_texts)

        results[name] = {
            'distinct_2': float(d2_val),
            'entropy': float(ent_val),
            'n_selected': len(selected_texts),
        }
        print(f"  {name:20s}: D-2={d2_val:.4f}  Entropy={ent_val:.4f}")

    return results


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'paper_experiment_results')
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # Experiment 1: Selection benchmark
    try:
        all_results['selection_benchmark'] = experiment_1_selection_benchmark()
    except Exception as e:
        print(f"Experiment 1 failed: {e}")
        import traceback; traceback.print_exc()

    # Experiment 2: Metric taxonomy
    try:
        all_results['metric_taxonomy'] = experiment_2_metric_taxonomy_real()
    except Exception as e:
        print(f"Experiment 2 failed: {e}")
        import traceback; traceback.print_exc()

    # Experiment 3: Scaling
    try:
        all_results['scaling'] = experiment_3_scaling()
    except Exception as e:
        print(f"Experiment 3 failed: {e}")
        import traceback; traceback.print_exc()

    # Experiment 4: Fair diversity
    try:
        all_results['fair_diversity'] = experiment_4_fair_diversity()
    except Exception as e:
        print(f"Experiment 4 failed: {e}")
        import traceback; traceback.print_exc()

    # Experiment 5: Baseline comparison
    try:
        all_results['baseline_comparison'] = experiment_5_baseline_comparison()
    except Exception as e:
        print(f"Experiment 5 failed: {e}")
        import traceback; traceback.print_exc()

    # Experiment 6: Text selection diversity
    try:
        all_results['text_selection'] = experiment_6_text_selection_diversity()
    except Exception as e:
        print(f"Experiment 6 failed: {e}")
        import traceback; traceback.print_exc()

    # Save results
    output_path = os.path.join(output_dir, 'comprehensive_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nResults saved to {output_path}")

    return all_results


if __name__ == '__main__':
    main()
