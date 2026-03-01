#!/usr/bin/env python3
"""
Real EPD convergence study: pool texts across seeds to get n=30 per (config, prompt),
then subsample at n={5,8,10,15,20,30} to study how EPD correlation changes with sample size.
"""
import json, os, sys, time, itertools
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_unified_experiment import (
    distinct_n, self_bleu, ngram_entropy, embedding_pairwise_distance,
    vendi_score, jaccard_diversity, compression_ratio_diversity,
    unique_sentence_ratio, type_token_ratio, compute_all_metrics, kendall_tau_pair
)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'scaled_results', 'taxonomy_texts.json')) as f:
        all_data = json.load(f)

    # Pool texts across seeds: for each (config, prompt), combine 3 seeds × 10 texts = 30
    pooled = {}
    for key, texts in all_data.items():
        parts = key.split('__')
        config, seed, prompt_id = parts[0], parts[1], parts[2]
        pool_key = (config, prompt_id)
        if pool_key not in pooled:
            pooled[pool_key] = []
        pooled[pool_key].extend(texts)

    configs = sorted(set(k[0] for k in pooled.keys()))
    prompts = sorted(set(k[1] for k in pooled.keys()))
    print(f"Pooled data: {len(configs)} configs × {len(prompts)} prompts")
    print(f"Texts per pool: {len(pooled[(configs[0], prompts[0])])}")

    sample_sizes = [5, 8, 10, 15, 20, 30]
    focus_metrics = ['distinct_2', 'self_bleu', 'epd', 'vendi', 'jaccard', 'crd', 'ttr']
    n_bootstrap = 10  # bootstrap repetitions for each sample size

    np.random.seed(42)
    results = {}

    for n_sample in sample_sizes:
        print(f"\n  Testing n={n_sample}...")
        per_prompt_taus = {f'{m1}_vs_{m2}': [] for m1, m2 in itertools.combinations(focus_metrics, 2)}

        for prompt_id in prompts:
            for boot in range(n_bootstrap):
                vectors = {m: [] for m in focus_metrics}
                valid = True
                for config in configs:
                    pool_key = (config, prompt_id)
                    if pool_key not in pooled:
                        valid = False
                        break
                    all_texts = pooled[pool_key]
                    # Subsample
                    if n_sample < len(all_texts):
                        indices = np.random.choice(len(all_texts), n_sample, replace=False)
                        texts = [all_texts[i] for i in indices]
                    else:
                        texts = all_texts
                    metrics = compute_all_metrics(texts)
                    for m in focus_metrics:
                        vectors[m].append(metrics[m])
                if not valid:
                    continue
                for m1, m2 in itertools.combinations(focus_metrics, 2):
                    tau = kendall_tau_pair(vectors[m1], vectors[m2])
                    per_prompt_taus[f'{m1}_vs_{m2}'].append(tau)

        results[str(n_sample)] = {}
        for pair_key, vals in per_prompt_taus.items():
            if vals:
                results[str(n_sample)][pair_key] = {
                    'mean_tau': round(float(np.mean(vals)), 4),
                    'std_tau': round(float(np.std(vals)), 4),
                    'n_replicates': len(vals),
                }

    # Print key results
    print("\n" + "=" * 70)
    print("EPD CONVERGENCE RESULTS (pooled across seeds, bootstrap)")
    print("=" * 70)
    key_pairs = ['distinct_2_vs_epd', 'epd_vs_vendi', 'distinct_2_vs_self_bleu',
                 'distinct_2_vs_crd', 'distinct_2_vs_ttr']
    for pair in key_pairs:
        print(f"\n  {pair}:")
        for n_s in sample_sizes:
            s = results[str(n_s)].get(pair, {})
            if s:
                print(f"    n={n_s:>2d}: τ = {s['mean_tau']:+.4f} ± {s['std_tau']:.4f}  (k={s['n_replicates']})")

    # Save
    output_path = os.path.join(base_dir, 'unified_results', 'epd_convergence_real.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

if __name__ == '__main__':
    main()
