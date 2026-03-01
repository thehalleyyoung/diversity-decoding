#!/usr/bin/env python3
"""
GPT-2 Expanded Taxonomy + Length Ablation Experiment.
Computes 10 metrics on GPT-2 data and performs length ablation on nano data.
"""

import json
import os
import sys
import time
import zlib
import re
import itertools
from collections import Counter
import numpy as np
from scipy import stats


# Import metrics from the expanded taxonomy script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_expanded_taxonomy import (
    distinct_n, self_bleu, ngram_entropy, embedding_pairwise_distance,
    vendi_score, jaccard_diversity, pos_sequence_diversity,
    compression_ratio_diversity, unique_sentence_ratio,
    semantic_embedding_diversity, get_embeddings_batch
)


def compute_metrics_no_sed(texts):
    """Compute 9 metrics (no SED) for a set of texts."""
    return {
        'distinct_2': distinct_n(texts, n=2),
        'self_bleu': self_bleu(texts),
        'entropy': ngram_entropy(texts, n=2),
        'epd': embedding_pairwise_distance(texts),
        'vendi': vendi_score(texts),
        'jaccard': jaccard_diversity(texts),
        'ptd': pos_sequence_diversity(texts),
        'crd': compression_ratio_diversity(texts),
        'usr': unique_sentence_ratio(texts),
    }


def truncate_texts(texts, max_tokens):
    """Truncate texts to max_tokens words."""
    return [' '.join(t.split()[:max_tokens]) for t in texts]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'expanded_results')
    os.makedirs(output_dir, exist_ok=True)

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    # === PART 1: GPT-2 Expanded Taxonomy ===
    print("=" * 60)
    print("PART 1: GPT-2 EXPANDED TAXONOMY (10 metrics)")
    print("=" * 60)

    gpt2_path = os.path.join(base_dir, 'real_results', 'h1_generated_texts.json')
    with open(gpt2_path) as f:
        gpt2_data = json.load(f)

    configs = sorted(gpt2_data.keys())
    metric_names_9 = ['distinct_2', 'self_bleu', 'entropy', 'epd', 'vendi',
                      'jaccard', 'ptd', 'crd', 'usr']
    metric_names_10 = metric_names_9 + ['sed']

    # Get embeddings for all GPT-2 texts
    all_gpt2_texts = []
    for config in configs:
        all_gpt2_texts.extend(gpt2_data[config])
    unique_gpt2 = sorted(set(all_gpt2_texts))
    print(f"\nGetting embeddings for {len(unique_gpt2)} unique GPT-2 texts...")
    embeddings = get_embeddings_batch(unique_gpt2, api_key)
    text_to_emb = dict(zip(unique_gpt2, embeddings))

    # Compute metrics per config
    print("Computing 10 metrics for GPT-2 configs...")
    gpt2_metrics = {}
    for config in configs:
        texts = gpt2_data[config]
        embs = [text_to_emb[t] for t in texts]
        metrics = compute_metrics_no_sed(texts)
        metrics['sed'] = semantic_embedding_diversity(texts, embs)
        gpt2_metrics[config] = metrics

    # Compute Kendall tau matrix
    metric_vectors = {m: [gpt2_metrics[c][m] for c in configs] for m in metric_names_10}
    n_metrics = len(metric_names_10)
    tau_matrix = np.eye(n_metrics)
    rho_matrix = np.eye(n_metrics)
    for i in range(n_metrics):
        for j in range(i+1, n_metrics):
            tau, _ = stats.kendalltau(metric_vectors[metric_names_10[i]],
                                      metric_vectors[metric_names_10[j]])
            rho, _ = stats.spearmanr(metric_vectors[metric_names_10[i]],
                                      metric_vectors[metric_names_10[j]])
            if np.isnan(tau): tau = 0.0
            if np.isnan(rho): rho = 0.0
            tau_matrix[i, j] = tau
            tau_matrix[j, i] = tau
            rho_matrix[i, j] = rho
            rho_matrix[j, i] = rho

    print("\nGPT-2 Kendall tau matrix (10 metrics):")
    header = f"{'':>12s}" + "".join(f"{m:>8s}" for m in metric_names_10)
    print(header)
    for i, m in enumerate(metric_names_10):
        row = f"{m:>12s}" + "".join(f"{tau_matrix[i,j]:>8.3f}" for j in range(n_metrics))
        print(row)

    # Key new pairs
    print("\nKey new pairs (GPT-2):")
    new_pairs = [
        ('distinct_2', 'sed'), ('distinct_2', 'crd'), ('distinct_2', 'usr'),
        ('sed', 'epd'), ('sed', 'vendi'), ('sed', 'ptd'),
        ('crd', 'ptd'), ('usr', 'ptd'), ('sed', 'crd'),
    ]
    gpt2_tau_pairs = {}
    for m1, m2 in new_pairs:
        i = metric_names_10.index(m1)
        j = metric_names_10.index(m2)
        tau_val = tau_matrix[i, j]
        rho_val = rho_matrix[i, j]
        print(f"  {m1:>12s} vs {m2:<12s}: tau={tau_val:+.3f}, rho={rho_val:+.3f}")
        gpt2_tau_pairs[f'{m1}_vs_{m2}'] = {'tau': round(tau_val, 4), 'rho': round(rho_val, 4)}

    # === PART 2: LENGTH ABLATION ===
    print("\n" + "=" * 60)
    print("PART 2: LENGTH ABLATION (nano texts truncated to 30 tokens)")
    print("=" * 60)

    nano_path = os.path.join(base_dir, 'scaled_results', 'taxonomy_texts.json')
    with open(nano_path) as f:
        nano_data = json.load(f)

    # Parse nano keys
    nano_groups = {}
    for key, texts in nano_data.items():
        parts = key.split('__')
        config = parts[0]
        seed = parts[1]
        prompt_id = parts[2]
        nano_groups[key] = {
            'config': config, 'seed': seed, 'prompt_id': prompt_id, 'texts': texts
        }

    nano_configs = sorted(set(g['config'] for g in nano_groups.values()))
    nano_seeds = sorted(set(g['seed'] for g in nano_groups.values()))
    nano_prompts = sorted(set(g['prompt_id'] for g in nano_groups.values()))

    # Compute metrics at different truncation levels
    truncation_levels = [30, 50, 100]  # 30 matches GPT-2 length
    ablation_metrics = ['distinct_2', 'self_bleu', 'entropy', 'epd', 'vendi', 'jaccard']

    ablation_results = {}
    for max_tokens in truncation_levels:
        print(f"\n  Computing metrics at truncation={max_tokens} tokens...")
        # Compute per-prompt tau
        per_prompt_taus = {f'{m1}_vs_{m2}': [] for m1, m2 in itertools.combinations(ablation_metrics, 2)}

        for prompt_id in nano_prompts:
            for seed in nano_seeds:
                vectors = {m: [] for m in ablation_metrics}
                valid = True
                for config in nano_configs:
                    key = f"{config}__{seed}__{prompt_id}"
                    if key not in nano_groups:
                        valid = False
                        break
                    truncated = truncate_texts(nano_groups[key]['texts'], max_tokens)
                    metrics = compute_metrics_no_sed(truncated)
                    for m in ablation_metrics:
                        vectors[m].append(metrics[m])

                if not valid:
                    continue

                for m1, m2 in itertools.combinations(ablation_metrics, 2):
                    tau, _ = stats.kendalltau(vectors[m1], vectors[m2])
                    if not np.isnan(tau):
                        per_prompt_taus[f'{m1}_vs_{m2}'].append(tau)

        ablation_results[max_tokens] = {}
        for pair_key, vals in per_prompt_taus.items():
            if vals:
                ablation_results[max_tokens][pair_key] = {
                    'mean_tau': round(np.mean(vals), 4),
                    'std_tau': round(np.std(vals), 4),
                }

    # Print length ablation results
    print("\nLength ablation: EPD vs Vendi (key divergent pair):")
    for max_tokens in truncation_levels:
        pair_key = 'epd_vs_vendi'
        if pair_key in ablation_results[max_tokens]:
            s = ablation_results[max_tokens][pair_key]
            print(f"  {max_tokens:>3d} tokens: tau = {s['mean_tau']:+.3f} ± {s['std_tau']:.3f}")

    print("\nLength ablation: D-2 vs SB (stable pair):")
    for max_tokens in truncation_levels:
        pair_key = 'distinct_2_vs_self_bleu'
        if pair_key in ablation_results[max_tokens]:
            s = ablation_results[max_tokens][pair_key]
            print(f"  {max_tokens:>3d} tokens: tau = {s['mean_tau']:+.3f} ± {s['std_tau']:.3f}")

    print("\nLength ablation: D-2 vs EPD (divergent pair):")
    for max_tokens in truncation_levels:
        pair_key = 'distinct_2_vs_epd'
        if pair_key in ablation_results[max_tokens]:
            s = ablation_results[max_tokens][pair_key]
            print(f"  {max_tokens:>3d} tokens: tau = {s['mean_tau']:+.3f} ± {s['std_tau']:.3f}")

    # === SAVE ALL RESULTS ===
    all_results = {
        'gpt2_expanded': {
            'metric_names': metric_names_10,
            'tau_matrix': tau_matrix.tolist(),
            'rho_matrix': rho_matrix.tolist(),
            'config_metrics': gpt2_metrics,
            'key_new_pairs': gpt2_tau_pairs,
        },
        'length_ablation': {
            'truncation_levels': truncation_levels,
            'metrics_tested': ablation_metrics,
            'results': {str(k): v for k, v in ablation_results.items()},
        }
    }

    output_path = os.path.join(output_dir, 'gpt2_expanded_and_ablation.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()
