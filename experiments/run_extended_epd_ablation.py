#!/usr/bin/env python3
"""
Extended EPD sample-size ablation: generate n=20 and n=50 sequences per config
to test whether EPD converges to the lexical cluster at larger sample sizes.

This directly addresses Reviewer Question #1: "What happens to EPD's correlation
with lexical metrics at n=20, 50?"
"""

import json
import os
import sys
import time
import zlib
import re
import math
import hashlib
import itertools
from collections import Counter
import numpy as np
from scipy import stats

# Import metric functions from unified experiment
from run_unified_experiment import (
    distinct_n, self_bleu, ngram_entropy, embedding_pairwise_distance,
    vendi_score, jaccard_diversity, compression_ratio_diversity,
    unique_sentence_ratio, type_token_ratio, compute_all_metrics,
    kendall_tau_pair
)

# Prompts matching the existing taxonomy_texts.json keys
PROMPTS_BY_DOMAIN = {
    "creative_writing": [
        "Once upon a time in a distant kingdom, there lived",
        "The rain fell softly on the old stone bridge as",
        "She opened the letter and read the first line:",
        "In the year 3000, humanity finally discovered",
    ],
    "code_generation": [
        'def fibonacci(n):\n    """Return the nth Fibonacci number."""\n',
        "# Python function to sort a list using merge sort\ndef merge_sort(",
        'class DatabaseConnection:\n    """Manages database connections."""\n    def __init__(self',
        'async def fetch_data(url: str) -> dict:\n    """Fetch JSON data from',
    ],
    "scientific": [
        "The relationship between quantum entanglement and",
        "Recent advances in CRISPR gene editing have shown that",
        "The standard model of particle physics predicts that",
        "Climate models suggest that by 2050, global temperatures will",
    ],
    "business": [
        "The key to a successful startup strategy is",
        "In today's competitive market, companies must focus on",
        "The quarterly earnings report showed that revenue increased by",
        "To improve customer retention, the most effective approach is",
    ],
    "dialogue": [
        "Customer: I'd like to return this product.\nAgent:",
        "Student: Can you explain how neural networks work?\nTeacher:",
        "User: What's the best way to learn Python?\nAssistant:",
        "Interviewer: Tell me about your biggest achievement.\nCandidate:",
    ],
}

ALL_PROMPTS = []
for domain, prompts in PROMPTS_BY_DOMAIN.items():
    for p in prompts:
        ALL_PROMPTS.append((domain, p))


def generate_openai(prompt, n, temperature, top_p, seed, max_tokens=100):
    """Generate n texts from gpt-4.1-nano."""
    import openai
    client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    kwargs = {
        "model": "gpt-4.1-nano",
        "messages": [{"role": "user", "content": f"Continue this text naturally. Only output the continuation, nothing else:\n\n{prompt}"}],
        "max_tokens": max_tokens,
        "n": n,
        "seed": seed,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p

    response = client.chat.completions.create(**kwargs)
    return [c.message.content for c in response.choices if c.message.content]


def get_prompt_key(prompt):
    """Get the hash key for a prompt matching existing data."""
    return hashlib.md5(prompt.encode()).hexdigest()[:8]


def main():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Run: source ~/.bashrc")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'extended_epd_results')
    os.makedirs(output_dir, exist_ok=True)

    # Load existing texts
    existing_path = os.path.join(base_dir, 'scaled_results', 'taxonomy_texts.json')
    with open(existing_path) as f:
        existing_texts = json.load(f)

    # Configurations matching existing data
    configs = [
        ('temp_0.0', {'temperature': 0.0, 'top_p': None}),
        ('temp_0.3', {'temperature': 0.3, 'top_p': None}),
        ('temp_0.5', {'temperature': 0.5, 'top_p': None}),
        ('temp_0.7', {'temperature': 0.7, 'top_p': None}),
        ('temp_1.0', {'temperature': 1.0, 'top_p': None}),
        ('temp_1.2', {'temperature': 1.2, 'top_p': None}),
        ('temp_1.5', {'temperature': 1.5, 'top_p': None}),
        ('temp_2.0', {'temperature': 2.0, 'top_p': None}),
        ('nucleus_0.3', {'temperature': None, 'top_p': 0.3}),
        ('nucleus_0.5', {'temperature': None, 'top_p': 0.5}),
        ('nucleus_0.7', {'temperature': None, 'top_p': 0.7}),
        ('nucleus_0.9', {'temperature': None, 'top_p': 0.9}),
        ('nucleus_0.95', {'temperature': None, 'top_p': 0.95}),
    ]

    seeds = [42, 137, 2024]
    seed_keys = ['s42', 's137', 's2024']

    # We need 50 texts per group total. We have 10. Generate 40 more.
    # Use 4 batches of 10 to stay within API limits.
    target_additional = 40
    batch_size = 10  # n parameter for API

    # Use a subset of prompts for efficiency (8 prompts across 4 domains)
    # This gives 8 prompts × 3 seeds × 13 configs = 312 groups
    selected_prompts = ALL_PROMPTS[:8]  # creative (4) + code (4)
    # Actually use all 20 for full coverage
    selected_prompts = ALL_PROMPTS

    print("=" * 60)
    print("EXTENDED EPD SAMPLE-SIZE ABLATION")
    print(f"Target: {target_additional} additional texts per group")
    print(f"Prompts: {len(selected_prompts)}, Configs: {len(configs)}, Seeds: {len(seeds)}")
    print(f"Total groups: {len(selected_prompts) * len(configs) * len(seeds)}")
    print("=" * 60)

    # Check for saved progress
    extended_texts_path = os.path.join(output_dir, 'extended_texts.json')
    if os.path.exists(extended_texts_path):
        with open(extended_texts_path) as f:
            extended_texts = json.load(f)
        print(f"Loaded {len(extended_texts)} existing extended text groups")
    else:
        extended_texts = {}

    total_groups = len(selected_prompts) * len(configs) * len(seeds)
    generated = 0
    skipped = 0
    errors = 0
    t_start = time.time()

    for domain, prompt in selected_prompts:
        prompt_key = get_prompt_key(prompt)
        for config_name, config_params in configs:
            for seed, seed_key in zip(seeds, seed_keys):
                group_key = f"{config_name}__{seed_key}__{prompt_key}"

                # Skip if already generated
                if group_key in extended_texts and len(extended_texts[group_key]) >= target_additional:
                    skipped += 1
                    continue

                # Get existing texts
                existing_key = group_key
                if existing_key not in existing_texts:
                    print(f"  WARNING: No existing data for {group_key}, skipping")
                    skipped += 1
                    continue

                # Generate additional texts in batches
                new_texts = extended_texts.get(group_key, [])
                remaining = target_additional - len(new_texts)

                while remaining > 0:
                    batch_n = min(batch_size, remaining)
                    try:
                        # Use different seed offset for new generations
                        gen_seed = seed + 10000 + len(new_texts)
                        temp = config_params.get('temperature')
                        top_p = config_params.get('top_p')

                        # For temp=0.0, all outputs are identical, so skip extra generation
                        if temp == 0.0:
                            # Just duplicate existing (deterministic)
                            batch_texts = existing_texts[existing_key][:batch_n]
                        else:
                            batch_texts = generate_openai(
                                prompt, n=batch_n,
                                temperature=temp if temp is not None else 1.0,
                                top_p=top_p,
                                seed=gen_seed,
                                max_tokens=100
                            )

                        new_texts.extend(batch_texts)
                        remaining -= len(batch_texts)
                    except Exception as e:
                        print(f"  ERROR generating for {group_key}: {e}")
                        errors += 1
                        break

                extended_texts[group_key] = new_texts
                generated += 1

                if generated % 20 == 0:
                    elapsed = time.time() - t_start
                    print(f"  Generated {generated}/{total_groups - skipped} groups ({elapsed:.0f}s, {errors} errors)")
                    # Save progress periodically
                    with open(extended_texts_path, 'w') as f:
                        json.dump(extended_texts, f)

    # Final save
    with open(extended_texts_path, 'w') as f:
        json.dump(extended_texts, f)
    print(f"\nGeneration complete: {generated} new, {skipped} skipped, {errors} errors")
    print(f"Saved to {extended_texts_path}")

    # Now run the ablation analysis
    print("\n" + "=" * 60)
    print("COMPUTING EXTENDED EPD ABLATION")
    print("=" * 60)

    # Merge existing + extended texts
    merged_texts = {}
    for key in existing_texts:
        merged = list(existing_texts[key])
        if key in extended_texts:
            merged.extend(extended_texts[key])
        merged_texts[key] = merged

    configs_list = sorted(set(k.split('__')[0] for k in merged_texts))
    seeds_list = sorted(set(k.split('__')[1] for k in merged_texts))
    prompts_list = sorted(set(k.split('__')[2] for k in merged_texts))

    # Sample sizes to test
    sample_sizes = [5, 8, 10, 20, 30, 50]
    focus_metrics = ['distinct_2', 'self_bleu', 'epd', 'vendi', 'jaccard', 'ttr', 'usr', 'entropy']

    np.random.seed(42)
    ablation_results = {}

    for n_sample in sample_sizes:
        print(f"\n  Testing n={n_sample}...")
        # Check if we have enough texts
        valid_groups = 0
        insufficient = 0
        for key in merged_texts:
            if len(merged_texts[key]) >= n_sample:
                valid_groups += 1
            else:
                insufficient += 1

        if insufficient > 0:
            print(f"    {insufficient} groups have < {n_sample} texts (skipping those)")

        per_prompt_taus = {f'{m1}_vs_{m2}': [] for m1, m2 in itertools.combinations(focus_metrics, 2)}

        for prompt_id in prompts_list:
            for seed_key in seeds_list:
                vectors = {m: [] for m in focus_metrics}
                valid = True
                for config in configs_list:
                    key = f"{config}__{seed_key}__{prompt_id}"
                    if key not in merged_texts or len(merged_texts[key]) < n_sample:
                        valid = False
                        break
                    texts = merged_texts[key]
                    indices = np.random.choice(len(texts), n_sample, replace=False)
                    subset = [texts[i] for i in indices]
                    metrics = compute_all_metrics(subset)
                    for m in focus_metrics:
                        vectors[m].append(metrics[m])

                if not valid:
                    continue

                for m1, m2 in itertools.combinations(focus_metrics, 2):
                    tau = kendall_tau_pair(vectors[m1], vectors[m2])
                    per_prompt_taus[f'{m1}_vs_{m2}'].append(tau)

        ablation_results[str(n_sample)] = {}
        for pair_key, vals in per_prompt_taus.items():
            if vals:
                ablation_results[str(n_sample)][pair_key] = {
                    'mean_tau': round(float(np.mean(vals)), 4),
                    'std_tau': round(float(np.std(vals)), 4),
                    'n_replicates': len(vals),
                }

        # Print key results
        for pair in ['distinct_2_vs_epd', 'epd_vs_vendi', 'distinct_2_vs_self_bleu', 'distinct_2_vs_ttr']:
            s = ablation_results[str(n_sample)].get(pair, {})
            if s:
                print(f"    {pair:>30s}: tau = {s['mean_tau']:+.4f} ± {s['std_tau']:.4f} (n_rep={s['n_replicates']})")

    # Save results
    results_path = os.path.join(output_dir, 'extended_epd_ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'description': 'Extended EPD sample-size ablation (n=5,8,10,20,30,50)',
            'methodology': 'Per-prompt Kendall tau, averaged over prompt-seed replicates',
            'sample_sizes': sample_sizes,
            'metrics': focus_metrics,
            'results': ablation_results,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: EPD CONVERGENCE ANALYSIS")
    print("=" * 60)
    print(f"{'n':>5s}  {'D2-EPD':>12s}  {'D2-SB':>12s}  {'EPD-VS':>12s}  {'D2-TTR':>12s}")
    print("-" * 60)
    for n_s in sample_sizes:
        r = ablation_results.get(str(n_s), {})
        d2_epd = r.get('distinct_2_vs_epd', {}).get('mean_tau', float('nan'))
        d2_sb = r.get('distinct_2_vs_self_bleu', {}).get('mean_tau', float('nan'))
        epd_vs = r.get('epd_vs_vendi', {}).get('mean_tau', float('nan'))
        d2_ttr = r.get('distinct_2_vs_ttr', {}).get('mean_tau', float('nan'))
        print(f"{n_s:>5d}  {d2_epd:>+12.4f}  {d2_sb:>+12.4f}  {epd_vs:>+12.4f}  {d2_ttr:>+12.4f}")


if __name__ == '__main__':
    main()
