#!/usr/bin/env python3
"""
Comprehensive experiments for the DivFlow paper rewrite.

Experiments:
1. Metric Taxonomy on real LLM outputs (gpt-4.1-nano) - 5 domains × 3 prompts × 6 decoding configs × 3 seeds
2. DPP/MMR reranking of LLM outputs vs baselines
3. SVD-inspired diversity reranking evaluation
4. Quality-Diversity Pareto frontier
5. Tool comparison: DivFlow vs naive approaches
6. Downstream task: diverse generation improves coverage of correct answers
"""

import json
import os
import sys
import time
import hashlib
import math
import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import openai

client = openai.OpenAI()

RESULTS_DIR = Path(__file__).parent / "paper_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# Utility: Text Diversity Metrics (self-contained)
# ============================================================

def distinct_n(texts, n=2):
    """Distinct-n: fraction of unique n-grams across all texts."""
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        all_ngrams.extend(zip(*[tokens[i:] for i in range(n)]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)

def self_bleu_corpus(texts, n=4):
    """Approximate Self-BLEU: average pairwise n-gram overlap."""
    if len(texts) < 2:
        return 1.0
    
    def get_ngrams(text, n):
        tokens = text.lower().split()
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    
    scores = []
    for i, ref in enumerate(texts):
        ref_ngrams = get_ngrams(ref, n)
        if not ref_ngrams:
            continue
        for j, hyp in enumerate(texts):
            if i == j:
                continue
            hyp_ngrams = get_ngrams(hyp, n)
            if not hyp_ngrams:
                scores.append(0.0)
                continue
            overlap = sum((ref_ngrams & hyp_ngrams).values())
            total = sum(hyp_ngrams.values())
            scores.append(overlap / total if total > 0 else 0.0)
    return np.mean(scores) if scores else 0.0

def ngram_entropy(texts, n=3):
    """Entropy of n-gram distribution across texts."""
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        all_ngrams.extend(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    if not all_ngrams:
        return 0.0
    counts = Counter(all_ngrams)
    total = sum(counts.values())
    probs = np.array([c/total for c in counts.values()])
    return -np.sum(probs * np.log2(probs + 1e-12))

def type_token_ratio(texts):
    """Average type-token ratio across texts."""
    ratios = []
    for text in texts:
        tokens = text.lower().split()
        if tokens:
            ratios.append(len(set(tokens)) / len(tokens))
    return np.mean(ratios) if ratios else 0.0

def unique_sentence_ratio(texts):
    """Fraction of unique texts (character 4-gram based)."""
    def char_ngrams(text, n=4):
        return set(text[i:i+n] for i in range(len(text)-n+1))
    
    sigs = [frozenset(char_ngrams(t)) for t in texts]
    # Count truly unique by Jaccard distance
    unique = len(set(tuple(sorted(s)) for s in sigs))
    return unique / len(texts) if texts else 0.0

def compression_ratio_diversity(texts):
    """Diversity via compression: higher ratio = more diverse."""
    import zlib
    individual = sum(len(zlib.compress(t.encode())) for t in texts)
    combined = len(zlib.compress('\n'.join(texts).encode()))
    return combined / individual if individual > 0 else 1.0

def jaccard_diversity(texts):
    """Average pairwise Jaccard distance of word sets."""
    if len(texts) < 2:
        return 0.0
    word_sets = [set(t.lower().split()) for t in texts]
    dists = []
    for i in range(len(word_sets)):
        for j in range(i+1, len(word_sets)):
            union = word_sets[i] | word_sets[j]
            inter = word_sets[i] & word_sets[j]
            dists.append(1 - len(inter)/len(union) if union else 0)
    return np.mean(dists)

def embedding_pairwise_distance(texts, embeddings=None):
    """Mean pairwise cosine distance of embeddings."""
    if embeddings is None:
        return None
    if len(embeddings) < 2:
        return 0.0
    from scipy.spatial.distance import pdist
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embeddings / norms
    dists = pdist(normed, metric='cosine')
    return float(np.mean(dists))

def vendi_score(embeddings):
    """Vendi Score: exp(entropy of eigenvalues of normalized kernel matrix)."""
    if embeddings is None or len(embeddings) < 2:
        return 1.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embeddings / norms
    K = normed @ normed.T
    K = (K + K.T) / 2  # ensure symmetric
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues = eigenvalues / eigenvalues.sum()
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    return float(np.exp(entropy))

def compute_all_metrics(texts, embeddings=None):
    """Compute all diversity metrics for a set of texts."""
    metrics = {
        'distinct_1': distinct_n(texts, 1),
        'distinct_2': distinct_n(texts, 2),
        'distinct_3': distinct_n(texts, 3),
        'self_bleu': self_bleu_corpus(texts, 4),
        'ngram_entropy': ngram_entropy(texts, 3),
        'ttr': type_token_ratio(texts),
        'usr': unique_sentence_ratio(texts),
        'jaccard': jaccard_diversity(texts),
        'crd': compression_ratio_diversity(texts),
    }
    if embeddings is not None:
        metrics['epd'] = embedding_pairwise_distance(texts, embeddings)
        metrics['vendi'] = vendi_score(embeddings)
    return metrics

# ============================================================
# Embedding helper
# ============================================================

def get_embeddings(texts, model="text-embedding-3-small"):
    """Get embeddings from OpenAI API."""
    # batch in groups of 100
    all_embs = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = client.embeddings.create(input=batch, model=model)
        all_embs.extend([e.embedding for e in resp.data])
    return np.array(all_embs)

# ============================================================
# LLM Generation
# ============================================================

PROMPTS_BY_DOMAIN = {
    'creative_writing': [
        "Write a short story opening about a character who discovers something unexpected.",
        "Describe a scene where two strangers meet in an unusual location.",
        "Write the opening paragraph of a mystery novel set in a small town.",
    ],
    'code_explanation': [
        "Explain how a hash table works and when to use one.",
        "Describe the difference between depth-first and breadth-first search.",
        "Explain what a closure is in programming and give an example use case.",
    ],
    'science': [
        "Explain why the sky appears blue during the day.",
        "Describe how vaccines work to protect against diseases.",
        "Explain the concept of entropy in thermodynamics.",
    ],
    'business': [
        "Describe strategies for a startup to acquire its first 1000 customers.",
        "Explain the key factors that make a pitch deck compelling to investors.",
        "Describe how to build a strong company culture in a remote team.",
    ],
    'reasoning': [
        "A farmer has 17 sheep. All but 9 die. How many are left? Explain step by step.",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets? Explain.",
        "Three friends split a dinner bill of $30, each paying $10. The waiter returns $5. They each take $1 back and tip $2. Where did the missing dollar go? Explain.",
    ],
}

DECODING_CONFIGS = [
    {'temperature': 0.3, 'top_p': 1.0, 'label': 'temp_0.3'},
    {'temperature': 0.7, 'top_p': 1.0, 'label': 'temp_0.7'},
    {'temperature': 1.0, 'top_p': 1.0, 'label': 'temp_1.0'},
    {'temperature': 1.5, 'top_p': 1.0, 'label': 'temp_1.5'},
    {'temperature': 1.0, 'top_p': 0.5, 'label': 'nucleus_0.5'},
    {'temperature': 1.0, 'top_p': 0.9, 'label': 'nucleus_0.9'},
]

def generate_texts(prompt, n=10, temperature=1.0, top_p=1.0, max_tokens=100, model="gpt-4.1-nano"):
    """Generate n texts from the LLM."""
    texts = []
    for _ in range(n):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=1,
            )
            texts.append(resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"  Error generating: {e}")
            texts.append("")
    return [t for t in texts if t]

# ============================================================
# Experiment 1: Metric Taxonomy on Real LLM Outputs
# ============================================================

def run_metric_taxonomy_experiment():
    """Run the metric taxonomy experiment across 5 domains × 3 prompts × 6 configs × 3 seeds."""
    print("=" * 60)
    print("EXPERIMENT 1: Metric Taxonomy on Real LLM Outputs")
    print("=" * 60)
    
    all_results = {
        'metadata': {
            'model': 'gpt-4.1-nano',
            'domains': list(PROMPTS_BY_DOMAIN.keys()),
            'n_configs': len(DECODING_CONFIGS),
            'n_sequences_per_config': 10,
            'seeds': [42, 123, 456],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'per_domain': {},
        'aggregate': {},
    }
    
    all_texts_data = []
    
    for domain, prompts in PROMPTS_BY_DOMAIN.items():
        print(f"\n--- Domain: {domain} ---")
        domain_results = {'configs': [], 'tau_matrices': []}
        
        for seed_idx, seed in enumerate([42, 123, 456]):
            print(f"  Seed {seed}")
            np.random.seed(seed)
            config_metrics = []
            
            for prompt in prompts:
                print(f"    Prompt: {prompt[:50]}...")
                for config in DECODING_CONFIGS:
                    print(f"      Config: {config['label']}", end="... ")
                    texts = generate_texts(
                        prompt, n=10,
                        temperature=config['temperature'],
                        top_p=config['top_p'],
                        max_tokens=100,
                    )
                    if len(texts) < 3:
                        print("too few texts, skipping")
                        continue
                    
                    # Get embeddings
                    embeddings = get_embeddings(texts)
                    
                    # Compute metrics
                    metrics = compute_all_metrics(texts, embeddings)
                    metrics['config'] = config['label']
                    metrics['prompt'] = prompt[:50]
                    metrics['domain'] = domain
                    metrics['seed'] = seed
                    config_metrics.append(metrics)
                    
                    all_texts_data.append({
                        'domain': domain,
                        'prompt': prompt[:50],
                        'config': config['label'],
                        'seed': seed,
                        'texts': texts,
                        'n_texts': len(texts),
                    })
                    print(f"D2={metrics['distinct_2']:.3f}, SB={metrics['self_bleu']:.3f}, EPD={metrics['epd']:.3f}")
            
            if len(config_metrics) >= 3:
                # Compute tau matrix for this seed
                metric_names = ['distinct_2', 'self_bleu', 'ngram_entropy', 'ttr', 'usr', 
                               'jaccard', 'crd', 'epd', 'vendi']
                values = {m: [cm[m] for cm in config_metrics] for m in metric_names}
                
                tau_matrix = np.zeros((len(metric_names), len(metric_names)))
                for i, m1 in enumerate(metric_names):
                    for j, m2 in enumerate(metric_names):
                        if i == j:
                            tau_matrix[i][j] = 1.0
                        else:
                            tau, _ = stats.kendalltau(values[m1], values[m2])
                            tau_matrix[i][j] = tau if not np.isnan(tau) else 0.0
                
                domain_results['tau_matrices'].append({
                    'seed': seed,
                    'n_configs': len(config_metrics),
                    'tau_matrix': tau_matrix.tolist(),
                    'metric_names': metric_names,
                })
            domain_results['configs'] = config_metrics
        
        # Average tau across seeds
        if domain_results['tau_matrices']:
            avg_tau = np.mean([np.array(t['tau_matrix']) for t in domain_results['tau_matrices']], axis=0)
            std_tau = np.std([np.array(t['tau_matrix']) for t in domain_results['tau_matrices']], axis=0)
            domain_results['mean_tau_matrix'] = avg_tau.tolist()
            domain_results['std_tau_matrix'] = std_tau.tolist()
            domain_results['metric_names'] = domain_results['tau_matrices'][0]['metric_names']
        
        all_results['per_domain'][domain] = domain_results
    
    # Aggregate across all domains
    all_tau_matrices = []
    for domain, dr in all_results['per_domain'].items():
        for tm in dr.get('tau_matrices', []):
            all_tau_matrices.append(np.array(tm['tau_matrix']))
    
    if all_tau_matrices:
        agg_mean = np.mean(all_tau_matrices, axis=0)
        agg_std = np.std(all_tau_matrices, axis=0)
        metric_names = all_results['per_domain'][list(all_results['per_domain'].keys())[0]]['tau_matrices'][0]['metric_names']
        all_results['aggregate'] = {
            'mean_tau_matrix': agg_mean.tolist(),
            'std_tau_matrix': agg_std.tolist(),
            'metric_names': metric_names,
            'n_observations': len(all_tau_matrices),
        }
        
        # Identify clusters at |tau| > 0.7
        n = len(metric_names)
        visited = [False] * n
        clusters = []
        for i in range(n):
            if visited[i]:
                continue
            cluster = [metric_names[i]]
            visited[i] = True
            for j in range(i+1, n):
                if not visited[j] and abs(agg_mean[i][j]) > 0.7:
                    cluster.append(metric_names[j])
                    visited[j] = True
            clusters.append(cluster)
        all_results['aggregate']['clusters_at_0.7'] = clusters
    
    # Save results
    with open(RESULTS_DIR / 'experiment1_metric_taxonomy.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save texts
    with open(RESULTS_DIR / 'experiment1_texts.json', 'w') as f:
        json.dump(all_texts_data, f, indent=2)
    
    print(f"\nSaved {len(all_texts_data)} text groups to experiment1_metric_taxonomy.json")
    return all_results

# ============================================================
# Experiment 2: DPP/MMR Reranking of LLM Outputs  
# ============================================================

def dpp_rerank(texts, embeddings, k):
    """Rerank texts using DPP-based selection for diversity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embeddings / norms
    K = normed @ normed.T
    
    # Greedy DPP MAP
    selected = []
    remaining = list(range(len(texts)))
    
    for _ in range(min(k, len(texts))):
        if not remaining:
            break
        if not selected:
            # Pick item with highest self-similarity (diagonal)
            best = max(remaining, key=lambda i: K[i, i])
        else:
            # Pick item maximizing log-det gain (Schur complement)
            best_score = -float('inf')
            best = remaining[0]
            for i in remaining:
                S_idx = selected + [i]
                submat = K[np.ix_(S_idx, S_idx)]
                sign, logdet = np.linalg.slogdet(submat + 1e-6 * np.eye(len(S_idx)))
                score = logdet if sign > 0 else -float('inf')
                if score > best_score:
                    best_score = score
                    best = i
        selected.append(best)
        remaining.remove(best)
    
    return selected

def mmr_rerank(texts, embeddings, query_embedding, k, lambda_param=0.5):
    """MMR reranking: balance relevance and diversity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embeddings / norms
    
    q_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)
    relevance = normed @ q_norm
    sim_matrix = normed @ normed.T
    
    selected = []
    remaining = list(range(len(texts)))
    
    for _ in range(min(k, len(texts))):
        if not remaining:
            break
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            best_score = -float('inf')
            best = remaining[0]
            for i in remaining:
                max_sim = max(sim_matrix[i, j] for j in selected)
                score = lambda_param * relevance[i] - (1 - lambda_param) * max_sim
                if score > best_score:
                    best_score = score
                    best = i
        selected.append(best)
        remaining.remove(best)
    
    return selected

def run_reranking_experiment():
    """Compare DPP, MMR, random, and top-k reranking on LLM outputs."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: DPP/MMR Reranking of LLM Outputs")
    print("=" * 60)
    
    results = {
        'metadata': {
            'model': 'gpt-4.1-nano',
            'n_candidates': 20,
            'k_select': 5,
            'n_prompts': 10,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'per_prompt': [],
        'aggregate': {},
    }
    
    prompts = [
        "What are effective ways to learn a new language?",
        "Describe different approaches to solving climate change.",
        "What makes a good leader?",
        "How can cities become more sustainable?",
        "What are the pros and cons of remote work?",
        "Explain different theories about the origin of the universe.",
        "What strategies help manage stress effectively?",
        "How might artificial intelligence change education?",
        "What are different perspectives on universal basic income?",
        "Describe approaches to reducing food waste.",
    ]
    
    method_metrics = defaultdict(lambda: defaultdict(list))
    
    for pi, prompt in enumerate(prompts):
        print(f"\n  Prompt {pi+1}/{len(prompts)}: {prompt[:50]}...")
        
        # Generate 20 candidate texts
        texts = generate_texts(prompt, n=20, temperature=1.0, max_tokens=120)
        if len(texts) < 10:
            print("    Too few texts, skipping")
            continue
        
        # Get embeddings
        all_embs = get_embeddings(texts)
        query_emb = get_embeddings([prompt])[0]
        
        k = 5
        prompt_results = {'prompt': prompt, 'methods': {}}
        
        # Random selection
        for trial in range(3):
            rand_idx = np.random.choice(len(texts), k, replace=False).tolist()
            rand_texts = [texts[i] for i in rand_idx]
            rand_embs = all_embs[rand_idx]
            m = compute_all_metrics(rand_texts, rand_embs)
            for mk, mv in m.items():
                method_metrics['random'][mk].append(mv)
        
        # Top-k by relevance (cosine to query)
        q_norm = query_emb / max(np.linalg.norm(query_emb), 1e-8)
        emb_norms = all_embs / np.maximum(np.linalg.norm(all_embs, axis=1, keepdims=True), 1e-8)
        relevances = emb_norms @ q_norm
        topk_idx = np.argsort(relevances)[-k:][::-1].tolist()
        topk_texts = [texts[i] for i in topk_idx]
        topk_embs = all_embs[topk_idx]
        m = compute_all_metrics(topk_texts, topk_embs)
        for mk, mv in m.items():
            method_metrics['topk'][mk].append(mv)
        prompt_results['methods']['topk'] = m
        
        # DPP reranking
        dpp_idx = dpp_rerank(texts, all_embs, k)
        dpp_texts = [texts[i] for i in dpp_idx]
        dpp_embs = all_embs[dpp_idx]
        m = compute_all_metrics(dpp_texts, dpp_embs)
        for mk, mv in m.items():
            method_metrics['dpp'][mk].append(mv)
        prompt_results['methods']['dpp'] = m
        
        # MMR reranking (lambda=0.5)
        mmr_idx = mmr_rerank(texts, all_embs, query_emb, k, lambda_param=0.5)
        mmr_texts = [texts[i] for i in mmr_idx]
        mmr_embs = all_embs[mmr_idx]
        m = compute_all_metrics(mmr_texts, mmr_embs)
        for mk, mv in m.items():
            method_metrics['mmr_0.5'][mk].append(mv)
        prompt_results['methods']['mmr_0.5'] = m
        
        # MMR (lambda=0.3, more diversity)
        mmr_idx2 = mmr_rerank(texts, all_embs, query_emb, k, lambda_param=0.3)
        mmr_texts2 = [texts[i] for i in mmr_idx2]
        mmr_embs2 = all_embs[mmr_idx2]
        m = compute_all_metrics(mmr_texts2, mmr_embs2)
        for mk, mv in m.items():
            method_metrics['mmr_0.3'][mk].append(mv)
        prompt_results['methods']['mmr_0.3'] = m
        
        results['per_prompt'].append(prompt_results)
        print(f"    DPP D2={prompt_results['methods']['dpp']['distinct_2']:.3f} "
              f"MMR D2={prompt_results['methods']['mmr_0.5']['distinct_2']:.3f} "
              f"TopK D2={prompt_results['methods']['topk']['distinct_2']:.3f}")
    
    # Aggregate
    agg = {}
    for method, metrics in method_metrics.items():
        agg[method] = {}
        for mk, vals in metrics.items():
            if isinstance(vals[0], (int, float)):
                agg[method][mk] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'n': len(vals),
                }
    results['aggregate'] = agg
    
    with open(RESULTS_DIR / 'experiment2_reranking.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n  Aggregate Results:")
    for method in ['random', 'topk', 'dpp', 'mmr_0.5', 'mmr_0.3']:
        if method in agg:
            d2 = agg[method].get('distinct_2', {})
            epd = agg[method].get('epd', {})
            sb = agg[method].get('self_bleu', {})
            print(f"    {method:12s}: D2={d2.get('mean',0):.3f}±{d2.get('std',0):.3f}  "
                  f"EPD={epd.get('mean',0):.3f}±{epd.get('std',0):.3f}  "
                  f"SB={sb.get('mean',0):.3f}±{sb.get('std',0):.3f}")
    
    return results

# ============================================================
# Experiment 3: Quality-Diversity Pareto Analysis
# ============================================================

def run_pareto_experiment():
    """Map the quality-diversity Pareto frontier across decoding configs."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Quality-Diversity Pareto Frontier")
    print("=" * 60)
    
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "What are the key principles of effective software design?",
        "Describe how photosynthesis works.",
        "What strategies help build resilient organizations?",
        "Explain the concept of opportunity cost with examples.",
    ]
    
    configs = [
        {'temperature': 0.1, 'top_p': 1.0},
        {'temperature': 0.3, 'top_p': 1.0},
        {'temperature': 0.5, 'top_p': 1.0},
        {'temperature': 0.7, 'top_p': 1.0},
        {'temperature': 0.9, 'top_p': 1.0},
        {'temperature': 1.0, 'top_p': 1.0},
        {'temperature': 1.2, 'top_p': 1.0},
        {'temperature': 1.5, 'top_p': 1.0},
        {'temperature': 1.0, 'top_p': 0.3},
        {'temperature': 1.0, 'top_p': 0.5},
        {'temperature': 1.0, 'top_p': 0.7},
        {'temperature': 1.0, 'top_p': 0.9},
        {'temperature': 1.0, 'top_p': 0.95},
    ]
    
    results = {
        'metadata': {'model': 'gpt-4.1-nano', 'n_prompts': len(prompts), 'n_configs': len(configs)},
        'points': [],
        'per_prompt': {},
    }
    
    # Quality proxy: average text length, vocabulary sophistication, coherence via self-consistency
    for pi, prompt in enumerate(prompts):
        print(f"\n  Prompt {pi+1}: {prompt[:50]}...")
        prompt_points = []
        
        for config in configs:
            label = f"t={config['temperature']},p={config['top_p']}"
            texts = generate_texts(prompt, n=10, temperature=config['temperature'], 
                                 top_p=config['top_p'], max_tokens=100)
            if len(texts) < 3:
                continue
            
            embeddings = get_embeddings(texts)
            metrics = compute_all_metrics(texts, embeddings)
            
            # Quality proxy: average embedding similarity to prompt (relevance)
            prompt_emb = get_embeddings([prompt])[0]
            p_norm = prompt_emb / max(np.linalg.norm(prompt_emb), 1e-8)
            emb_norms = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
            relevances = emb_norms @ p_norm
            quality = float(np.mean(relevances))
            
            point = {
                'config': label,
                'temperature': config['temperature'],
                'top_p': config['top_p'],
                'diversity_d2': metrics['distinct_2'],
                'diversity_epd': metrics['epd'],
                'diversity_jaccard': metrics['jaccard'],
                'quality_relevance': quality,
                'self_bleu': metrics['self_bleu'],
            }
            prompt_points.append(point)
            results['points'].append(point)
            print(f"    {label}: D2={metrics['distinct_2']:.3f} Q={quality:.3f}")
        
        results['per_prompt'][prompt[:50]] = prompt_points
    
    # Compute Pareto frontier (diversity_d2 vs quality)
    points = [(p['diversity_d2'], p['quality_relevance'], p['config']) for p in results['points']]
    pareto = []
    for d, q, c in points:
        dominated = False
        for d2, q2, c2 in points:
            if d2 >= d and q2 >= q and (d2 > d or q2 > q):
                dominated = True
                break
        if not dominated:
            pareto.append({'diversity': d, 'quality': q, 'config': c})
    results['pareto_frontier'] = pareto
    
    with open(RESULTS_DIR / 'experiment3_pareto.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Pareto frontier: {len(pareto)} points")
    return results

# ============================================================
# Experiment 4: Downstream Task - Diverse Answers Improve Coverage
# ============================================================

def run_downstream_experiment():
    """Show diverse generation covers more correct answer aspects."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Downstream Coverage via Diverse Generation")
    print("=" * 60)
    
    # Questions with multiple valid aspects/answers
    questions = [
        {
            'prompt': "What are the main causes of the French Revolution?",
            'aspects': ['economic inequality', 'tax', 'Enlightenment', 'monarchy', 'food shortage', 'debt', 'estates'],
        },
        {
            'prompt': "What factors contribute to climate change?",
            'aspects': ['greenhouse gas', 'CO2', 'deforestation', 'fossil fuel', 'methane', 'industrial', 'agriculture'],
        },
        {
            'prompt': "What are the benefits of regular exercise?",
            'aspects': ['heart', 'mental health', 'weight', 'sleep', 'energy', 'bone', 'immune', 'mood'],
        },
        {
            'prompt': "What programming languages should a beginner learn and why?",
            'aspects': ['Python', 'JavaScript', 'simple', 'web', 'data', 'community', 'career', 'syntax'],
        },
        {
            'prompt': "What are the advantages and disadvantages of nuclear energy?",
            'aspects': ['clean', 'waste', 'efficient', 'danger', 'expensive', 'reliable', 'carbon', 'radiation'],
        },
    ]
    
    results = {'questions': [], 'aggregate': {}}
    
    method_coverages = defaultdict(list)
    
    for qi, q in enumerate(questions):
        print(f"\n  Q{qi+1}: {q['prompt'][:50]}...")
        
        # Low diversity (temp=0.3)
        low_div = generate_texts(q['prompt'], n=10, temperature=0.3, max_tokens=150)
        # Medium diversity (temp=0.7)
        med_div = generate_texts(q['prompt'], n=10, temperature=0.7, max_tokens=150)
        # High diversity (temp=1.2)
        high_div = generate_texts(q['prompt'], n=10, temperature=1.2, max_tokens=150)
        
        # DPP reranked from high pool
        pool = generate_texts(q['prompt'], n=20, temperature=1.0, max_tokens=150)
        if len(pool) >= 5:
            pool_embs = get_embeddings(pool)
            dpp_idx = dpp_rerank(pool, pool_embs, 10)
            dpp_texts = [pool[i] for i in dpp_idx]
        else:
            dpp_texts = pool
        
        def count_coverage(texts, aspects):
            covered = set()
            for t in texts:
                t_lower = t.lower()
                for a in aspects:
                    if a.lower() in t_lower:
                        covered.add(a)
            return len(covered) / len(aspects), list(covered)
        
        q_result = {'prompt': q['prompt'], 'aspects': q['aspects']}
        for label, texts in [('low_div', low_div), ('med_div', med_div), 
                            ('high_div', high_div), ('dpp_rerank', dpp_texts)]:
            cov, covered = count_coverage(texts, q['aspects'])
            q_result[label] = {
                'coverage': cov,
                'covered_aspects': covered,
                'n_texts': len(texts),
            }
            method_coverages[label].append(cov)
            print(f"    {label:12s}: coverage={cov:.2f} ({len(covered)}/{len(q['aspects'])} aspects)")
        
        results['questions'].append(q_result)
    
    # Aggregate
    for method, covs in method_coverages.items():
        results['aggregate'][method] = {
            'mean_coverage': float(np.mean(covs)),
            'std_coverage': float(np.std(covs)),
            'n': len(covs),
        }
    
    with open(RESULTS_DIR / 'experiment4_downstream.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n  Aggregate coverage:")
    for method in ['low_div', 'med_div', 'high_div', 'dpp_rerank']:
        agg = results['aggregate'][method]
        print(f"    {method:12s}: {agg['mean_coverage']:.3f} ± {agg['std_coverage']:.3f}")
    
    return results

# ============================================================
# Experiment 5: Tool Comparison (DPP selection time and quality)
# ============================================================

def run_tool_comparison():
    """Compare DivFlow's DPP against naive random selection at scale."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Tool Scalability Comparison")
    print("=" * 60)
    
    results = {'scaling': [], 'quality_comparison': []}
    
    for n in [50, 100, 200, 500, 1000]:
        print(f"\n  n={n}:")
        d = 50
        k = 10
        np.random.seed(42)
        X = np.random.randn(n, d)
        
        # Kernel computation
        from scipy.spatial.distance import cdist
        dists = cdist(X, X, 'sqeuclidean')
        gamma = 1.0 / d
        K = np.exp(-gamma * dists)
        
        # DPP greedy
        t0 = time.time()
        for trial in range(10):
            selected = []
            remaining = list(range(n))
            for _ in range(k):
                if not selected:
                    best = max(remaining, key=lambda i: K[i, i])
                else:
                    best_score = -float('inf')
                    best = remaining[0]
                    for i in remaining:
                        S_idx = selected + [i]
                        submat = K[np.ix_(S_idx, S_idx)]
                        sign, logdet = np.linalg.slogdet(submat + 1e-8 * np.eye(len(S_idx)))
                        if sign > 0 and logdet > best_score:
                            best_score = logdet
                            best = i
                    selected.append(best)
                    remaining.remove(best)
        dpp_time = (time.time() - t0) / 10
        
        # Random selection
        t0 = time.time()
        for trial in range(10):
            sel = np.random.choice(n, k, replace=False)
        random_time = (time.time() - t0) / 10
        
        # Spread comparison (final trial)
        dpp_spread = np.mean(cdist(X[selected], X[selected], 'euclidean')[np.triu_indices(k, k=1)])
        rand_sel = np.random.choice(n, k, replace=False)
        rand_spread = np.mean(cdist(X[rand_sel], X[rand_sel], 'euclidean')[np.triu_indices(k, k=1)])
        
        results['scaling'].append({
            'n': n, 'd': d, 'k': k,
            'dpp_time_ms': dpp_time * 1000,
            'random_time_ms': random_time * 1000,
            'dpp_spread': float(dpp_spread),
            'random_spread': float(rand_spread),
            'spread_improvement': float((dpp_spread - rand_spread) / rand_spread * 100),
        })
        print(f"    DPP: {dpp_time*1000:.1f}ms, spread={dpp_spread:.3f} | "
              f"Random: {random_time*1000:.3f}ms, spread={rand_spread:.3f} | "
              f"Improvement: {(dpp_spread-rand_spread)/rand_spread*100:.1f}%")
    
    with open(RESULTS_DIR / 'experiment5_tool_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

# ============================================================
# Experiment 6: EPD Independence Analysis (Model-Dependent)
# ============================================================

def run_epd_analysis():
    """Detailed EPD analysis: show model-dependent behavior."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: EPD Independence Analysis")
    print("=" * 60)
    
    prompts = [
        "Tell me about the history of computing.",
        "Describe the water cycle in nature.",
        "What are effective study techniques?",
        "Explain how machine learning works.",
        "Describe different types of renewable energy.",
    ]
    
    results = {'per_prompt': [], 'aggregate': {}}
    
    all_d2_epd_taus = []
    all_d2_sb_taus = []
    all_d2_jaccard_taus = []
    
    for pi, prompt in enumerate(prompts):
        print(f"\n  Prompt {pi+1}: {prompt[:50]}...")
        
        config_d2 = []
        config_epd = []
        config_sb = []
        config_jaccard = []
        
        for config in DECODING_CONFIGS:
            texts = generate_texts(prompt, n=10, temperature=config['temperature'],
                                 top_p=config['top_p'], max_tokens=80)
            if len(texts) < 3:
                continue
            embeddings = get_embeddings(texts)
            metrics = compute_all_metrics(texts, embeddings)
            config_d2.append(metrics['distinct_2'])
            config_epd.append(metrics['epd'])
            config_sb.append(metrics['self_bleu'])
            config_jaccard.append(metrics['jaccard'])
        
        if len(config_d2) >= 4:
            tau_d2_epd, _ = stats.kendalltau(config_d2, config_epd)
            tau_d2_sb, _ = stats.kendalltau(config_d2, config_sb)
            tau_d2_jac, _ = stats.kendalltau(config_d2, config_jaccard)
            
            all_d2_epd_taus.append(tau_d2_epd)
            all_d2_sb_taus.append(tau_d2_sb)
            all_d2_jaccard_taus.append(tau_d2_jac)
            
            results['per_prompt'].append({
                'prompt': prompt,
                'tau_d2_epd': float(tau_d2_epd),
                'tau_d2_sb': float(tau_d2_sb),
                'tau_d2_jaccard': float(tau_d2_jac),
                'n_configs': len(config_d2),
            })
            print(f"    D2-EPD τ={tau_d2_epd:.3f}  D2-SB τ={tau_d2_sb:.3f}  D2-Jac τ={tau_d2_jac:.3f}")
    
    if all_d2_epd_taus:
        results['aggregate'] = {
            'd2_epd': {'mean': float(np.mean(all_d2_epd_taus)), 'std': float(np.std(all_d2_epd_taus))},
            'd2_sb': {'mean': float(np.mean(all_d2_sb_taus)), 'std': float(np.std(all_d2_sb_taus))},
            'd2_jaccard': {'mean': float(np.mean(all_d2_jaccard_taus)), 'std': float(np.std(all_d2_jaccard_taus))},
        }
    
    with open(RESULTS_DIR / 'experiment6_epd_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("DivFlow Paper Experiments")
    print("=" * 60)
    
    t0 = time.time()
    
    # Run all experiments
    r1 = run_metric_taxonomy_experiment()
    r2 = run_reranking_experiment()
    r3 = run_pareto_experiment()
    r4 = run_downstream_experiment()
    r5 = run_tool_comparison()
    r6 = run_epd_analysis()
    
    elapsed = time.time() - t0
    print(f"\n\nAll experiments completed in {elapsed:.1f}s")
    print(f"Results saved to {RESULTS_DIR}")
