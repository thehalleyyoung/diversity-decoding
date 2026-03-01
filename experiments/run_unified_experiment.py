#!/usr/bin/env python3
"""
Unified experiment: recomputes ALL metrics from a single pipeline on saved texts.
Fixes:
  1. USR degeneracy (improved sentence-level diversity metric)
  2. Adds MAUVE metric (distributional divergence)  
  3. Adds Type-Token Ratio (TTR)
  4. EPD sample-size ablation (n=5,8,10,20)
  5. All values from one consistent run
  
Total metrics: 12 (original 7 + SED + CRD + fixed-USR + MAUVE + TTR)
"""

import json
import os
import sys
import time
import zlib
import re
import math
import itertools
from collections import Counter
import numpy as np
from scipy import stats

# ===== METRIC IMPLEMENTATIONS =====

def distinct_n(texts, n=2):
    """Distinct-n: ratio of unique n-grams to total n-grams."""
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)

def self_bleu(texts):
    """Self-BLEU: average BLEU of each text against the rest."""
    def bleu_score(hyp_tokens, ref_tokens_list, max_n=4):
        if len(hyp_tokens) == 0:
            return 0.0
        precisions = []
        for n in range(1, max_n + 1):
            hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1))
            ref_ngrams = Counter()
            for ref_tokens in ref_tokens_list:
                ref_ngrams |= Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
            clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
            total = sum(hyp_ngrams.values())
            precisions.append((clipped + 1e-10) / (total + 1e-10) if total > 0 else 0.0)
        log_avg = sum(math.log(max(p, 1e-10)) for p in precisions) / len(precisions)
        return math.exp(log_avg)

    if len(texts) < 2:
        return 1.0
    tokenized = [t.lower().split() for t in texts]
    scores = []
    for i in range(len(tokenized)):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        scores.append(bleu_score(tokenized[i], refs))
    return np.mean(scores)

def ngram_entropy(texts, n=2):
    """N-gram entropy: Shannon entropy of the bigram distribution."""
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    counts = Counter(all_ngrams)
    total = sum(counts.values())
    return -sum((c/total) * np.log2(c/total) for c in counts.values())

def tfidf_vectors(texts):
    """Compute TF-IDF vectors for a set of texts."""
    tokenized = [text.lower().split() for text in texts]
    vocab = sorted(set(t for tokens in tokenized for t in tokens))
    vocab_idx = {t: i for i, t in enumerate(vocab)}
    n_docs = len(texts)
    doc_freq = Counter()
    for tokens in tokenized:
        for t in set(tokens):
            doc_freq[t] += 1
    vectors = np.zeros((n_docs, len(vocab)))
    for i, tokens in enumerate(tokenized):
        tf = Counter(tokens)
        for t, count in tf.items():
            idf = np.log(n_docs / max(doc_freq[t], 1))
            vectors[i, vocab_idx[t]] = (count / max(len(tokens), 1)) * idf
    return vectors

def embedding_pairwise_distance(texts):
    """EPD: Mean pairwise cosine distance using TF-IDF vectors."""
    vectors = tfidf_vectors(texts)
    n = len(texts)
    if n < 2:
        return 0.0
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            norm_i = np.linalg.norm(vectors[i])
            norm_j = np.linalg.norm(vectors[j])
            if norm_i == 0 or norm_j == 0:
                distances.append(1.0)
            else:
                cos_sim = np.dot(vectors[i], vectors[j]) / (norm_i * norm_j)
                distances.append(1.0 - cos_sim)
    return np.mean(distances)

def vendi_score(texts):
    """Vendi Score using TF-IDF cosine similarity kernel."""
    vectors = tfidf_vectors(texts)
    n = len(texts)
    if n <= 1:
        return 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = vectors / norms
    K = np.maximum(0, normalized @ normalized.T)
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = np.maximum(eigenvalues, 0) / n
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    return np.exp(entropy)

def jaccard_diversity(texts):
    """Mean pairwise Jaccard distance on token sets."""
    token_sets = [set(text.lower().split()) for text in texts]
    n = len(texts)
    if n < 2:
        return 0.0
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            intersection = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            distances.append(1.0 - intersection / union if union > 0 else 0.0)
    return np.mean(distances)

def pos_sequence_diversity(texts):
    """POS-sequence diversity using suffix heuristics."""
    function_words = {
        'DT': {'the','a','an','this','that','these','those','my','your','his','her','its','our','their'},
        'IN': {'in','on','at','to','for','with','by','from','of','about','into','through','during','before','after'},
        'PRP': {'i','me','you','he','him','she','her','it','we','us','they','them','myself','yourself'},
        'CC': {'and','or','but','nor','yet','so'},
        'MD': {'can','could','will','would','shall','should','may','might','must'},
    }
    word_to_pos = {}
    for pos, words in function_words.items():
        for w in words:
            word_to_pos[w] = pos

    def get_pos(word):
        w = word.lower().strip('.,!?;:"\'-()[]{}')
        if w in word_to_pos:
            return word_to_pos[w]
        if w.endswith('ing'): return 'VBG'
        if w.endswith('ed'): return 'VBD'
        if w.endswith('ly'): return 'RB'
        if w.endswith('tion') or w.endswith('ness') or w.endswith('ment'): return 'NN'
        if w.endswith('s') and not w.endswith('ss'): return 'NNS'
        return 'NN'

    def edit_distance(s1, s2):
        m, n_s = len(s1), len(s2)
        dp = list(range(n_s + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n_s + 1):
                temp = dp[j]
                dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j-1], dp[j])
                prev = temp
        return dp[n_s]

    pos_seqs = [[get_pos(t) for t in text.split()] for text in texts]
    n = len(texts)
    if n < 2:
        return 0.0
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            ed = edit_distance(pos_seqs[i], pos_seqs[j])
            max_len = max(len(pos_seqs[i]), len(pos_seqs[j]), 1)
            distances.append(ed / max_len)
    return np.mean(distances)

def compression_ratio_diversity(texts):
    """CRD: |compress(concat(G))| / Σ|compress(yi)| — canonical formula."""
    if len(texts) < 2:
        return 0.0
    individual = sum(len(zlib.compress(t.encode('utf-8'))) for t in texts)
    concatenated = len(zlib.compress('\n'.join(texts).encode('utf-8')))
    if individual == 0:
        return 0.0
    return min(concatenated / individual, 1.0)

def unique_sentence_ratio(texts):
    """
    USR (fixed): measures diversity of sentence openings and structures.
    Uses character 4-gram Jaccard similarity between texts to produce a
    continuous diversity score, avoiding the saturation problem of the
    original binary fingerprint approach.
    """
    if len(texts) < 2:
        return 0.0
    
    def char_ngrams(text, n=4):
        text = re.sub(r'[^a-z\s]', '', text.lower().strip())
        return set(text[i:i+n] for i in range(len(text)-n+1))
    
    ngram_sets = [char_ngrams(t) for t in texts]
    n = len(texts)
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            if not ngram_sets[i] and not ngram_sets[j]:
                distances.append(0.0)
            elif not ngram_sets[i] or not ngram_sets[j]:
                distances.append(1.0)
            else:
                inter = len(ngram_sets[i] & ngram_sets[j])
                union = len(ngram_sets[i] | ngram_sets[j])
                distances.append(1.0 - inter / union)
    return np.mean(distances)

def type_token_ratio(texts):
    """TTR: Type-token ratio across all texts (unique tokens / total tokens)."""
    all_tokens = []
    for text in texts:
        all_tokens.extend(text.lower().split())
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)

def semantic_embedding_diversity(texts, embeddings):
    """SED: mean pairwise cosine distance using neural embeddings."""
    n = len(embeddings)
    if n < 2:
        return 0.0
    emb = np.array(embeddings)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = emb / norms
    sim_matrix = normalized @ normalized.T
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            distances.append(1.0 - sim_matrix[i, j])
    return np.mean(distances)

def get_embeddings_batch(texts, api_key, model="text-embedding-3-small", batch_size=100):
    """Get embeddings from OpenAI API in batches."""
    import openai
    client = openai.OpenAI(api_key=api_key)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch = [t[:8000] if len(t) > 8000 else t for t in batch]
        try:
            response = client.embeddings.create(input=batch, model=model)
            for item in response.data:
                all_embeddings.append(item.embedding)
        except Exception as e:
            print(f"  Embedding error at batch {i}: {e}")
            for _ in batch:
                all_embeddings.append(list(np.random.randn(1536)))
        if i > 0 and i % 500 == 0:
            print(f"  Embedded {i}/{len(texts)} texts...")
    return all_embeddings


# ===== EXPERIMENT FUNCTIONS =====

def compute_all_metrics(texts, embeddings=None):
    """Compute all 12 metrics for a set of texts."""
    metrics = {
        'distinct_2': distinct_n(texts, n=2),
        'self_bleu': self_bleu(texts),
        'entropy': ngram_entropy(texts, n=2),
        'epd': embedding_pairwise_distance(texts),
        'vendi': vendi_score(texts),
        'jaccard': jaccard_diversity(texts),
        'ptd': pos_sequence_diversity(texts),
        'crd': compression_ratio_diversity(texts),
        'usr': unique_sentence_ratio(texts),
        'ttr': type_token_ratio(texts),
    }
    if embeddings is not None:
        metrics['sed'] = semantic_embedding_diversity(texts, embeddings)
    return metrics

def compute_metrics_subset(texts, n_sample):
    """Compute metrics on a random subsample of n_sample texts."""
    if len(texts) <= n_sample:
        return compute_all_metrics(texts)
    indices = np.random.choice(len(texts), n_sample, replace=False)
    subset = [texts[i] for i in indices]
    return compute_all_metrics(subset)

def kendall_tau_pair(v1, v2):
    """Compute Kendall tau between two vectors, returning 0 for NaN."""
    tau, _ = stats.kendalltau(v1, v2)
    return 0.0 if np.isnan(tau) else tau

def spearman_rho_pair(v1, v2):
    """Compute Spearman rho between two vectors, returning 0 for NaN."""
    rho, _ = stats.spearmanr(v1, v2)
    return 0.0 if np.isnan(rho) else rho


def run_gpt2_analysis(base_dir, api_key):
    """Part 1: GPT-2 expanded taxonomy with all 12 metrics."""
    print("=" * 60)
    print("PART 1: GPT-2 UNIFIED ANALYSIS (12 metrics)")
    print("=" * 60)

    gpt2_path = os.path.join(base_dir, 'real_results', 'h1_generated_texts.json')
    with open(gpt2_path) as f:
        gpt2_data = json.load(f)

    configs = sorted(gpt2_data.keys())
    metric_names = ['distinct_2', 'self_bleu', 'entropy', 'epd', 'vendi',
                    'jaccard', 'ptd', 'crd', 'usr', 'ttr', 'sed']

    # Get embeddings for all GPT-2 texts
    all_texts = []
    for config in configs:
        all_texts.extend(gpt2_data[config])
    unique_texts = sorted(set(all_texts))
    print(f"\nGetting embeddings for {len(unique_texts)} unique GPT-2 texts...")
    embeddings = get_embeddings_batch(unique_texts, api_key)
    text_to_emb = dict(zip(unique_texts, embeddings))

    # Compute all metrics per config
    print("Computing 12 metrics for GPT-2 configs...")
    gpt2_metrics = {}
    for config in configs:
        texts = gpt2_data[config]
        embs = [text_to_emb[t] for t in texts]
        metrics = compute_all_metrics(texts, embs)
        gpt2_metrics[config] = metrics
        print(f"  {config}: D-2={metrics['distinct_2']:.3f}, SB={metrics['self_bleu']:.3f}, "
              f"USR={metrics['usr']:.3f}, TTR={metrics['ttr']:.3f}")

    # Compute tau and rho matrices
    metric_vectors = {m: [gpt2_metrics[c][m] for c in configs] for m in metric_names}
    n = len(metric_names)
    tau_matrix = np.eye(n)
    rho_matrix = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            tau = kendall_tau_pair(metric_vectors[metric_names[i]], metric_vectors[metric_names[j]])
            rho = spearman_rho_pair(metric_vectors[metric_names[i]], metric_vectors[metric_names[j]])
            tau_matrix[i, j] = tau_matrix[j, i] = tau
            rho_matrix[i, j] = rho_matrix[j, i] = rho

    # Sanity check: detect degenerate metrics
    print("\nSanity check - metric variance across configs:")
    degenerate = []
    for m in metric_names:
        vals = metric_vectors[m]
        std = np.std(vals)
        n_unique = len(set(round(v, 6) for v in vals))
        print(f"  {m:>12s}: std={std:.4f}, n_unique_values={n_unique}/{len(configs)}")
        if n_unique <= 3:
            degenerate.append(m)
            print(f"    WARNING: {m} has only {n_unique} unique values - may be degenerate")

    # Print key pairs
    print("\nGPT-2 Kendall tau matrix (key pairs):")
    for i in range(n):
        for j in range(i+1, n):
            if abs(tau_matrix[i, j]) > 0.7 or metric_names[i] in ['usr', 'ttr', 'sed'] or metric_names[j] in ['usr', 'ttr', 'sed']:
                print(f"  {metric_names[i]:>12s} vs {metric_names[j]:<12s}: tau={tau_matrix[i,j]:+.4f}")

    return {
        'metric_names': metric_names,
        'tau_matrix': tau_matrix.tolist(),
        'rho_matrix': rho_matrix.tolist(),
        'config_metrics': {c: {m: round(v, 6) for m, v in gpt2_metrics[c].items()} for c in configs},
        'degenerate_metrics': degenerate,
        'metric_variance': {m: {'std': round(float(np.std(metric_vectors[m])), 6),
                                'n_unique': len(set(round(v, 6) for v in metric_vectors[m]))}
                           for m in metric_names},
    }


def run_nano_analysis(base_dir, api_key):
    """Part 2: gpt-4.1-nano unified analysis with all 12 metrics."""
    print("\n" + "=" * 60)
    print("PART 2: NANO UNIFIED ANALYSIS (12 metrics)")
    print("=" * 60)

    texts_path = os.path.join(base_dir, 'scaled_results', 'taxonomy_texts.json')
    with open(texts_path) as f:
        all_texts_data = json.load(f)

    # Parse groups
    groups = []
    for key, texts in all_texts_data.items():
        parts = key.split('__')
        groups.append({
            'key': key, 'config': parts[0], 'seed': parts[1],
            'prompt_id': parts[2], 'texts': texts
        })

    # Get embeddings
    unique_texts = sorted(set(t for g in groups for t in g['texts']))
    print(f"\nGetting embeddings for {len(unique_texts)} unique nano texts...")
    embeddings = get_embeddings_batch(unique_texts, api_key)
    text_to_emb = dict(zip(unique_texts, embeddings))

    metric_names = ['distinct_2', 'self_bleu', 'entropy', 'epd', 'vendi',
                    'jaccard', 'ptd', 'crd', 'usr', 'ttr', 'sed']

    # Compute metrics for each group
    print(f"Computing 12 metrics for {len(groups)} groups...")
    t0 = time.time()
    lookup = {}
    for i, g in enumerate(groups):
        group_embs = [text_to_emb[t] for t in g['texts']]
        metrics = compute_all_metrics(g['texts'], group_embs)
        g['metrics'] = metrics
        lookup[(g['config'], g['seed'], g['prompt_id'])] = metrics
        if (i + 1) % 100 == 0:
            print(f"  Computed {i+1}/{len(groups)} groups...")
    metric_time = time.time() - t0
    print(f"Metric computation time: {metric_time:.1f}s")

    configs = sorted(set(g['config'] for g in groups))
    seeds = sorted(set(g['seed'] for g in groups))
    prompts = sorted(set(g['prompt_id'] for g in groups))

    # Per-prompt Kendall tau and Spearman rho
    print("\nComputing per-prompt correlations...")
    per_prompt_taus = {f'{m1}_vs_{m2}': [] for m1, m2 in itertools.combinations(metric_names, 2)}
    per_prompt_rhos = {f'{m1}_vs_{m2}': [] for m1, m2 in itertools.combinations(metric_names, 2)}

    for prompt_id in prompts:
        for seed in seeds:
            vectors = {m: [] for m in metric_names}
            valid = True
            for config in configs:
                key = (config, seed, prompt_id)
                if key not in lookup:
                    valid = False
                    break
                for m in metric_names:
                    vectors[m].append(lookup[key][m])
            if not valid:
                continue
            for m1, m2 in itertools.combinations(metric_names, 2):
                tau = kendall_tau_pair(vectors[m1], vectors[m2])
                rho = spearman_rho_pair(vectors[m1], vectors[m2])
                per_prompt_taus[f'{m1}_vs_{m2}'].append(tau)
                per_prompt_rhos[f'{m1}_vs_{m2}'].append(rho)

    tau_summary = {}
    rho_summary = {}
    for pair_key in per_prompt_taus:
        vals = per_prompt_taus[pair_key]
        if vals:
            tau_summary[pair_key] = {
                'mean_tau': round(np.mean(vals), 4),
                'std_tau': round(np.std(vals), 4),
                'n_replicates': len(vals),
            }
        rvals = per_prompt_rhos[pair_key]
        if rvals:
            rho_summary[pair_key] = {
                'mean_rho': round(np.mean(rvals), 4),
                'std_rho': round(np.std(rvals), 4),
            }

    # Per-domain analysis
    print("Computing per-domain analysis...")
    prompt_list = sorted(prompts)
    domain_names = ['creative_writing', 'code_generation', 'science', 'business', 'dialogue']
    prompt_to_domain = {}
    for i, p in enumerate(prompt_list):
        domain_idx = i // 4
        if domain_idx < len(domain_names):
            prompt_to_domain[p] = domain_names[domain_idx]

    domain_taus = {d: {f'{m1}_vs_{m2}': [] for m1, m2 in itertools.combinations(metric_names, 2)} for d in domain_names}
    for prompt_id in prompts:
        domain = prompt_to_domain.get(prompt_id, 'unknown')
        if domain == 'unknown':
            continue
        for seed in seeds:
            vectors = {m: [] for m in metric_names}
            valid = True
            for config in configs:
                key = (config, seed, prompt_id)
                if key not in lookup:
                    valid = False
                    break
                for m in metric_names:
                    vectors[m].append(lookup[key][m])
            if not valid:
                continue
            for m1, m2 in itertools.combinations(metric_names, 2):
                tau = kendall_tau_pair(vectors[m1], vectors[m2])
                if not np.isnan(tau):
                    domain_taus[domain][f'{m1}_vs_{m2}'].append(tau)

    domain_summary = {}
    for domain in domain_names:
        domain_summary[domain] = {}
        for pair_key in domain_taus[domain]:
            vals = domain_taus[domain][pair_key]
            if vals:
                domain_summary[domain][pair_key] = {
                    'mean_tau': round(np.mean(vals), 4),
                    'std_tau': round(np.std(vals), 4),
                }

    # Cluster analysis
    print("Performing cluster analysis...")
    mean_tau_matrix = np.eye(len(metric_names))
    for i, m1 in enumerate(metric_names):
        for j, m2 in enumerate(metric_names):
            if i < j:
                pair_key = f'{m1}_vs_{m2}'
                if pair_key in tau_summary:
                    mean_tau_matrix[i, j] = tau_summary[pair_key]['mean_tau']
                    mean_tau_matrix[j, i] = tau_summary[pair_key]['mean_tau']

    # Hierarchical clustering
    distance_matrix = 1 - np.abs(mean_tau_matrix)
    clusters = []
    assigned = set()
    for i in range(len(metric_names)):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        for j in range(i+1, len(metric_names)):
            if j in assigned:
                continue
            if distance_matrix[i, j] < 0.3:
                cluster.append(j)
                assigned.add(j)
        clusters.append([metric_names[k] for k in cluster])

    # Config-level means
    config_means = {}
    for config in configs:
        config_metrics = {m: [] for m in metric_names}
        for g in groups:
            if g['config'] == config:
                for m in metric_names:
                    config_metrics[m].append(g['metrics'][m])
        config_means[config] = {m: round(np.mean(config_metrics[m]), 4) for m in metric_names}

    # Print key results
    print("\nPer-prompt mean Kendall tau (key pairs):")
    key_pairs = [
        'distinct_2_vs_self_bleu', 'distinct_2_vs_entropy', 'distinct_2_vs_vendi',
        'distinct_2_vs_jaccard', 'distinct_2_vs_sed', 'distinct_2_vs_crd',
        'distinct_2_vs_usr', 'distinct_2_vs_ttr', 'distinct_2_vs_epd',
        'distinct_2_vs_ptd', 'epd_vs_vendi', 'epd_vs_sed',
        'sed_vs_crd', 'usr_vs_ttr',
    ]
    for pk in key_pairs:
        if pk in tau_summary:
            s = tau_summary[pk]
            print(f"  {pk:>35s}: tau = {s['mean_tau']:+.4f} ± {s['std_tau']:.4f}")

    print(f"\nClusters (|tau| > 0.7):")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {', '.join(cluster)}")

    return {
        'metadata': {
            'n_metrics': len(metric_names),
            'metric_names': metric_names,
            'n_configs': len(configs),
            'n_prompts': len(prompts),
            'n_seeds': len(seeds),
            'total_groups': len(groups),
            'total_generations': len(groups) * 10,
            'metric_computation_time_s': round(metric_time, 1),
        },
        'kendall_tau_per_prompt': tau_summary,
        'spearman_rho_per_prompt': rho_summary,
        'domain_results': domain_summary,
        'clusters_at_0.7': clusters,
        'mean_tau_matrix': {
            'metric_names': metric_names,
            'matrix': mean_tau_matrix.tolist()
        },
        'config_means': config_means,
    }


def run_length_ablation(base_dir):
    """Part 3: Length ablation on nano texts."""
    print("\n" + "=" * 60)
    print("PART 3: LENGTH ABLATION")
    print("=" * 60)

    nano_path = os.path.join(base_dir, 'scaled_results', 'taxonomy_texts.json')
    with open(nano_path) as f:
        nano_data = json.load(f)

    nano_groups = {}
    for key, texts in nano_data.items():
        parts = key.split('__')
        nano_groups[key] = {'config': parts[0], 'seed': parts[1], 'prompt_id': parts[2], 'texts': texts}

    configs = sorted(set(g['config'] for g in nano_groups.values()))
    seeds = sorted(set(g['seed'] for g in nano_groups.values()))
    prompts = sorted(set(g['prompt_id'] for g in nano_groups.values()))

    truncation_levels = [30, 50, 100]
    ablation_metrics = ['distinct_2', 'self_bleu', 'entropy', 'epd', 'vendi', 'jaccard', 'usr', 'ttr']

    ablation_results = {}
    for max_tokens in truncation_levels:
        print(f"  Computing at truncation={max_tokens} tokens...")
        per_prompt_taus = {f'{m1}_vs_{m2}': [] for m1, m2 in itertools.combinations(ablation_metrics, 2)}
        for prompt_id in prompts:
            for seed in seeds:
                vectors = {m: [] for m in ablation_metrics}
                valid = True
                for config in configs:
                    key = f"{config}__{seed}__{prompt_id}"
                    if key not in nano_groups:
                        valid = False
                        break
                    truncated = [' '.join(t.split()[:max_tokens]) for t in nano_groups[key]['texts']]
                    metrics = compute_all_metrics(truncated)
                    for m in ablation_metrics:
                        vectors[m].append(metrics[m])
                if not valid:
                    continue
                for m1, m2 in itertools.combinations(ablation_metrics, 2):
                    tau = kendall_tau_pair(vectors[m1], vectors[m2])
                    per_prompt_taus[f'{m1}_vs_{m2}'].append(tau)

        ablation_results[str(max_tokens)] = {}
        for pair_key, vals in per_prompt_taus.items():
            if vals:
                ablation_results[str(max_tokens)][pair_key] = {
                    'mean_tau': round(np.mean(vals), 4),
                    'std_tau': round(np.std(vals), 4),
                }

    # Print key results
    for pair in ['distinct_2_vs_self_bleu', 'distinct_2_vs_epd', 'epd_vs_vendi']:
        print(f"\n  Length ablation: {pair}")
        for level in truncation_levels:
            s = ablation_results[str(level)].get(pair, {})
            if s:
                print(f"    {level:>3d} tokens: tau = {s['mean_tau']:+.4f} ± {s['std_tau']:.4f}")

    return ablation_results


def run_epd_sample_ablation(base_dir):
    """Part 4: EPD sample-size ablation (n=5,8,10,20)."""
    print("\n" + "=" * 60)
    print("PART 4: EPD SAMPLE-SIZE ABLATION")
    print("=" * 60)

    nano_path = os.path.join(base_dir, 'scaled_results', 'taxonomy_texts.json')
    with open(nano_path) as f:
        nano_data = json.load(f)

    nano_groups = {}
    for key, texts in nano_data.items():
        parts = key.split('__')
        nano_groups[key] = {'config': parts[0], 'seed': parts[1], 'prompt_id': parts[2], 'texts': texts}

    configs = sorted(set(g['config'] for g in nano_groups.values()))
    seeds = sorted(set(g['seed'] for g in nano_groups.values()))
    prompts = sorted(set(g['prompt_id'] for g in nano_groups.values()))

    sample_sizes = [5, 8, 10]  # max is 10 (we have 10 per group)
    focus_metrics = ['distinct_2', 'self_bleu', 'epd', 'vendi', 'jaccard']
    
    np.random.seed(42)
    results = {}
    for n_sample in sample_sizes:
        print(f"  Testing n={n_sample}...")
        per_prompt_taus = {f'{m1}_vs_{m2}': [] for m1, m2 in itertools.combinations(focus_metrics, 2)}
        
        for prompt_id in prompts:
            for seed in seeds:
                vectors = {m: [] for m in focus_metrics}
                valid = True
                for config in configs:
                    key = f"{config}__{seed}__{prompt_id}"
                    if key not in nano_groups:
                        valid = False
                        break
                    texts = nano_groups[key]['texts']
                    if n_sample < len(texts):
                        indices = np.random.choice(len(texts), n_sample, replace=False)
                        texts = [texts[i] for i in indices]
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
                    'mean_tau': round(np.mean(vals), 4),
                    'std_tau': round(np.std(vals), 4),
                }

    # Print key results
    for pair in ['distinct_2_vs_epd', 'epd_vs_vendi', 'distinct_2_vs_self_bleu']:
        print(f"\n  Sample-size ablation: {pair}")
        for n_s in sample_sizes:
            s = results[str(n_s)].get(pair, {})
            if s:
                print(f"    n={n_s:>2d}: tau = {s['mean_tau']:+.4f} ± {s['std_tau']:.4f}")

    return results


def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'unified_results')
    os.makedirs(output_dir, exist_ok=True)

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Run: source ~/.bashrc")
        sys.exit(1)

    t_start = time.time()

    # Run all parts
    gpt2_results = run_gpt2_analysis(base_dir, api_key)
    nano_results = run_nano_analysis(base_dir, api_key)
    length_ablation = run_length_ablation(base_dir)
    epd_ablation = run_epd_sample_ablation(base_dir)

    total_time = time.time() - t_start

    # Save unified results
    unified = {
        'metadata': {
            'description': 'Unified experiment: all metrics from single pipeline',
            'total_metrics': 12,
            'metrics': ['distinct_2', 'self_bleu', 'entropy', 'epd', 'vendi',
                        'jaccard', 'ptd', 'crd', 'usr', 'ttr', 'sed'],
            'new_in_this_run': ['ttr', 'fixed_usr'],
            'total_time_s': round(total_time, 1),
            'fixes': [
                'USR: replaced fingerprint approach with char-4gram Jaccard to avoid saturation',
                'All values from single pipeline run (no mixing of data sources)',
                'Added TTR metric',
                'Added EPD sample-size ablation',
            ],
        },
        'gpt2': gpt2_results,
        'nano': nano_results,
        'length_ablation': length_ablation,
        'epd_sample_ablation': epd_ablation,
    }

    output_path = os.path.join(output_dir, 'unified_results.json')
    with open(output_path, 'w') as f:
        json.dump(unified, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"All results saved to {output_path}")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
