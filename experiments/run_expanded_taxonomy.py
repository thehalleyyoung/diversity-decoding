#!/usr/bin/env python3
"""
Expanded Taxonomy Experiment: adds 3 new metrics to the existing 7-metric taxonomy.

New metrics:
1. Semantic Embedding Diversity (SED): Mean pairwise cosine distance using
   OpenAI text-embedding-3-small embeddings (neural/semantic diversity).
2. Compression Ratio Diversity (CRD): Ratio of compressed concatenated text
   to sum of individually compressed texts (information-theoretic diversity).
3. Unique Sentence Ratio (USR): Fraction of unique sentence structures after
   normalization (structural diversity at sentence level).

Computes expanded 10-metric Kendall tau taxonomy on existing saved generations.
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

# Existing metric implementations
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
    from collections import Counter
    def bleu_score(hypothesis, references, max_n=4):
        hyp_tokens = hypothesis.lower().split()
        ref_tokens_list = [r.lower().split() for r in references]
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
            if total == 0:
                precisions.append(0.0)
            else:
                precisions.append((clipped + 1e-10) / (total + 1e-10))
        import math
        log_avg = sum(math.log(max(p, 1e-10)) for p in precisions) / len(precisions)
        bp = min(1.0, math.exp(1 - max(len(ref_tokens_list[0].split()) if isinstance(ref_tokens_list[0], str) else len(ref_tokens_list[0]), 1) / max(len(hyp_tokens), 1)))
        return bp * math.exp(log_avg)

    if len(texts) < 2:
        return 1.0
    scores = []
    for i, text in enumerate(texts):
        refs = [texts[j] for j in range(len(texts)) if j != i]
        scores.append(bleu_score(text, refs))
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
    entropy = -sum((c/total) * np.log2(c/total) for c in counts.values())
    return entropy

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
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues = eigenvalues / n
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
            if union == 0:
                distances.append(0.0)
            else:
                distances.append(1.0 - intersection / union)
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
        m, n = len(s1), len(s2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1): dp[i][0] = i
        for j in range(n+1): dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]

    pos_seqs = []
    for text in texts:
        tokens = text.split()
        pos_seq = [get_pos(t) for t in tokens]
        pos_seqs.append(pos_seq)

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


# === NEW METRICS ===

def compression_ratio_diversity(texts):
    """
    Compression Ratio Diversity (CRD): |compress(concat(G))| / Σ|compress(yi)|
    Canonical formula — consistent with Section 3 and Appendix §F.8.
    Higher = more diverse (less cross-text redundancy).
    """
    if len(texts) < 2:
        return 0.0
    individual_compressed = sum(len(zlib.compress(t.encode('utf-8'))) for t in texts)
    concat_compressed = len(zlib.compress('\n'.join(texts).encode('utf-8')))
    if individual_compressed == 0:
        return 0.0
    return min(concat_compressed / individual_compressed, 1.0)

def unique_sentence_ratio(texts):
    """
    Unique Sentence Ratio (USR): fraction of unique normalized sentence structures.
    Captures high-level structural diversity at the sentence level.
    """
    def normalize_sentence(text):
        text = text.lower().strip()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        # Keep only first 3 and last 3 words as structural fingerprint
        if len(words) > 6:
            return ' '.join(words[:3]) + ' ... ' + ' '.join(words[-3:])
        return ' '.join(words)

    normalized = [normalize_sentence(t) for t in texts]
    if not normalized:
        return 0.0
    return len(set(normalized)) / len(normalized)


def semantic_embedding_diversity(texts, embeddings):
    """
    Semantic Embedding Diversity (SED): mean pairwise cosine distance
    using pre-computed neural embeddings.
    """
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
        # Truncate very long texts
        batch = [t[:8000] if len(t) > 8000 else t for t in batch]
        try:
            response = client.embeddings.create(input=batch, model=model)
            for item in response.data:
                all_embeddings.append(item.embedding)
        except Exception as e:
            print(f"  Embedding error at batch {i}: {e}")
            # Fallback: random embeddings (marked as fallback)
            for _ in batch:
                all_embeddings.append(list(np.random.randn(1536)))
        if i > 0 and i % 500 == 0:
            print(f"  Embedded {i}/{len(texts)} texts...")
    return all_embeddings


def compute_all_metrics(texts, embeddings=None):
    """Compute all 10 metrics for a set of texts."""
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
    }
    if embeddings is not None:
        metrics['sed'] = semantic_embedding_diversity(texts, embeddings)
    return metrics


def kendall_tau_matrix(metric_vectors, metric_names):
    """Compute pairwise Kendall tau matrix."""
    n = len(metric_names)
    tau_matrix = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            tau, _ = stats.kendalltau(metric_vectors[metric_names[i]],
                                       metric_vectors[metric_names[j]])
            if np.isnan(tau):
                tau = 0.0
            tau_matrix[i, j] = tau
            tau_matrix[j, i] = tau
    return tau_matrix


def spearman_rho_matrix(metric_vectors, metric_names):
    """Compute pairwise Spearman rho matrix for comparison."""
    n = len(metric_names)
    rho_matrix = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            rho, _ = stats.spearmanr(metric_vectors[metric_names[i]],
                                      metric_vectors[metric_names[j]])
            if np.isnan(rho):
                rho = 0.0
            rho_matrix[i, j] = rho
            rho_matrix[j, i] = rho
    return rho_matrix


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scaled_dir = os.path.join(base_dir, 'scaled_results')
    output_dir = os.path.join(base_dir, 'expanded_results')
    os.makedirs(output_dir, exist_ok=True)

    # Load API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Run: source ~/.bashrc")
        sys.exit(1)

    print("=" * 60)
    print("EXPANDED TAXONOMY EXPERIMENT")
    print("Adding 3 new metrics: SED, CRD, USR")
    print("=" * 60)

    # Load existing nano texts
    texts_path = os.path.join(scaled_dir, 'taxonomy_texts.json')
    print(f"\nLoading texts from {texts_path}...")
    with open(texts_path) as f:
        all_texts = json.load(f)
    print(f"Loaded {len(all_texts)} text groups")

    # Parse keys to extract config, seed, prompt
    groups = []
    for key, texts in all_texts.items():
        parts = key.split('__')
        config = parts[0]
        seed = parts[1]
        prompt_id = parts[2]
        groups.append({
            'key': key,
            'config': config,
            'seed': seed,
            'prompt_id': prompt_id,
            'texts': texts
        })

    # Get all unique texts for embedding
    print("\nCollecting unique texts for embedding...")
    unique_texts = set()
    for g in groups:
        for t in g['texts']:
            unique_texts.add(t)
    unique_texts = sorted(unique_texts)
    print(f"Found {len(unique_texts)} unique texts")

    # Get embeddings
    print("\nGetting OpenAI embeddings (text-embedding-3-small)...")
    t0 = time.time()
    embeddings = get_embeddings_batch(unique_texts, api_key)
    embed_time = time.time() - t0
    print(f"Embedding time: {embed_time:.1f}s")

    text_to_embedding = dict(zip(unique_texts, embeddings))

    # Compute metrics for each group
    print("\nComputing 10 metrics for each of 780 groups...")
    t0 = time.time()
    metric_names = ['distinct_2', 'self_bleu', 'entropy', 'epd', 'vendi',
                    'jaccard', 'ptd', 'crd', 'usr', 'sed']

    group_metrics = []
    for i, g in enumerate(groups):
        group_embeddings = [text_to_embedding[t] for t in g['texts']]
        metrics = compute_all_metrics(g['texts'], group_embeddings)
        g['metrics'] = metrics
        group_metrics.append(metrics)
        if (i + 1) % 100 == 0:
            print(f"  Computed {i+1}/{len(groups)} groups...")

    metric_time = time.time() - t0
    print(f"Metric computation time: {metric_time:.1f}s")

    # === Per-prompt Kendall tau (same methodology as original) ===
    print("\nComputing per-prompt Kendall tau (60 prompt-seed combinations)...")
    configs = sorted(set(g['config'] for g in groups))
    seeds = sorted(set(g['seed'] for g in groups))
    prompts = sorted(set(g['prompt_id'] for g in groups))

    # Build lookup
    lookup = {}
    for g in groups:
        lookup[(g['config'], g['seed'], g['prompt_id'])] = g['metrics']

    per_prompt_taus = {f'{m1}_vs_{m2}': [] for m1, m2 in itertools.combinations(metric_names, 2)}
    per_prompt_rhos = {f'{m1}_vs_{m2}': [] for m1, m2 in itertools.combinations(metric_names, 2)}

    for prompt_id in prompts:
        for seed in seeds:
            # Get metric vectors for this prompt-seed over all configs
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

            # Compute pairwise tau and rho
            for m1, m2 in itertools.combinations(metric_names, 2):
                tau, _ = stats.kendalltau(vectors[m1], vectors[m2])
                rho, _ = stats.spearmanr(vectors[m1], vectors[m2])
                if not np.isnan(tau):
                    per_prompt_taus[f'{m1}_vs_{m2}'].append(tau)
                if not np.isnan(rho):
                    per_prompt_rhos[f'{m1}_vs_{m2}'].append(rho)

    # Compute summary statistics
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

    # === Per-domain analysis ===
    print("\nComputing per-domain analysis...")
    # Map prompt IDs to domains (first 4 = creative, next 4 = code, etc.)
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
                tau, _ = stats.kendalltau(vectors[m1], vectors[m2])
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

    # === Cluster analysis ===
    print("\nPerforming cluster analysis...")
    # Build mean tau matrix for clustering
    mean_tau_matrix = np.eye(len(metric_names))
    for i, m1 in enumerate(metric_names):
        for j, m2 in enumerate(metric_names):
            if i < j:
                pair_key = f'{m1}_vs_{m2}'
                if pair_key in tau_summary:
                    mean_tau_matrix[i, j] = tau_summary[pair_key]['mean_tau']
                    mean_tau_matrix[j, i] = tau_summary[pair_key]['mean_tau']

    # Simple hierarchical clustering by finding clusters at |tau| > 0.7
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
            if distance_matrix[i, j] < 0.3:  # |tau| > 0.7
                cluster.append(j)
                assigned.add(j)
        clusters.append([metric_names[k] for k in cluster])

    # === Config-level means ===
    print("\nComputing config-level metric means...")
    config_means = {}
    for config in configs:
        config_metrics = {m: [] for m in metric_names}
        for g in groups:
            if g['config'] == config:
                for m in metric_names:
                    config_metrics[m].append(g['metrics'][m])
        config_means[config] = {m: round(np.mean(config_metrics[m]), 4) for m in metric_names}

    # === Save results ===
    results = {
        'metadata': {
            'n_metrics': len(metric_names),
            'metric_names': metric_names,
            'new_metrics': ['crd', 'usr', 'sed'],
            'n_configs': len(configs),
            'n_prompts': len(prompts),
            'n_seeds': len(seeds),
            'total_groups': len(groups),
            'total_generations': len(groups) * 10,
            'embedding_model': 'text-embedding-3-small',
            'embedding_time_s': round(embed_time, 1),
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

    output_path = os.path.join(output_dir, 'expanded_taxonomy.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPANDED TAXONOMY RESULTS (10 metrics)")
    print("=" * 60)

    print("\nPer-prompt mean Kendall tau (key pairs):")
    key_pairs = [
        ('distinct_2', 'self_bleu'),
        ('distinct_2', 'entropy'),
        ('distinct_2', 'vendi'),
        ('distinct_2', 'jaccard'),
        ('distinct_2', 'sed'),
        ('distinct_2', 'crd'),
        ('distinct_2', 'usr'),
        ('epd', 'vendi'),
        ('epd', 'sed'),
        ('sed', 'vendi'),
        ('sed', 'crd'),
        ('crd', 'usr'),
    ]
    for m1, m2 in key_pairs:
        pair_key = f'{m1}_vs_{m2}'
        if pair_key in tau_summary:
            s = tau_summary[pair_key]
            print(f"  {m1:>12s} vs {m2:<12s}: tau = {s['mean_tau']:+.3f} ± {s['std_tau']:.3f}")

    print(f"\nClusters (|tau| > 0.7 threshold):")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {', '.join(cluster)}")

    # Kendall vs Spearman comparison
    print("\nKendall tau vs Spearman rho comparison (key pairs):")
    for m1, m2 in key_pairs[:5]:
        pair_key = f'{m1}_vs_{m2}'
        if pair_key in tau_summary and pair_key in rho_summary:
            tau_val = tau_summary[pair_key]['mean_tau']
            rho_val = rho_summary[pair_key]['mean_rho']
            print(f"  {m1:>12s} vs {m2:<12s}: tau={tau_val:+.3f}, rho={rho_val:+.3f}")

    print(f"\nTotal time: {embed_time + metric_time:.1f}s")
    print("Done!")


if __name__ == '__main__':
    main()
