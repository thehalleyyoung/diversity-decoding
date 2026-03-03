#!/usr/bin/env python3
"""
diversity-taxonomy: Compute the diversity metric taxonomy on your own generation sets.

Usage:
    python3 diversity_taxonomy.py --input generations.json [--output results.json] [--metrics all]

Input format (JSON):
    {
        "config_name_1": ["text1", "text2", ...],
        "config_name_2": ["text1", "text2", ...],
        ...
    }

Each key is a decoding configuration name. Each value is a list of generated texts.
The tool computes pairwise Kendall tau across all metrics and identifies clusters.

Metrics available:
    distinct_2  - Distinct bigram ratio (lexical)
    self_bleu   - Self-BLEU (lexical, lower=more diverse)
    entropy     - Bigram entropy (lexical)
    epd         - Embedding pairwise distance (TF-IDF cosine)
    vendi       - Vendi Score (kernel-based effective diversity)
    jaccard     - Jaccard diversity (set-overlap)
    ptd         - POS-sequence diversity (structural)
    crd         - Compression ratio diversity (information-theoretic)
    usr         - Unique sentence ratio (character n-gram Jaccard distance)
    ttr         - Type-token ratio (lexical)

Example:
    python3 diversity_taxonomy.py --input my_generations.json --threshold 0.7
"""

import json
import sys
import os
import argparse
import itertools
from collections import Counter
import numpy as np
from scipy import stats
import zlib
import re


# === Metric implementations ===

def distinct_n(texts, n=2):
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)

def self_bleu(texts):
    import math
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

def embedding_pairwise_distance(texts):
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
            vectors[i, vocab_idx[t]] = (count / max(len(tokens), 1)) * np.log(n_docs / max(doc_freq[t], 1))
    if n_docs < 2:
        return 0.0
    distances = []
    for i in range(n_docs):
        for j in range(i+1, n_docs):
            ni, nj = np.linalg.norm(vectors[i]), np.linalg.norm(vectors[j])
            if ni == 0 or nj == 0:
                distances.append(1.0)
            else:
                distances.append(1.0 - np.dot(vectors[i], vectors[j]) / (ni * nj))
    return np.mean(distances)

def vendi_score(texts):
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
            vectors[i, vocab_idx[t]] = (count / max(len(tokens), 1)) * np.log(n_docs / max(doc_freq[t], 1))
    if n_docs <= 1:
        return 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = vectors / norms
    K = np.maximum(0, normalized @ normalized.T)
    eigenvalues = np.maximum(np.linalg.eigvalsh(K), 0) / n_docs
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return np.exp(-np.sum(eigenvalues * np.log(eigenvalues)))

def jaccard_diversity(texts):
    token_sets = [set(text.lower().split()) for text in texts]
    n = len(texts)
    if n < 2:
        return 0.0
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            inter = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            distances.append(1.0 - inter / union if union > 0 else 0.0)
    return np.mean(distances)

def pos_sequence_diversity(texts):
    func_words = {'DT': {'the','a','an','this','that','these','those'},
                  'IN': {'in','on','at','to','for','with','by','from','of','about'},
                  'PRP': {'i','me','you','he','him','she','her','it','we','they'},
                  'CC': {'and','or','but','nor','yet','so'},
                  'MD': {'can','could','will','would','shall','should','may','might','must'}}
    w2p = {}
    for pos, words in func_words.items():
        for w in words:
            w2p[w] = pos

    def get_pos(word):
        w = word.lower().strip('.,!?;:"\'-()[]{}')
        if w in w2p: return w2p[w]
        if w.endswith('ing'): return 'VBG'
        if w.endswith('ed'): return 'VBD'
        if w.endswith('ly'): return 'RB'
        if w.endswith(('tion','ness','ment')): return 'NN'
        if w.endswith('s') and not w.endswith('ss'): return 'NNS'
        return 'NN'

    def edit_dist(s1, s2):
        m, n = len(s1), len(s2)
        dp = list(range(n+1))
        for i in range(1, m+1):
            prev, dp[0] = dp[0], i
            for j in range(1, n+1):
                temp = dp[j]
                dp[j] = prev if s1[i-1]==s2[j-1] else 1+min(prev, dp[j-1], dp[j])
                prev = temp
        return dp[n]

    pos_seqs = [[get_pos(t) for t in text.split()] for text in texts]
    n = len(texts)
    if n < 2:
        return 0.0
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            ed = edit_dist(pos_seqs[i], pos_seqs[j])
            distances.append(ed / max(len(pos_seqs[i]), len(pos_seqs[j]), 1))
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
    """Character 4-gram Jaccard distance between all text pairs."""
    def char_ngrams(text, k=4):
        t = text.lower().strip()
        return set(t[i:i+k] for i in range(len(t) - k + 1))
    if len(texts) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(texts)):
        ci = char_ngrams(texts[i])
        for j in range(i+1, len(texts)):
            cj = char_ngrams(texts[j])
            union = len(ci | cj)
            if union > 0:
                total += 1.0 - len(ci & cj) / union
            else:
                total += 1.0
            count += 1
    return total / count if count > 0 else 0.0


def type_token_ratio(texts):
    """Ratio of unique word types to total word tokens."""
    all_tokens = []
    for text in texts:
        tokens = re.sub(r'[^a-z\s]', '', text.lower()).split()
        all_tokens.extend(tokens)
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)


METRICS = {
    'distinct_2': ('Distinct-2', '↑', distinct_n),
    'self_bleu': ('Self-BLEU', '↓', self_bleu),
    'entropy': ('Bigram Entropy', '↑', ngram_entropy),
    'epd': ('Embedding PD', '↑', embedding_pairwise_distance),
    'vendi': ('Vendi Score', '↑', vendi_score),
    'jaccard': ('Jaccard Div.', '↑', jaccard_diversity),
    'ptd': ('POS-Seq Div.', '↑', pos_sequence_diversity),
    'crd': ('Compression Div.', '↑', compression_ratio_diversity),
    'usr': ('Unique Sent. Ratio', '↑', unique_sentence_ratio),
    'ttr': ('Type-Token Ratio', '↑', type_token_ratio),
}


def main():
    parser = argparse.ArgumentParser(
        description='Compute diversity metric taxonomy on generation sets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', '-i', required=True, help='Input JSON/JSONL file with generations')
    parser.add_argument('--output', '-o', default=None, help='Output JSON file (default: stdout)')
    parser.add_argument('--metrics', '-m', default='all',
                        help='Comma-separated metric names or "all" (default: all)')
    parser.add_argument('--threshold', '-t', type=float, default=0.7,
                        help='|tau| threshold for cluster identification (default: 0.7)')
    parser.add_argument('--format', '-f', choices=['table', 'json'], default='table',
                        help='Output format (default: table)')
    parser.add_argument('--input-format', choices=['json', 'jsonl', 'csv', 'parquet'], default=None,
                        help='Input format (default: auto-detect from extension)')
    parser.add_argument('--text-field', default=None,
                        help='JSON field name for text (JSONL input only)')
    args = parser.parse_args()

    # Load data
    fmt = args.input_format
    if fmt is None:
        # Auto-detect from extension
        lower = args.input.lower()
        if lower.endswith('.jsonl'):
            fmt = 'jsonl'
        elif lower.endswith('.csv'):
            fmt = 'csv'
        elif lower.endswith('.parquet'):
            fmt = 'parquet'
        else:
            fmt = 'json'

    if fmt == 'jsonl':
        from src.io.jsonl_loader import load_texts_jsonl
        texts = load_texts_jsonl(args.input, text_field=args.text_field)
        data = {"default": texts}
    elif fmt == 'csv':
        from src.io.csv_loader import load_csv
        result = load_csv(args.input, text_column=args.text_field or 'text')
        data = result if isinstance(result, dict) else {"default": result}
    elif fmt == 'parquet':
        from src.io.csv_loader import load_parquet
        result = load_parquet(args.input, text_column=args.text_field or 'text')
        data = result if isinstance(result, dict) else {"default": result}
    else:
        with open(args.input) as f:
            raw = json.load(f)
        # Support HuggingFace JSON: list or {"data": [...]}
        if isinstance(raw, list) or (isinstance(raw, dict) and "data" in raw and not all(isinstance(v, list) for v in raw.values())):
            from src.io.jsonl_loader import load_texts_hf_json
            texts = load_texts_hf_json(args.input, text_field=args.text_field)
            data = {"default": texts}
        else:
            data = raw

    if not isinstance(data, dict):
        print("Error: Input must be a JSON object with config names as keys", file=sys.stderr)
        sys.exit(1)

    configs = sorted(data.keys())
    print(f"Loaded {len(configs)} configurations", file=sys.stderr)

    # Select metrics
    if args.metrics == 'all':
        selected = list(METRICS.keys())
    else:
        selected = [m.strip() for m in args.metrics.split(',')]
        for m in selected:
            if m not in METRICS:
                print(f"Error: Unknown metric '{m}'. Available: {', '.join(METRICS.keys())}", file=sys.stderr)
                sys.exit(1)

    # Compute metrics per config
    print(f"Computing {len(selected)} metrics...", file=sys.stderr)
    config_metrics = {}
    for config in configs:
        texts = data[config]
        metrics = {}
        for m in selected:
            name, direction, func = METRICS[m]
            metrics[m] = func(texts)
        config_metrics[config] = metrics

    # Compute Kendall tau matrix
    metric_vectors = {m: [config_metrics[c][m] for c in configs] for m in selected}
    n = len(selected)
    tau_matrix = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            tau, _ = stats.kendalltau(metric_vectors[selected[i]], metric_vectors[selected[j]])
            tau = 0.0 if np.isnan(tau) else tau
            tau_matrix[i, j] = tau
            tau_matrix[j, i] = tau

    # Identify clusters
    assigned = set()
    clusters = []
    for i in range(n):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        for j in range(i+1, n):
            if j in assigned:
                continue
            if abs(tau_matrix[i, j]) >= args.threshold:
                cluster.append(j)
                assigned.add(j)
        clusters.append([selected[k] for k in cluster])

    # Identify redundant and independent pairs
    redundant = [(selected[i], selected[j], tau_matrix[i,j])
                 for i in range(n) for j in range(i+1, n)
                 if abs(tau_matrix[i,j]) >= args.threshold]
    independent = [(selected[i], selected[j], tau_matrix[i,j])
                   for i in range(n) for j in range(i+1, n)
                   if abs(tau_matrix[i,j]) < 0.15]

    # Output
    if args.format == 'table':
        print(f"\n{'='*60}")
        print(f"DIVERSITY METRIC TAXONOMY ({len(configs)} configurations)")
        print(f"{'='*60}")

        # Metrics per config
        print(f"\nMetric values per configuration:")
        header = f"{'Config':>20s}" + "".join(f"  {METRICS[m][0]:>14s}" for m in selected)
        print(header)
        print("-" * len(header))
        for config in configs:
            row = f"{config:>20s}"
            for m in selected:
                row += f"  {config_metrics[config][m]:>14.4f}"
            print(row)

        # Tau matrix
        print(f"\nKendall tau correlation matrix:")
        abbrevs = [m[:6] for m in selected]
        header = f"{'':>12s}" + "".join(f"{a:>8s}" for a in abbrevs)
        print(header)
        for i, m in enumerate(selected):
            row = f"{m:>12s}" + "".join(f"{tau_matrix[i,j]:>8.3f}" for j in range(n))
            print(row)

        # Clusters
        print(f"\nClusters (|tau| > {args.threshold}):")
        for i, cluster in enumerate(clusters):
            names = [METRICS[m][0] for m in cluster]
            print(f"  Cluster {i+1}: {', '.join(names)}")

        # Recommendations
        print(f"\nRecommendations:")
        if redundant:
            print(f"  Redundant pairs ({len(redundant)}):")
            for m1, m2, tau in sorted(redundant, key=lambda x: -abs(x[2]))[:5]:
                print(f"    {METRICS[m1][0]} ↔ {METRICS[m2][0]}: tau={tau:+.3f} (report only one)")
        if independent:
            print(f"  Independent pairs ({len(independent)}):")
            for m1, m2, tau in independent:
                print(f"    {METRICS[m1][0]} ↔ {METRICS[m2][0]}: tau={tau:+.3f} (report both)")

        # Minimal suite
        print(f"\n  Minimal non-redundant suite: report one metric from each cluster:")
        for i, cluster in enumerate(clusters):
            recommended = cluster[0]
            print(f"    From cluster {i+1}: {METRICS[recommended][0]} ({recommended})")

    else:
        result = {
            'config_metrics': {c: {m: round(v, 6) for m, v in config_metrics[c].items()} for c in configs},
            'tau_matrix': {'metrics': selected, 'values': tau_matrix.tolist()},
            'clusters': clusters,
            'redundant_pairs': [{'m1': m1, 'm2': m2, 'tau': round(t, 4)} for m1, m2, t in redundant],
            'independent_pairs': [{'m1': m1, 'm2': m2, 'tau': round(t, 4)} for m1, m2, t in independent],
        }
        output = json.dumps(result, indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Results saved to {args.output}", file=sys.stderr)
        else:
            print(output)


if __name__ == '__main__':
    main()
