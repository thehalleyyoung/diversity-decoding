#!/usr/bin/env python3
"""Diversity Decoding Arena: Diversity Benchmark Experiments.

Benchmarks diversity metrics (Distinct-N, Self-BLEU, Vendi, FBD) on synthetic text,
measures metric correlations, compares selection algorithms, and tests streaming vs batch.
Outputs: diversity_benchmark_results.json
"""

import json
import os
import time
import numpy as np
from collections import Counter
from itertools import combinations

np.random.seed(42)

# ---------------------------------------------------------------------------
# Synthetic text generation
# ---------------------------------------------------------------------------

VOCAB = [f"word_{i}" for i in range(500)]

def generate_text(n_words, diversity_level=0.5, seed_offset=0):
    """Generate synthetic text with controlled diversity.
    
    diversity_level: 0.0 = very repetitive, 1.0 = maximally diverse
    """
    rng = np.random.RandomState(42 + seed_offset)
    vocab_size = max(5, int(len(VOCAB) * diversity_level))
    active_vocab = VOCAB[:vocab_size]
    # Zipf-like distribution with diversity controlling flatness
    alpha = 1.0 + (1.0 - diversity_level) * 2.0  # more repetitive → steeper
    probs = np.array([1.0 / (i + 1) ** alpha for i in range(vocab_size)])
    probs /= probs.sum()
    words = rng.choice(active_vocab, size=n_words, p=probs)
    return " ".join(words)


def generate_response_set(n_responses, n_words=30, diversity_level=0.5):
    """Generate a set of text responses."""
    return [generate_text(n_words, diversity_level, seed_offset=i)
            for i in range(n_responses)]


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def distinct_n(texts, n=2):
    """Distinct-N: fraction of unique n-grams across all texts."""
    all_ngrams = []
    total = 0
    for text in texts:
        words = text.split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)
        total += len(ngrams)
    if total == 0:
        return 0.0
    return len(set(all_ngrams)) / total


def self_bleu(texts, n=4):
    """Self-BLEU: average BLEU of each text against all others (lower = more diverse)."""
    def ngram_counts(text, n):
        words = text.split()
        return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

    def bleu_single(hypothesis, references, max_n=4):
        """Simplified BLEU score."""
        score = 0.0
        for nn in range(1, min(max_n, n) + 1):
            hyp_counts = ngram_counts(hypothesis, nn)
            ref_counts = Counter()
            for ref in references:
                ref_counts |= ngram_counts(ref, nn)
            clipped = sum(min(hyp_counts[ng], ref_counts[ng]) for ng in hyp_counts)
            total = sum(hyp_counts.values())
            if total > 0:
                score += np.log(max(clipped / total, 1e-10))
            else:
                score += np.log(1e-10)
        return np.exp(score / min(max_n, n))

    if len(texts) < 2:
        return 0.0
    scores = []
    for i, text in enumerate(texts):
        refs = [t for j, t in enumerate(texts) if j != i]
        scores.append(bleu_single(text, refs))
    return float(np.mean(scores))


def vendi_score(texts):
    """Vendi Score: exp(entropy of eigenvalues of similarity matrix)."""
    n = len(texts)
    # Compute similarity matrix via word overlap
    word_sets = [set(t.split()) for t in texts]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if len(word_sets[i]) == 0 and len(word_sets[j]) == 0:
                K[i, j] = 1.0
            else:
                K[i, j] = len(word_sets[i] & word_sets[j]) / max(len(word_sets[i] | word_sets[j]), 1)
    # Normalize
    K = K / n
    eigvals = np.linalg.eigvalsh(K)
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) == 0:
        return 1.0
    eigvals = eigvals / eigvals.sum()
    entropy = -np.sum(eigvals * np.log(eigvals + 1e-12))
    return float(np.exp(entropy))


def fbd_score(texts, dim=50):
    """Fréchet-Based Diversity: Fréchet distance of text embeddings from uniform."""
    # Embed texts as bag-of-words vectors projected to lower dim
    all_words = set()
    for t in texts:
        all_words.update(t.split())
    word_list = sorted(all_words)
    word2idx = {w: i for i, w in enumerate(word_list)}
    V = len(word_list)

    # Random projection for dimensionality reduction
    proj = np.random.randn(V, dim) / np.sqrt(dim) if V > dim else np.eye(V, dim)

    embeddings = np.zeros((len(texts), dim))
    for i, text in enumerate(texts):
        bow = np.zeros(V)
        for w in text.split():
            bow[word2idx[w]] += 1
        if bow.sum() > 0:
            bow /= bow.sum()
        embeddings[i] = bow @ proj[:V, :dim]

    # Compute FBD (Fréchet distance from zero-mean identity-cov reference)
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False) + np.eye(dim) * 1e-6

    # Fréchet distance: ||mu||^2 + trace(Sigma + I - 2*(Sigma)^{1/2})
    eigvals = np.linalg.eigvalsh(sigma)
    eigvals = np.maximum(eigvals, 0)
    sqrt_sigma_trace = np.sum(np.sqrt(eigvals))
    fbd = float(np.sum(mu ** 2) + np.trace(sigma) + dim - 2 * sqrt_sigma_trace)
    return fbd


# ---------------------------------------------------------------------------
# Selection algorithms
# ---------------------------------------------------------------------------

def compute_similarity_matrix(texts):
    """Word-overlap Jaccard similarity matrix."""
    n = len(texts)
    word_sets = [set(t.split()) for t in texts]
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            union = len(word_sets[i] | word_sets[j])
            S[i, j] = len(word_sets[i] & word_sets[j]) / max(union, 1)
    return S


def select_flow(texts, k, S=None):
    """Flow-based diverse selection (greedy max-min)."""
    if S is None:
        S = compute_similarity_matrix(texts)
    n = len(texts)
    selected = [np.random.randint(n)]
    for _ in range(k - 1):
        min_sims = np.array([min(S[i, s] for s in selected) for i in range(n)])
        min_sims[selected] = np.inf
        selected.append(int(np.argmin(min_sims)))
    return selected


def select_mmr(texts, k, lambda_param=0.5, S=None):
    """Maximal Marginal Relevance selection."""
    if S is None:
        S = compute_similarity_matrix(texts)
    n = len(texts)
    # Quality: word count as proxy
    quality = np.array([len(set(t.split())) for t in texts], dtype=float)
    quality /= quality.max() + 1e-10

    selected = [int(np.argmax(quality))]
    for _ in range(k - 1):
        best_score, best_idx = -np.inf, -1
        for i in range(n):
            if i in selected:
                continue
            max_sim = max(S[i, s] for s in selected)
            score = lambda_param * quality[i] - (1 - lambda_param) * max_sim
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
    return selected


def select_dpp(texts, k, S=None):
    """DPP-inspired greedy selection (determinantal point process)."""
    if S is None:
        S = compute_similarity_matrix(texts)
    n = len(texts)
    L = np.eye(n) + S * 0.5  # DPP kernel

    selected = []
    remaining = list(range(n))
    for _ in range(k):
        if not remaining:
            break
        if not selected:
            # Pick highest self-quality
            scores = [L[i, i] for i in remaining]
            best = remaining[int(np.argmax(scores))]
        else:
            best_score, best = -np.inf, remaining[0]
            for i in remaining:
                sub = selected + [i]
                det = np.linalg.det(L[np.ix_(sub, sub)])
                if det > best_score:
                    best_score, best = det, i
        selected.append(best)
        remaining.remove(best)
    return selected


def select_kmedoids(texts, k, S=None):
    """K-medoids diverse selection."""
    if S is None:
        S = compute_similarity_matrix(texts)
    n = len(texts)
    dist = 1 - S

    # Initialize with k-means++ style
    medoids = [np.random.randint(n)]
    for _ in range(k - 1):
        min_dists = np.array([min(dist[i, m] for m in medoids) for i in range(n)])
        min_dists[medoids] = 0
        probs = min_dists / (min_dists.sum() + 1e-10)
        medoids.append(int(np.random.choice(n, p=probs)))

    # Assign and refine (2 iterations)
    for _ in range(2):
        assignments = [[] for _ in range(k)]
        for i in range(n):
            closest = int(np.argmin([dist[i, m] for m in medoids]))
            assignments[closest].append(i)
        for c in range(k):
            if assignments[c]:
                costs = [sum(dist[i, j] for j in assignments[c]) for i in assignments[c]]
                medoids[c] = assignments[c][int(np.argmin(costs))]

    return medoids


# ---------------------------------------------------------------------------
# Kendall tau correlation
# ---------------------------------------------------------------------------

def kendall_tau(x, y):
    """Compute Kendall's tau rank correlation."""
    n = len(x)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi = x[i] - x[j]
            yi = y[i] - y[j]
            if xi * yi > 0:
                concordant += 1
            elif xi * yi < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


# ---------------------------------------------------------------------------
# Experiment 1: Metric benchmarking
# ---------------------------------------------------------------------------

def run_metric_benchmark():
    print("=" * 60)
    print("Experiment 1: Diversity Metric Benchmarking (500 responses)")
    print("=" * 60)

    diversity_levels = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    results = []

    for div_level in diversity_levels:
        texts = generate_response_set(500, n_words=30, diversity_level=div_level)

        t0 = time.time()
        d1 = distinct_n(texts, n=1)
        d2 = distinct_n(texts, n=2)
        d3 = distinct_n(texts, n=3)
        t_distinct = time.time() - t0

        t0 = time.time()
        sb = self_bleu(texts[:50], n=4)  # Subsample for speed
        t_bleu = time.time() - t0

        t0 = time.time()
        vs = vendi_score(texts[:100])  # Subsample for speed
        t_vendi = time.time() - t0

        t0 = time.time()
        fbd = fbd_score(texts[:200])
        t_fbd = time.time() - t0

        entry = {
            "diversity_level": div_level,
            "n_responses": len(texts),
            "metrics": {
                "distinct_1": round(d1, 4),
                "distinct_2": round(d2, 4),
                "distinct_3": round(d3, 4),
                "self_bleu": round(sb, 4),
                "vendi_score": round(vs, 4),
                "fbd_score": round(fbd, 4),
            },
            "timings_s": {
                "distinct_n": round(t_distinct, 4),
                "self_bleu": round(t_bleu, 4),
                "vendi_score": round(t_vendi, 4),
                "fbd_score": round(t_fbd, 4),
            },
        }
        results.append(entry)
        print(f"  div={div_level}: D1={d1:.3f}, D2={d2:.3f}, SB={sb:.3f}, "
              f"VS={vs:.2f}, FBD={fbd:.2f}")

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Metric correlation
# ---------------------------------------------------------------------------

def run_metric_correlation():
    print("\n" + "=" * 60)
    print("Experiment 2: Metric Correlation (Kendall Tau)")
    print("=" * 60)

    diversity_levels = np.linspace(0.05, 0.95, 20)
    metric_values = {"distinct_2": [], "self_bleu": [], "vendi": [], "fbd": []}

    for div_level in diversity_levels:
        texts = generate_response_set(100, n_words=25, diversity_level=div_level)
        metric_values["distinct_2"].append(distinct_n(texts, n=2))
        metric_values["self_bleu"].append(self_bleu(texts[:30]))
        metric_values["vendi"].append(vendi_score(texts[:50]))
        metric_values["fbd"].append(fbd_score(texts[:50]))

    metric_names = list(metric_values.keys())
    correlations = {}
    for i, m1 in enumerate(metric_names):
        for j, m2 in enumerate(metric_names):
            if i < j:
                tau = kendall_tau(metric_values[m1], metric_values[m2])
                key = f"{m1}_vs_{m2}"
                correlations[key] = round(tau, 4)
                print(f"  {key}: tau={tau:.4f}")

    return {
        "n_diversity_levels": len(diversity_levels),
        "correlations": correlations,
        "metric_ranges": {
            name: {"min": round(min(vals), 4), "max": round(max(vals), 4)}
            for name, vals in metric_values.items()
        },
    }


# ---------------------------------------------------------------------------
# Experiment 3: Selection algorithm comparison
# ---------------------------------------------------------------------------

def run_selection_comparison():
    print("\n" + "=" * 60)
    print("Experiment 3: Diverse Selection Algorithm Comparison")
    print("=" * 60)

    texts = generate_response_set(200, n_words=30, diversity_level=0.6)
    S = compute_similarity_matrix(texts)
    k = 20
    results = []

    methods = {
        "Flow": lambda: select_flow(texts, k, S),
        "MMR": lambda: select_mmr(texts, k, S=S),
        "DPP": lambda: select_dpp(texts, k, S=S),
        "K-Medoids": lambda: select_kmedoids(texts, k, S=S),
        "Random": lambda: list(np.random.choice(len(texts), k, replace=False)),
    }

    for name, selector in methods.items():
        t0 = time.time()
        selected = selector()
        elapsed = time.time() - t0

        selected_texts = [texts[i] for i in selected]
        d2 = distinct_n(selected_texts, n=2)
        sb = self_bleu(selected_texts)
        vs = vendi_score(selected_texts)

        # Average pairwise distance
        avg_dist = 0.0
        count = 0
        for i, j in combinations(selected, 2):
            avg_dist += 1 - S[i, j]
            count += 1
        avg_dist /= max(count, 1)

        entry = {
            "method": name,
            "k": k,
            "selection_time_s": round(elapsed, 4),
            "distinct_2": round(d2, 4),
            "self_bleu": round(sb, 4),
            "vendi_score": round(vs, 4),
            "avg_pairwise_distance": round(avg_dist, 4),
        }
        results.append(entry)
        print(f"  {name}: D2={d2:.3f}, SB={sb:.3f}, VS={vs:.2f}, "
              f"dist={avg_dist:.3f}, time={elapsed:.3f}s")

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Streaming vs batch diversity
# ---------------------------------------------------------------------------

def run_streaming_vs_batch():
    print("\n" + "=" * 60)
    print("Experiment 4: Streaming vs Batch Diversity")
    print("=" * 60)

    total_responses = 200
    texts = generate_response_set(total_responses, n_words=30, diversity_level=0.6)
    batch_sizes = [10, 20, 50, 100, 200]
    results = []

    for batch_size in batch_sizes:
        # Streaming: compute diversity incrementally
        t0 = time.time()
        streaming_d2_values = []
        for start in range(0, total_responses, batch_size):
            batch = texts[start:start + batch_size]
            if batch:
                streaming_d2_values.append(distinct_n(batch, n=2))
        streaming_d2 = float(np.mean(streaming_d2_values))
        streaming_time = time.time() - t0

        # Batch: compute on all at once
        t0 = time.time()
        batch_d2 = distinct_n(texts[:total_responses], n=2)
        batch_time = time.time() - t0

        # Also compare Vendi score
        streaming_vendi = float(np.mean([
            vendi_score(texts[s:s+batch_size])
            for s in range(0, total_responses, batch_size)
            if texts[s:s+batch_size]
        ]))
        batch_vendi = vendi_score(texts[:min(100, total_responses)])

        entry = {
            "batch_size": batch_size,
            "total_responses": total_responses,
            "n_batches": len(streaming_d2_values),
            "streaming_distinct_2": round(streaming_d2, 4),
            "batch_distinct_2": round(batch_d2, 4),
            "d2_discrepancy": round(abs(streaming_d2 - batch_d2), 4),
            "streaming_vendi": round(streaming_vendi, 4),
            "batch_vendi": round(batch_vendi, 4),
            "vendi_discrepancy": round(abs(streaming_vendi - batch_vendi), 4),
            "streaming_time_s": round(streaming_time, 4),
            "batch_time_s": round(batch_time, 4),
        }
        results.append(entry)
        print(f"  batch_size={batch_size}: stream_D2={streaming_d2:.3f} vs batch_D2={batch_d2:.3f}, "
              f"stream_V={streaming_vendi:.2f} vs batch_V={batch_vendi:.2f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Diversity Decoding Arena - Benchmark Experiments")
    print("=" * 60)
    t_start = time.time()

    all_results = {
        "experiment": "diversity_decoding_arena_benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    all_results["metric_benchmark"] = run_metric_benchmark()
    all_results["metric_correlation"] = run_metric_correlation()
    all_results["selection_comparison"] = run_selection_comparison()
    all_results["streaming_vs_batch"] = run_streaming_vs_batch()

    total_time = time.time() - t_start
    all_results["total_time_s"] = round(total_time, 2)

    # Summary
    best_selector = max(
        all_results["selection_comparison"],
        key=lambda e: e["avg_pairwise_distance"]
    )
    all_results["summary"] = {
        "best_selection_method": best_selector["method"],
        "best_pairwise_distance": best_selector["avg_pairwise_distance"],
        "strongest_metric_correlation": max(
            all_results["metric_correlation"]["correlations"].items(),
            key=lambda x: abs(x[1])
        ),
        "mean_streaming_batch_d2_discrepancy": round(np.mean(
            [e["d2_discrepancy"] for e in all_results["streaming_vs_batch"]]
        ), 4),
    }

    out_path = os.path.join(os.path.dirname(__file__), "diversity_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Results written to {out_path}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Summary: {json.dumps(all_results['summary'], indent=2, default=str)}")


if __name__ == "__main__":
    main()
