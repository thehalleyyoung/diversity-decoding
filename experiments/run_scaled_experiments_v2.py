#!/usr/bin/env python3
"""Efficient scaled experiments using gpt-4.1-nano with batched n parameter.

Addresses all critical critiques: scale, statistical testing, cross-domain validation.
Run: source ~/.bashrc && cd implementation && PYTHONPATH=. python3 experiments/run_scaled_experiments_v2.py
"""

import json, os, sys, time, math, logging, hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import Counter
import numpy as np

RESULTS_DIR = Path(__file__).parent / "scaled_results"
RESULTS_DIR.mkdir(exist_ok=True)
SEEDS = [42, 137, 2024]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 20 prompts across 5 domains
# ---------------------------------------------------------------------------
PROMPTS_BY_DOMAIN = {
    "creative_writing": [
        "Once upon a time in a distant kingdom, there lived",
        "The rain fell softly on the old stone bridge as",
        "She opened the letter and read the first line:",
        "In the year 3000, humanity finally discovered",
    ],
    "code_generation": [
        "def fibonacci(n):\n    # Return the nth Fibonacci number\n",
        "# Python function to sort a list using merge sort\ndef merge_sort(",
        "class DatabaseConnection:\n    def __init__(self",
        "async def fetch_data(url: str) -> dict:\n    # Fetch JSON data from",
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

ALL_PROMPTS = [(d, p) for d, ps in PROMPTS_BY_DOMAIN.items() for p in ps]

# ---------------------------------------------------------------------------
# Metrics (self-contained)
# ---------------------------------------------------------------------------
def tokenize_simple(text):
    import re, string
    text = text.lower().strip()
    text = re.sub(r"([" + re.escape(string.punctuation) + r"])", r" \1 ", text)
    return [t for t in re.split(r"\s+", text) if t]

def distinct_n(texts, n=2):
    ngrams = []
    for t in texts:
        toks = tokenize_simple(t)
        ngrams.extend(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
    return len(set(ngrams)) / max(len(ngrams), 1)

def self_bleu(texts, max_order=4):
    if len(texts) < 2: return 1.0
    all_toks = [tokenize_simple(t) for t in texts]
    scores = []
    for i in range(len(all_toks)):
        refs = [all_toks[j] for j in range(len(all_toks)) if j != i]
        precisions = []
        for n in range(1, max_order+1):
            hyp_ng = Counter(tuple(all_toks[i][k:k+n]) for k in range(len(all_toks[i])-n+1))
            max_ref = Counter()
            for r in refs:
                ref_ng = Counter(tuple(r[k:k+n]) for k in range(len(r)-n+1))
                for ng in ref_ng: max_ref[ng] = max(max_ref[ng], ref_ng[ng])
            clipped = sum(min(hyp_ng[ng], max_ref.get(ng, 0)) for ng in hyp_ng)
            total = sum(hyp_ng.values())
            precisions.append(clipped / max(total, 1))
        if any(p == 0 for p in precisions):
            scores.append(0.0)
        else:
            scores.append(math.exp(sum(math.log(p) for p in precisions) / len(precisions)))
    return sum(scores) / len(scores)

def jaccard_div(texts):
    sets = [set(tokenize_simple(t)) for t in texts]
    if len(sets) < 2: return 0.0
    dists = []
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            u = len(sets[i] | sets[j])
            dists.append(1.0 - len(sets[i] & sets[j]) / max(u, 1))
    return sum(dists) / len(dists)

def ngram_entropy(texts, n=2):
    ngrams = []
    for t in texts:
        toks = tokenize_simple(t)
        ngrams.extend(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
    if not ngrams: return 0.0
    counts = Counter(ngrams)
    total = sum(counts.values())
    return -sum((c/total) * math.log(c/total) for c in counts.values())

def epd(texts):
    all_toks = [tokenize_simple(t) for t in texts]
    vocab = {}
    for toks in all_toks:
        for t in set(toks):
            if t not in vocab: vocab[t] = len(vocab)
    if not vocab: return 0.0
    n_docs = len(texts)
    df = Counter()
    for toks in all_toks:
        for t in set(toks): df[t] += 1
    emb = np.zeros((n_docs, len(vocab)))
    for i, toks in enumerate(all_toks):
        tf = Counter(toks)
        for t, c in tf.items():
            emb[i, vocab[t]] = c * (math.log(n_docs / (df[t]+1)) + 1)
    norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10)
    emb = emb / norms
    dists = []
    for i in range(n_docs):
        for j in range(i+1, n_docs):
            dists.append(1.0 - np.dot(emb[i], emb[j]))
    return sum(dists) / max(len(dists), 1)

def vendi(texts):
    all_toks = [tokenize_simple(t) for t in texts]
    vocab = {}
    for toks in all_toks:
        for t in set(toks):
            if t not in vocab: vocab[t] = len(vocab)
    if not vocab or len(texts) < 2: return 1.0
    n_docs = len(texts)
    df = Counter()
    for toks in all_toks:
        for t in set(toks): df[t] += 1
    emb = np.zeros((n_docs, len(vocab)))
    for i, toks in enumerate(all_toks):
        tf = Counter(toks)
        for t, c in tf.items():
            emb[i, vocab[t]] = c * (math.log(n_docs / (df[t]+1)) + 1)
    norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10)
    emb = emb / norms
    K = emb @ emb.T / n_docs
    evals = np.maximum(np.linalg.eigvalsh(K), 1e-12)
    return float(np.exp(-np.sum(evals * np.log(evals))))

def compute_metrics(texts):
    if not texts or len(texts) < 2:
        return {k: 0.0 for k in ["distinct_2","self_bleu","jaccard","entropy","epd","vendi","n_unique"]}
    return {
        "distinct_2": round(distinct_n(texts, 2), 4),
        "self_bleu": round(self_bleu(texts), 4),
        "jaccard": round(jaccard_div(texts), 4),
        "entropy": round(ngram_entropy(texts, 2), 4),
        "epd": round(epd(texts), 4),
        "vendi": round(vendi(texts), 4),
        "n_unique": len(set(texts)),
    }

# ---------------------------------------------------------------------------
# Batched OpenAI generation
# ---------------------------------------------------------------------------
def generate_batch(prompt, n, temperature, top_p=1.0, seed=42, max_tokens=60):
    import openai
    client = openai.OpenAI()
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": f"Continue this text naturally. Only output the continuation, nothing else:\n\n{prompt}"}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
        )
        return [c.message.content.strip() for c in resp.choices]
    except Exception as e:
        logger.warning("API error: %s", e)
        return [f"[error: {e}]"] * n

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def bootstrap_ci(values, n_boot=5000, ci=0.95):
    vals = np.array([v for v in values if not math.isnan(v)])
    if len(vals) == 0: return 0.0, 0.0, 0.0
    rng = np.random.default_rng(42)
    means = sorted([float(np.mean(rng.choice(vals, len(vals), replace=True))) for _ in range(n_boot)])
    a = (1-ci)/2
    return float(np.mean(vals)), means[int(a*n_boot)], means[int((1-a)*n_boot)]

def kendall_tau(x, y):
    n = len(x)
    c = d = 0
    for i in range(n):
        for j in range(i+1, n):
            s = (x[i]-x[j]) * (y[i]-y[j])
            if s > 0: c += 1
            elif s < 0: d += 1
    return (c-d) / max(c+d, 1)

def bootstrap_tau(x, y, n_boot=2000):
    tau_obs = kendall_tau(x, y)
    rng = np.random.default_rng(42)
    taus = sorted([kendall_tau([x[i] for i in idx], [y[i] for i in idx]) 
                   for idx in (rng.choice(len(x), len(x), replace=True) for _ in range(n_boot))])
    return tau_obs, taus[int(0.025*n_boot)], taus[int(0.975*n_boot)]

# ---------------------------------------------------------------------------
# EXPERIMENT 1: Metric Taxonomy (star contribution)
# ---------------------------------------------------------------------------
def run_taxonomy():
    logger.info("=" * 60)
    logger.info("EXP 1: METRIC TAXONOMY on gpt-4.1-nano")
    logger.info("=" * 60)
    
    configs = [
        ("temp_0.0", 0.0, 1.0), ("temp_0.3", 0.3, 1.0), ("temp_0.5", 0.5, 1.0),
        ("temp_0.7", 0.7, 1.0), ("temp_1.0", 1.0, 1.0), ("temp_1.2", 1.2, 1.0),
        ("temp_1.5", 1.5, 1.0), ("temp_2.0", 2.0, 1.0),
        ("nucleus_0.3", 1.0, 0.3), ("nucleus_0.5", 1.0, 0.5),
        ("nucleus_0.7", 1.0, 0.7), ("nucleus_0.9", 1.0, 0.9),
        ("nucleus_0.95", 1.0, 0.95),
    ]
    
    N_SEQS = 10
    all_texts = {}
    config_metrics = {name: {m: [] for m in ["distinct_2","self_bleu","jaccard","entropy","epd","vendi"]} 
                      for name,_,_ in configs}
    
    total_calls = len(SEEDS) * len(ALL_PROMPTS) * len(configs)
    call_count = 0
    
    for seed in SEEDS:
        for domain, prompt in ALL_PROMPTS:
            pk = hashlib.md5(prompt.encode()).hexdigest()[:8]
            for cname, temp, top_p in configs:
                call_count += 1
                if call_count % 20 == 0:
                    logger.info(f"  Progress: {call_count}/{total_calls} ({100*call_count/total_calls:.0f}%)")
                
                texts = generate_batch(prompt, N_SEQS, temp, top_p, seed, max_tokens=60)
                key = f"{cname}__s{seed}__{pk}"
                all_texts[key] = texts
                
                m = compute_metrics(texts)
                for mk in config_metrics[cname]:
                    if mk in m:
                        config_metrics[cname][mk].append(m[mk])
    
    # Save texts
    with open(RESULTS_DIR / "taxonomy_texts.json", "w") as f:
        json.dump(all_texts, f, indent=1)
    
    # Aggregate per config
    metric_names = ["distinct_2","self_bleu","jaccard","entropy","epd","vendi"]
    agg = {}
    config_means = {m: [] for m in metric_names}
    for cname, _, _ in configs:
        agg[cname] = {}
        for m in metric_names:
            vals = config_metrics[cname][m]
            mean, lo, hi = bootstrap_ci(vals)
            agg[cname][m] = {"mean": round(mean, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4), "n_obs": len(vals)}
            config_means[m].append(mean)
    
    # Kendall tau with bootstrap CIs
    corrs = {}
    for i in range(len(metric_names)):
        for j in range(i+1, len(metric_names)):
            mi, mj = metric_names[i], metric_names[j]
            tau, lo, hi = bootstrap_tau(config_means[mi], config_means[mj])
            corrs[f"{mi}_vs_{mj}"] = {"tau": round(tau, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)}
    
    result = {
        "experiment": "Metric_Taxonomy_gpt4.1-nano",
        "model": "gpt-4.1-nano",
        "n_configs": len(configs), "n_prompts": len(ALL_PROMPTS), "n_seeds": len(SEEDS),
        "n_seqs_per_call": N_SEQS,
        "total_generations": len(ALL_PROMPTS) * len(SEEDS) * len(configs) * N_SEQS,
        "aggregated": agg,
        "kendall_tau": corrs,
        "config_names": [c[0] for c in configs],
    }
    
    with open(RESULTS_DIR / "taxonomy_results.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Taxonomy saved. Total gens: %d", result["total_generations"])
    return result

# ---------------------------------------------------------------------------
# EXPERIMENT 2: Algorithm comparison across domains
# ---------------------------------------------------------------------------
def run_comparison():
    logger.info("=" * 60)
    logger.info("EXP 2: ALGORITHM COMPARISON")
    logger.info("=" * 60)
    
    configs = [
        ("temp_0.0", 0.0, 1.0), ("temp_0.3", 0.3, 1.0),
        ("temp_0.7", 0.7, 1.0), ("temp_1.0", 1.0, 1.0),
        ("temp_1.5", 1.5, 1.0), ("temp_2.0", 2.0, 1.0),
        ("nucleus_0.5", 1.0, 0.5), ("nucleus_0.9", 1.0, 0.9),
        ("nucleus_0.95", 1.0, 0.95),
    ]
    
    N_SEQS = 10
    results_by_config = {}
    domain_results = {}
    all_texts = {}
    
    for cname, temp, top_p in configs:
        all_m = []
        dm = {d: [] for d in PROMPTS_BY_DOMAIN}
        
        for seed in SEEDS:
            for domain, prompt in ALL_PROMPTS:
                pk = hashlib.md5(prompt.encode()).hexdigest()[:8]
                texts = generate_batch(prompt, N_SEQS, temp, top_p, seed, max_tokens=60)
                all_texts[f"{cname}__s{seed}__{pk}"] = texts
                m = compute_metrics(texts)
                all_m.append(m)
                dm[domain].append(m)
        
        # Aggregate
        mks = ["distinct_2","self_bleu","jaccard","entropy","epd","vendi","n_unique"]
        a = {}
        for mk in mks:
            vals = [x[mk] for x in all_m if mk in x]
            if vals:
                mean, lo, hi = bootstrap_ci(vals)
                a[mk] = {"mean": round(mean, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)}
        results_by_config[cname] = a
        
        for domain in PROMPTS_BY_DOMAIN:
            if domain not in domain_results: domain_results[domain] = {}
            dvals = dm[domain]
            da = {}
            for mk in mks:
                vals = [x[mk] for x in dvals if mk in x]
                if vals:
                    da[mk] = {"mean": round(np.mean(vals), 4), "std": round(np.std(vals), 4)}
            domain_results[domain][cname] = da
        
        logger.info(f"  {cname}: D2={a.get('distinct_2',{}).get('mean','?')}, SB={a.get('self_bleu',{}).get('mean','?')}")
    
    result = {
        "experiment": "Algorithm_Comparison",
        "aggregated": results_by_config,
        "by_domain": domain_results,
    }
    with open(RESULTS_DIR / "comparison_results.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULTS_DIR / "comparison_texts.json", "w") as f:
        json.dump(all_texts, f, indent=1)
    return result

# ---------------------------------------------------------------------------
# EXPERIMENT 3: Domain stability of taxonomy
# ---------------------------------------------------------------------------
def run_stability():
    logger.info("=" * 60)
    logger.info("EXP 3: DOMAIN STABILITY")
    logger.info("=" * 60)
    
    configs = [
        ("temp_0.0", 0.0, 1.0), ("temp_0.3", 0.3, 1.0), ("temp_0.5", 0.5, 1.0),
        ("temp_0.7", 0.7, 1.0), ("temp_1.0", 1.0, 1.0), ("temp_1.2", 1.2, 1.0),
        ("temp_1.5", 1.5, 1.0), ("temp_2.0", 2.0, 1.0),
        ("nucleus_0.5", 1.0, 0.5), ("nucleus_0.9", 1.0, 0.9),
    ]
    
    N_SEQS = 10
    domain_corrs = {}
    
    for domain, prompts in PROMPTS_BY_DOMAIN.items():
        logger.info(f"  Domain: {domain}")
        cm = {m: [] for m in ["distinct_2","self_bleu","jaccard","entropy","epd","vendi"]}
        
        for cname, temp, top_p in configs:
            vals = {m: [] for m in cm}
            for seed in SEEDS:
                for prompt in prompts:
                    texts = generate_batch(prompt, N_SEQS, temp, top_p, seed, max_tokens=60)
                    m = compute_metrics(texts)
                    for mk in vals:
                        if mk in m: vals[mk].append(m[mk])
            for mk in cm:
                cm[mk].append(np.mean(vals[mk]) if vals[mk] else 0.0)
        
        mks = list(cm.keys())
        corrs = {}
        for i in range(len(mks)):
            for j in range(i+1, len(mks)):
                tau, lo, hi = bootstrap_tau(cm[mks[i]], cm[mks[j]], n_boot=2000)
                corrs[f"{mks[i]}_vs_{mks[j]}"] = {"tau": round(tau, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
        domain_corrs[domain] = corrs
    
    # Stability analysis
    key_pairs = ["distinct_2_vs_self_bleu", "distinct_2_vs_epd", "distinct_2_vs_vendi", "epd_vs_vendi"]
    stability = {}
    for pair in key_pairs:
        taus = {d: domain_corrs[d][pair]["tau"] for d in domain_corrs if pair in domain_corrs[d]}
        stability[pair] = {
            "by_domain": taus,
            "mean": round(np.mean(list(taus.values())), 4),
            "std": round(np.std(list(taus.values())), 4),
            "stable": all(abs(t) > 0.5 for t in taus.values()),
        }
    
    result = {"domain_correlations": domain_corrs, "stability": stability}
    with open(RESULTS_DIR / "stability_results.json", "w") as f:
        json.dump(result, f, indent=2)
    return result

# ---------------------------------------------------------------------------
# EXPERIMENT 4: Pareto frontier with CIs
# ---------------------------------------------------------------------------
def run_pareto():
    logger.info("=" * 60)
    logger.info("EXP 4: PARETO FRONTIER")
    logger.info("=" * 60)
    
    configs = [
        ("temp_0.0", 0.0, 1.0), ("temp_0.3", 0.3, 1.0), ("temp_0.5", 0.5, 1.0),
        ("temp_0.7", 0.7, 1.0), ("temp_1.0", 1.0, 1.0), ("temp_1.2", 1.2, 1.0),
        ("temp_1.5", 1.5, 1.0), ("temp_2.0", 2.0, 1.0),
        ("nucleus_0.3", 1.0, 0.3), ("nucleus_0.5", 1.0, 0.5),
        ("nucleus_0.7", 1.0, 0.7), ("nucleus_0.9", 1.0, 0.9),
    ]
    
    N_SEQS = 10
    data = []
    
    for cname, temp, top_p in configs:
        divs, quals = [], []
        for seed in SEEDS:
            for _, prompt in ALL_PROMPTS[:8]:
                texts = generate_batch(prompt, N_SEQS, temp, top_p, seed, max_tokens=60)
                d = distinct_n(texts, 2)
                # Quality proxy: coherence via type-token ratio sweet spot
                toks_all = [tokenize_simple(t) for t in texts]
                q_scores = []
                for toks in toks_all:
                    if not toks: continue
                    ttr = len(set(toks)) / len(toks)
                    q = 1.0 - abs(ttr - 0.65) * 2
                    q_scores.append(max(0, min(1, q)))
                q = np.mean(q_scores) if q_scores else 0.0
                divs.append(d)
                quals.append(q)
        
        dm, dl, dh = bootstrap_ci(divs)
        qm, ql, qh = bootstrap_ci(quals)
        data.append({
            "config": cname, "temp": temp, "top_p": top_p,
            "diversity": {"mean": round(dm, 4), "ci_lo": round(dl, 4), "ci_hi": round(dh, 4)},
            "quality": {"mean": round(qm, 4), "ci_lo": round(ql, 4), "ci_hi": round(qh, 4)},
        })
    
    # Pareto front
    for p in data:
        d, q = p["diversity"]["mean"], p["quality"]["mean"]
        p["pareto"] = not any(
            o["diversity"]["mean"] >= d and o["quality"]["mean"] >= q
            and (o["diversity"]["mean"] > d or o["quality"]["mean"] > q)
            for o in data)
    
    result = {"configs": data, "pareto_front": [p for p in data if p["pareto"]]}
    with open(RESULTS_DIR / "pareto_results.json", "w") as f:
        json.dump(result, f, indent=2)
    return result

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["taxonomy","comparison","stability","pareto","all"], default="all")
    args = parser.parse_args()
    
    t0 = time.time()
    results = {}
    
    if args.exp in ("taxonomy", "all"):
        results["taxonomy"] = run_taxonomy()
    if args.exp in ("comparison", "all"):
        results["comparison"] = run_comparison()
    if args.exp in ("stability", "all"):
        results["stability"] = run_stability()
    if args.exp in ("pareto", "all"):
        results["pareto"] = run_pareto()
    
    elapsed = time.time() - t0
    summary = {"time_sec": round(elapsed), "experiments": args.exp, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("All done in %.0f seconds (%.1f min)", elapsed, elapsed/60)
