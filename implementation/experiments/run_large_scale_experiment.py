#!/usr/bin/env python3
"""Large-scale diversity metric taxonomy experiment with semantic embeddings.

Addresses critiques W1 (scale), W4 (consistent implementations), W6 (multi-metric Pareto),
and W9 (semantic metrics) by running 40 prompts × 13 configs × 10 sequences on gpt-4.1-nano
with text-embedding-3-small for neural semantic diversity.

Run: source ~/.bashrc && cd implementation && python3 experiments/run_large_scale_experiment.py
"""

import json, os, sys, time, math, logging, hashlib, itertools
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import Counter
import numpy as np
from scipy.stats import kendalltau
from scipy.spatial.distance import pdist, squareform, cosine

RESULTS_DIR = Path(__file__).parent / "large_scale_results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_FILE = RESULTS_DIR / "generation_cache.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 40 prompts across 5 domains (8 per domain) - addresses W1
# ---------------------------------------------------------------------------
PROMPTS_BY_DOMAIN = {
    "creative_writing": [
        "Once upon a time in a distant kingdom, there lived",
        "The rain fell softly on the old stone bridge as",
        "She opened the letter and read the first line:",
        "In the year 3000, humanity finally discovered",
        "The detective examined the crime scene and noticed",
        "Under the pale moonlight, the garden came alive with",
        "He had been walking for days when he finally reached",
        "The old woman sat by the fire and began to tell",
    ],
    "code_generation": [
        "def fibonacci(n):\n    # Return the nth Fibonacci number\n",
        "# Python function to sort a list using merge sort\ndef merge_sort(",
        "class DatabaseConnection:\n    def __init__(self",
        "async def fetch_data(url: str) -> dict:\n    # Fetch JSON data from",
        "def binary_search(arr, target):\n    # Implement binary search\n",
        "class LRUCache:\n    def __init__(self, capacity: int):\n",
        "def parse_csv(filepath: str) -> list:\n    # Parse CSV file and return rows\n",
        "def validate_email(email: str) -> bool:\n    # Check if email is valid\n",
    ],
    "scientific": [
        "The relationship between quantum entanglement and",
        "Recent advances in CRISPR gene editing have shown that",
        "The standard model of particle physics predicts that",
        "Climate models suggest that by 2050, global temperatures will",
        "The discovery of gravitational waves confirmed that",
        "Photosynthesis in plants converts light energy into",
        "The human immune system responds to viral infections by",
        "Neural plasticity allows the brain to reorganize by",
    ],
    "business": [
        "The key to a successful startup strategy is",
        "In today's competitive market, companies must focus on",
        "The quarterly earnings report showed that revenue increased by",
        "To improve customer retention, the most effective approach is",
        "Supply chain disruptions have forced companies to",
        "The merger between the two companies will create",
        "Digital transformation requires organizations to",
        "Sustainable business practices can improve profitability by",
    ],
    "dialogue": [
        "Customer: I'd like to return this product.\nAgent:",
        "Student: Can you explain how neural networks work?\nTeacher:",
        "User: What's the best way to learn Python?\nAssistant:",
        "Interviewer: Tell me about your biggest achievement.\nCandidate:",
        "Patient: I've been having trouble sleeping lately.\nDoctor:",
        "Manager: We need to discuss the project timeline.\nDeveloper:",
        "Tourist: Can you recommend a good restaurant nearby?\nLocal:",
        "Parent: My child is struggling with math.\nTutor:",
    ],
}

ALL_PROMPTS = [(d, p) for d, ps in PROMPTS_BY_DOMAIN.items() for p in ps]
assert len(ALL_PROMPTS) == 40, f"Expected 40 prompts, got {len(ALL_PROMPTS)}"

# Configurations: 13 decoding configs (same as original)
CONFIGS = [
    {"name": "temp_0.0", "temperature": 0.0, "top_p": 1.0},
    {"name": "temp_0.3", "temperature": 0.3, "top_p": 1.0},
    {"name": "temp_0.5", "temperature": 0.5, "top_p": 1.0},
    {"name": "temp_0.7", "temperature": 0.7, "top_p": 1.0},
    {"name": "temp_1.0", "temperature": 1.0, "top_p": 1.0},
    {"name": "temp_1.2", "temperature": 1.2, "top_p": 1.0},
    {"name": "temp_1.5", "temperature": 1.5, "top_p": 1.0},
    {"name": "temp_2.0", "temperature": 2.0, "top_p": 1.0},
    {"name": "nucleus_0.3", "temperature": 1.0, "top_p": 0.3},
    {"name": "nucleus_0.5", "temperature": 1.0, "top_p": 0.5},
    {"name": "nucleus_0.7", "temperature": 1.0, "top_p": 0.7},
    {"name": "nucleus_0.9", "temperature": 1.0, "top_p": 0.9},
    {"name": "nucleus_0.95", "temperature": 1.0, "top_p": 0.95},
]

SEEDS = [42, 137, 2024]
N_SEQS = 10  # sequences per config-prompt-seed

# ---------------------------------------------------------------------------
# Metric implementations (consistent across all analyses)
# ---------------------------------------------------------------------------
import re, string
_PUNCT_RE = re.compile(r"([" + re.escape(string.punctuation) + r"])")
_WS_RE = re.compile(r"\s+")

def tokenize(text):
    text = text.lower().strip()
    text = _PUNCT_RE.sub(r" \1 ", text)
    return [t for t in _WS_RE.split(text) if t]

def distinct_n(texts, n=2):
    ngrams = []
    for t in texts:
        toks = tokenize(t)
        ngrams.extend(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
    return len(set(ngrams)) / max(len(ngrams), 1)

def self_bleu(texts, max_order=4):
    if len(texts) < 2: return 1.0
    all_toks = [tokenize(t) for t in texts]
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
            clipped = sum(min(hyp_ng[ng], max_ref.get(ng,0)) for ng in hyp_ng)
            total = sum(hyp_ng.values())
            precisions.append(clipped / max(total, 1))
        if any(p == 0 for p in precisions):
            scores.append(0.0)
        else:
            scores.append(math.exp(sum(math.log(p) for p in precisions) / len(precisions)))
    return sum(scores) / len(scores)

def jaccard_div(texts):
    sets = [set(tokenize(t)) for t in texts]
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
        toks = tokenize(t)
        ngrams.extend(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
    if not ngrams: return 0.0
    counts = Counter(ngrams)
    total = sum(counts.values())
    return -sum((c/total) * math.log(c/total) for c in counts.values())

def tfidf_epd(texts):
    """EPD using TF-IDF (consistent implementation for all models)."""
    all_toks = [tokenize(t) for t in texts]
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
            dists.append(1.0 - float(np.dot(emb[i], emb[j])))
    return sum(dists) / max(len(dists), 1)

def vendi_score(texts):
    """Vendi Score using TF-IDF kernel (consistent implementation)."""
    all_toks = [tokenize(t) for t in texts]
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

def semantic_epd(embeddings):
    """Semantic embedding pairwise distance using neural embeddings."""
    if len(embeddings) < 2: return 0.0
    emb = np.array(embeddings)
    norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10)
    emb = emb / norms
    dists = []
    for i in range(len(emb)):
        for j in range(i+1, len(emb)):
            dists.append(1.0 - float(np.dot(emb[i], emb[j])))
    return sum(dists) / max(len(dists), 1)

def semantic_vendi(embeddings):
    """Vendi Score using neural embeddings."""
    if len(embeddings) < 2: return 1.0
    emb = np.array(embeddings)
    norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10)
    emb = emb / norms
    n = len(emb)
    K = emb @ emb.T / n
    evals = np.maximum(np.linalg.eigvalsh(K), 1e-12)
    return float(np.exp(-np.sum(evals * np.log(evals))))

def compute_all_metrics(texts, embeddings=None):
    """Compute all 8 metrics (6 lexical + 2 semantic if embeddings provided)."""
    m = {
        "distinct_2": round(distinct_n(texts, 2), 4),
        "self_bleu": round(self_bleu(texts), 4),
        "jaccard": round(jaccard_div(texts), 4),
        "entropy": round(ngram_entropy(texts, 2), 4),
        "tfidf_epd": round(tfidf_epd(texts), 4),
        "tfidf_vendi": round(vendi_score(texts), 4),
        "n_unique": len(set(texts)),
    }
    if embeddings is not None and len(embeddings) >= 2:
        m["semantic_epd"] = round(semantic_epd(embeddings), 4)
        m["semantic_vendi"] = round(semantic_vendi(embeddings), 4)
    return m

# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------
def generate_batch(prompt, n, temperature, top_p=1.0, seed=42, max_tokens=80):
    import openai
    client = openai.OpenAI()
    kwargs = dict(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": f"Continue this text naturally. Only output the continuation, nothing else:\n\n{prompt}"}],
        top_p=top_p,
        max_tokens=max_tokens,
        n=n,
        seed=seed,
    )
    if temperature > 0:
        kwargs["temperature"] = temperature
    else:
        kwargs["temperature"] = 0
    try:
        resp = client.chat.completions.create(**kwargs)
        return [c.message.content.strip() for c in resp.choices]
    except Exception as e:
        logger.warning("API error: %s", e)
        return [f"[error: {e}]"] * n

def get_embeddings(texts, batch_size=100):
    """Get embeddings from text-embedding-3-small."""
    import openai
    client = openai.OpenAI()
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
            all_emb.extend([d.embedding for d in resp.data])
        except Exception as e:
            logger.warning("Embedding error: %s", e)
            all_emb.extend([[0.0]*1536] * len(batch))
    return all_emb

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_generation_phase():
    """Phase 1: Generate all texts and cache them."""
    if CACHE_FILE.exists():
        logger.info("Loading cached generations from %s", CACHE_FILE)
        with open(CACHE_FILE) as f:
            return json.load(f)

    results = {}
    total = len(ALL_PROMPTS) * len(CONFIGS) * len(SEEDS)
    done = 0
    t0 = time.time()

    for (domain, prompt) in ALL_PROMPTS:
        for cfg in CONFIGS:
            for seed in SEEDS:
                key = f"{domain}|{prompt[:50]}|{cfg['name']}|{seed}"
                texts = generate_batch(
                    prompt, N_SEQS,
                    temperature=cfg["temperature"],
                    top_p=cfg["top_p"],
                    seed=seed,
                    max_tokens=80
                )
                results[key] = {
                    "domain": domain,
                    "prompt": prompt,
                    "config": cfg["name"],
                    "seed": seed,
                    "texts": texts
                }
                done += 1
                if done % 20 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    remaining = (total - done) / rate
                    logger.info(f"Generated {done}/{total} ({done*N_SEQS} texts). "
                               f"ETA: {remaining/60:.1f}min")

    # Save cache
    with open(CACHE_FILE, 'w') as f:
        json.dump(results, f)
    logger.info(f"Generated {done*N_SEQS} total texts in {time.time()-t0:.1f}s")
    return results

def run_embedding_phase(gen_results):
    """Phase 2: Get semantic embeddings for all texts."""
    emb_file = RESULTS_DIR / "embedding_cache.json"
    if emb_file.exists():
        logger.info("Loading cached embeddings")
        with open(emb_file) as f:
            return json.load(f)

    # Collect all unique texts
    all_texts = set()
    for v in gen_results.values():
        all_texts.update(v["texts"])
    all_texts = [t for t in all_texts if not t.startswith("[error")]
    logger.info(f"Getting embeddings for {len(all_texts)} unique texts")

    # Get embeddings in batches
    embeddings = get_embeddings(all_texts, batch_size=200)
    text_to_emb = {t: e for t, e in zip(all_texts, embeddings)}

    with open(emb_file, 'w') as f:
        json.dump(text_to_emb, f)
    logger.info(f"Cached {len(text_to_emb)} embeddings")
    return text_to_emb

def run_analysis(gen_results, text_to_emb):
    """Phase 3: Compute metrics and Kendall tau analysis."""
    logger.info("Computing metrics for all generation groups...")

    # Compute metrics for each (config, prompt, seed) triple
    metric_records = []
    for key, rec in gen_results.items():
        texts = [t for t in rec["texts"] if not t.startswith("[error")]
        if len(texts) < 2:
            continue
        embs = [text_to_emb.get(t) for t in texts]
        embs = [e for e in embs if e is not None]
        metrics = compute_all_metrics(texts, embs if len(embs) >= 2 else None)
        metrics["domain"] = rec["domain"]
        metrics["config"] = rec["config"]
        metrics["seed"] = rec["seed"]
        metrics["prompt"] = rec["prompt"][:60]
        metrics["n_texts"] = len(texts)
        metric_records.append(metrics)

    logger.info(f"Computed metrics for {len(metric_records)} groups")

    # ---- Per-prompt Kendall tau (the right methodology) ----
    metric_names = ["distinct_2", "self_bleu", "jaccard", "entropy",
                    "tfidf_epd", "tfidf_vendi", "semantic_epd", "semantic_vendi"]

    # Group by (prompt, seed)
    from collections import defaultdict
    prompt_seed_groups = defaultdict(list)
    for rec in metric_records:
        ps_key = (rec["prompt"], rec["seed"])
        prompt_seed_groups[ps_key].append(rec)

    # For each (prompt, seed), compute pairwise Kendall tau across configs
    pair_taus = defaultdict(list)
    for ps_key, recs in prompt_seed_groups.items():
        if len(recs) < 5:  # need enough configs
            continue
        # Sort by config name for consistency
        recs_sorted = sorted(recs, key=lambda r: r["config"])
        for i, m1 in enumerate(metric_names):
            for j, m2 in enumerate(metric_names):
                if j <= i:
                    continue
                vals1 = [r.get(m1) for r in recs_sorted]
                vals2 = [r.get(m2) for r in recs_sorted]
                if any(v is None for v in vals1) or any(v is None for v in vals2):
                    continue
                if len(set(vals1)) < 2 or len(set(vals2)) < 2:
                    continue
                tau, pval = kendalltau(vals1, vals2)
                if not np.isnan(tau):
                    pair_taus[(m1, m2)].append(tau)

    # Compute mean and std of tau for each pair
    tau_summary = {}
    for (m1, m2), taus in pair_taus.items():
        tau_summary[f"{m1}_vs_{m2}"] = {
            "mean_tau": round(float(np.mean(taus)), 4),
            "std_tau": round(float(np.std(taus)), 4),
            "n_pairs": len(taus),
            "ci_lower": round(float(np.percentile(taus, 2.5)), 4),
            "ci_upper": round(float(np.percentile(taus, 97.5)), 4),
        }

    # ---- Per-domain analysis ----
    domain_taus = {}
    for domain in PROMPTS_BY_DOMAIN:
        domain_recs = [r for r in metric_records if r["domain"] == domain]
        domain_ps_groups = defaultdict(list)
        for rec in domain_recs:
            ps_key = (rec["prompt"], rec["seed"])
            domain_ps_groups[ps_key].append(rec)

        domain_pair_taus = defaultdict(list)
        for ps_key, recs in domain_ps_groups.items():
            if len(recs) < 5:
                continue
            recs_sorted = sorted(recs, key=lambda r: r["config"])
            for i, m1 in enumerate(metric_names):
                for j, m2 in enumerate(metric_names):
                    if j <= i:
                        continue
                    vals1 = [r.get(m1) for r in recs_sorted]
                    vals2 = [r.get(m2) for r in recs_sorted]
                    if any(v is None for v in vals1) or any(v is None for v in vals2):
                        continue
                    if len(set(vals1)) < 2 or len(set(vals2)) < 2:
                        continue
                    tau, _ = kendalltau(vals1, vals2)
                    if not np.isnan(tau):
                        domain_pair_taus[(m1, m2)].append(tau)

        domain_taus[domain] = {}
        for (m1, m2), taus in domain_pair_taus.items():
            domain_taus[domain][f"{m1}_vs_{m2}"] = {
                "mean_tau": round(float(np.mean(taus)), 4),
                "std_tau": round(float(np.std(taus)), 4),
                "n_pairs": len(taus),
            }

    # ---- Config-level metrics (averaged across prompts and seeds) ----
    config_metrics = defaultdict(lambda: defaultdict(list))
    for rec in metric_records:
        for m in metric_names:
            if rec.get(m) is not None:
                config_metrics[rec["config"]][m].append(rec[m])

    config_summary = {}
    for cfg, metrics in config_metrics.items():
        config_summary[cfg] = {m: {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
        } for m, vals in metrics.items()}

    # ---- Cross-domain stability ----
    domains = list(PROMPTS_BY_DOMAIN.keys())
    cross_domain = {}
    for i, d1 in enumerate(domains):
        for j, d2 in enumerate(domains):
            if j <= i:
                continue
            # Compare tau vectors between domains
            common_pairs = set(domain_taus.get(d1, {}).keys()) & set(domain_taus.get(d2, {}).keys())
            if len(common_pairs) < 3:
                continue
            v1 = [domain_taus[d1][p]["mean_tau"] for p in sorted(common_pairs)]
            v2 = [domain_taus[d2][p]["mean_tau"] for p in sorted(common_pairs)]
            tau, _ = kendalltau(v1, v2)
            cross_domain[f"{d1}_vs_{d2}"] = round(float(tau), 4) if not np.isnan(tau) else 0.0

    # ---- Assemble final results ----
    total_texts = sum(r["n_texts"] for r in metric_records)
    results = {
        "experiment": "large_scale_taxonomy_v2",
        "description": "40 prompts × 13 configs × 3 seeds × 10 sequences with semantic embeddings",
        "total_generations": total_texts,
        "n_prompts": len(ALL_PROMPTS),
        "n_configs": len(CONFIGS),
        "n_seeds": len(SEEDS),
        "n_domains": len(PROMPTS_BY_DOMAIN),
        "n_prompt_seed_pairs": len(prompt_seed_groups),
        "metrics_computed": metric_names,
        "kendall_tau_per_prompt": tau_summary,
        "domain_results": domain_taus,
        "cross_domain_stability": cross_domain,
        "config_metrics": config_summary,
    }

    out_file = RESULTS_DIR / "large_scale_analysis.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved analysis to {out_file}")

    # Also save the raw metric records
    with open(RESULTS_DIR / "metric_records.json", 'w') as f:
        json.dump(metric_records, f, indent=2)

    return results

def print_summary(results):
    """Print key findings."""
    print("\n" + "="*70)
    print("LARGE-SCALE TAXONOMY RESULTS")
    print("="*70)
    print(f"Total generations: {results['total_generations']}")
    print(f"Prompts: {results['n_prompts']}, Configs: {results['n_configs']}, Seeds: {results['n_seeds']}")
    print(f"Prompt-seed pairs: {results['n_prompt_seed_pairs']}")
    print()

    print("Per-prompt mean Kendall τ (key pairs):")
    key_pairs = [
        "distinct_2_vs_self_bleu",
        "distinct_2_vs_entropy",
        "distinct_2_vs_jaccard",
        "distinct_2_vs_tfidf_vendi",
        "distinct_2_vs_tfidf_epd",
        "distinct_2_vs_semantic_epd",
        "distinct_2_vs_semantic_vendi",
        "tfidf_epd_vs_semantic_epd",
        "tfidf_vendi_vs_semantic_vendi",
        "semantic_epd_vs_semantic_vendi",
        "self_bleu_vs_semantic_epd",
        "entropy_vs_semantic_epd",
    ]
    tau_data = results["kendall_tau_per_prompt"]
    for pair in key_pairs:
        if pair in tau_data:
            d = tau_data[pair]
            print(f"  {pair}: τ = {d['mean_tau']:.3f} ± {d['std_tau']:.3f} (n={d['n_pairs']})")

    print("\nCross-domain stability:")
    for pair, tau in results.get("cross_domain_stability", {}).items():
        print(f"  {pair}: {tau:.3f}")

    # Build full 8x8 matrix for display
    metrics = results["metrics_computed"]
    print(f"\nFull {len(metrics)}×{len(metrics)} mean τ matrix:")
    header = "            " + "  ".join(f"{m[:8]:>8s}" for m in metrics)
    print(header)
    for m1 in metrics:
        row = f"{m1[:12]:12s}"
        for m2 in metrics:
            if m1 == m2:
                row += f"  {'1.000':>8s}"
            else:
                key1 = f"{m1}_vs_{m2}"
                key2 = f"{m2}_vs_{m1}"
                if key1 in tau_data:
                    row += f"  {tau_data[key1]['mean_tau']:8.3f}"
                elif key2 in tau_data:
                    row += f"  {tau_data[key2]['mean_tau']:8.3f}"
                else:
                    row += f"  {'---':>8s}"
        print(row)

# ===========================================================================
# SECTION 1: Extended Model Configurations
# ===========================================================================

MODEL_CONFIGS = {
    "gpt-4.1-nano": {
        "provider": "openai",
        "model_id": "gpt-4.1-nano",
        "max_tokens": 80,
        "description": "Lightweight GPT-4.1 variant for fast iteration",
        "supports_logprobs": True,
        "supports_n": True,
        "cost_per_1k_input": 0.0001,
        "cost_per_1k_output": 0.0004,
    },
    "gpt-4.1-mini": {
        "provider": "openai",
        "model_id": "gpt-4.1-mini",
        "max_tokens": 100,
        "description": "Mid-range GPT-4.1 variant",
        "supports_logprobs": True,
        "supports_n": True,
        "cost_per_1k_input": 0.0004,
        "cost_per_1k_output": 0.0016,
    },
    "gpt-2-small": {
        "provider": "huggingface",
        "model_id": "gpt2",
        "max_tokens": 80,
        "description": "GPT-2 small (117M parameters)",
        "supports_logprobs": True,
        "supports_n": False,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "local": True,
    },
    "gpt-2-medium": {
        "provider": "huggingface",
        "model_id": "gpt2-medium",
        "max_tokens": 80,
        "description": "GPT-2 medium (345M parameters)",
        "supports_logprobs": True,
        "supports_n": False,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "local": True,
    },
    "gpt-2-large": {
        "provider": "huggingface",
        "model_id": "gpt2-large",
        "max_tokens": 80,
        "description": "GPT-2 large (774M parameters)",
        "supports_logprobs": True,
        "supports_n": False,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "local": True,
    },
    "gpt-2-xl": {
        "provider": "huggingface",
        "model_id": "gpt2-xl",
        "max_tokens": 80,
        "description": "GPT-2 XL (1.5B parameters)",
        "supports_logprobs": True,
        "supports_n": False,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "local": True,
    },
    "claude-api-placeholder": {
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307",
        "max_tokens": 100,
        "description": "Anthropic Claude 3 Haiku placeholder",
        "supports_logprobs": False,
        "supports_n": False,
        "cost_per_1k_input": 0.00025,
        "cost_per_1k_output": 0.00125,
    },
    "gemini-api-placeholder": {
        "provider": "google",
        "model_id": "gemini-1.5-flash",
        "max_tokens": 100,
        "description": "Google Gemini 1.5 Flash placeholder",
        "supports_logprobs": False,
        "supports_n": False,
        "cost_per_1k_input": 0.000075,
        "cost_per_1k_output": 0.0003,
    },
    "llama-api-placeholder": {
        "provider": "together",
        "model_id": "meta-llama/Llama-3-8b-chat-hf",
        "max_tokens": 80,
        "description": "Llama 3 8B via Together API placeholder",
        "supports_logprobs": True,
        "supports_n": False,
        "cost_per_1k_input": 0.0002,
        "cost_per_1k_output": 0.0002,
    },
    "mistral-api-placeholder": {
        "provider": "mistral",
        "model_id": "mistral-small-latest",
        "max_tokens": 80,
        "description": "Mistral Small placeholder",
        "supports_logprobs": False,
        "supports_n": False,
        "cost_per_1k_input": 0.0002,
        "cost_per_1k_output": 0.0006,
    },
}


# ===========================================================================
# SECTION 2: Extended Algorithm Family Configurations
# ===========================================================================

ALGORITHM_FAMILY_CONFIGS = {
    "ancestral": {
        "description": "Pure ancestral (multinomial) sampling from the full distribution",
        "family": "ancestral",
        "configs": [
            {"name": "ancestral_t0.5", "temperature": 0.5, "top_p": 1.0, "top_k": 0},
            {"name": "ancestral_t0.7", "temperature": 0.7, "top_p": 1.0, "top_k": 0},
            {"name": "ancestral_t1.0", "temperature": 1.0, "top_p": 1.0, "top_k": 0},
            {"name": "ancestral_t1.2", "temperature": 1.2, "top_p": 1.0, "top_k": 0},
            {"name": "ancestral_t1.5", "temperature": 1.5, "top_p": 1.0, "top_k": 0},
        ],
    },
    "temperature": {
        "description": "Temperature scaling applied before sampling",
        "family": "temperature",
        "configs": [
            {"name": "temp_0.0", "temperature": 0.0, "top_p": 1.0, "top_k": 0},
            {"name": "temp_0.1", "temperature": 0.1, "top_p": 1.0, "top_k": 0},
            {"name": "temp_0.3", "temperature": 0.3, "top_p": 1.0, "top_k": 0},
            {"name": "temp_0.5", "temperature": 0.5, "top_p": 1.0, "top_k": 0},
            {"name": "temp_0.7", "temperature": 0.7, "top_p": 1.0, "top_k": 0},
            {"name": "temp_1.0", "temperature": 1.0, "top_p": 1.0, "top_k": 0},
            {"name": "temp_1.2", "temperature": 1.2, "top_p": 1.0, "top_k": 0},
            {"name": "temp_1.5", "temperature": 1.5, "top_p": 1.0, "top_k": 0},
            {"name": "temp_2.0", "temperature": 2.0, "top_p": 1.0, "top_k": 0},
        ],
    },
    "nucleus": {
        "description": "Nucleus (top-p) sampling — truncate tail of distribution",
        "family": "nucleus",
        "configs": [
            {"name": "nucleus_0.1", "temperature": 1.0, "top_p": 0.1, "top_k": 0},
            {"name": "nucleus_0.2", "temperature": 1.0, "top_p": 0.2, "top_k": 0},
            {"name": "nucleus_0.3", "temperature": 1.0, "top_p": 0.3, "top_k": 0},
            {"name": "nucleus_0.5", "temperature": 1.0, "top_p": 0.5, "top_k": 0},
            {"name": "nucleus_0.7", "temperature": 1.0, "top_p": 0.7, "top_k": 0},
            {"name": "nucleus_0.8", "temperature": 1.0, "top_p": 0.8, "top_k": 0},
            {"name": "nucleus_0.9", "temperature": 1.0, "top_p": 0.9, "top_k": 0},
            {"name": "nucleus_0.95", "temperature": 1.0, "top_p": 0.95, "top_k": 0},
            {"name": "nucleus_0.99", "temperature": 1.0, "top_p": 0.99, "top_k": 0},
        ],
    },
    "top_k": {
        "description": "Top-k sampling — keep only top k tokens at each step",
        "family": "top_k",
        "configs": [
            {"name": "topk_1", "temperature": 1.0, "top_p": 1.0, "top_k": 1},
            {"name": "topk_5", "temperature": 1.0, "top_p": 1.0, "top_k": 5},
            {"name": "topk_10", "temperature": 1.0, "top_p": 1.0, "top_k": 10},
            {"name": "topk_20", "temperature": 1.0, "top_p": 1.0, "top_k": 20},
            {"name": "topk_40", "temperature": 1.0, "top_p": 1.0, "top_k": 40},
            {"name": "topk_80", "temperature": 1.0, "top_p": 1.0, "top_k": 80},
            {"name": "topk_100", "temperature": 1.0, "top_p": 1.0, "top_k": 100},
            {"name": "topk_200", "temperature": 1.0, "top_p": 1.0, "top_k": 200},
            {"name": "topk_500", "temperature": 1.0, "top_p": 1.0, "top_k": 500},
        ],
    },
    "typical": {
        "description": "Typical sampling — sample tokens near expected information content",
        "family": "typical",
        "configs": [
            {"name": "typical_0.2", "temperature": 1.0, "typical_p": 0.2},
            {"name": "typical_0.5", "temperature": 1.0, "typical_p": 0.5},
            {"name": "typical_0.7", "temperature": 1.0, "typical_p": 0.7},
            {"name": "typical_0.9", "temperature": 1.0, "typical_p": 0.9},
            {"name": "typical_0.95", "temperature": 1.0, "typical_p": 0.95},
        ],
    },
    "contrastive": {
        "description": "Contrastive decoding — penalize degenerate model behavior",
        "family": "contrastive",
        "configs": [
            {"name": "contrastive_a0.1", "alpha": 0.1, "temperature": 1.0},
            {"name": "contrastive_a0.3", "alpha": 0.3, "temperature": 1.0},
            {"name": "contrastive_a0.5", "alpha": 0.5, "temperature": 1.0},
            {"name": "contrastive_a0.7", "alpha": 0.7, "temperature": 1.0},
            {"name": "contrastive_a1.0", "alpha": 1.0, "temperature": 1.0},
        ],
    },
    "qd_beam_search": {
        "description": "Quality-Diversity beam search — diverse beam search with QD objective",
        "family": "qd_beam_search",
        "configs": [
            {"name": "qd_beam_b5_g2", "beam_width": 5, "num_groups": 2, "diversity_penalty": 0.5},
            {"name": "qd_beam_b5_g5", "beam_width": 5, "num_groups": 5, "diversity_penalty": 0.5},
            {"name": "qd_beam_b10_g5", "beam_width": 10, "num_groups": 5, "diversity_penalty": 0.5},
            {"name": "qd_beam_b10_g5_d1", "beam_width": 10, "num_groups": 5, "diversity_penalty": 1.0},
            {"name": "qd_beam_b20_g10", "beam_width": 20, "num_groups": 10, "diversity_penalty": 0.5},
            {"name": "qd_beam_b20_g10_d2", "beam_width": 20, "num_groups": 10, "diversity_penalty": 2.0},
        ],
    },
    "stein_variational": {
        "description": "Stein variational inference for diverse text generation",
        "family": "stein_variational",
        "configs": [
            {"name": "stein_n5_k0.1", "n_particles": 5, "kernel_bandwidth": 0.1, "n_steps": 10},
            {"name": "stein_n5_k0.5", "n_particles": 5, "kernel_bandwidth": 0.5, "n_steps": 10},
            {"name": "stein_n10_k0.1", "n_particles": 10, "kernel_bandwidth": 0.1, "n_steps": 20},
            {"name": "stein_n10_k0.5", "n_particles": 10, "kernel_bandwidth": 0.5, "n_steps": 20},
            {"name": "stein_n10_k1.0", "n_particles": 10, "kernel_bandwidth": 1.0, "n_steps": 20},
            {"name": "stein_n20_k0.5", "n_particles": 20, "kernel_bandwidth": 0.5, "n_steps": 30},
        ],
    },
    "mcts": {
        "description": "Monte Carlo Tree Search for diverse sequence generation",
        "family": "mcts",
        "configs": [
            {"name": "mcts_s100_c1", "n_simulations": 100, "c_puct": 1.0, "temperature": 1.0},
            {"name": "mcts_s100_c2", "n_simulations": 100, "c_puct": 2.0, "temperature": 1.0},
            {"name": "mcts_s500_c1", "n_simulations": 500, "c_puct": 1.0, "temperature": 1.0},
            {"name": "mcts_s500_c2", "n_simulations": 500, "c_puct": 2.0, "temperature": 1.0},
            {"name": "mcts_s1000_c1", "n_simulations": 1000, "c_puct": 1.0, "temperature": 0.5},
            {"name": "mcts_s1000_c2", "n_simulations": 1000, "c_puct": 2.0, "temperature": 0.5},
        ],
    },
    "dpp": {
        "description": "Determinantal Point Process for diverse subset selection",
        "family": "dpp",
        "configs": [
            {"name": "dpp_k5_rbf", "k": 5, "kernel": "rbf", "sigma": 0.5},
            {"name": "dpp_k5_cosine", "k": 5, "kernel": "cosine", "sigma": 1.0},
            {"name": "dpp_k10_rbf", "k": 10, "kernel": "rbf", "sigma": 0.5},
            {"name": "dpp_k10_cosine", "k": 10, "kernel": "cosine", "sigma": 1.0},
            {"name": "dpp_k10_rbf_s1", "k": 10, "kernel": "rbf", "sigma": 1.0},
            {"name": "dpp_k20_rbf", "k": 20, "kernel": "rbf", "sigma": 0.5},
            {"name": "dpp_k20_cosine", "k": 20, "kernel": "cosine", "sigma": 1.0},
        ],
    },
    "eta_sampling": {
        "description": "Eta sampling — entropy-aware truncation for sampling",
        "family": "eta_sampling",
        "configs": [
            {"name": "eta_0.001", "eta": 0.001, "temperature": 1.0},
            {"name": "eta_0.01", "eta": 0.01, "temperature": 1.0},
            {"name": "eta_0.1", "eta": 0.1, "temperature": 1.0},
            {"name": "eta_0.3", "eta": 0.3, "temperature": 1.0},
        ],
    },
    "mirostat": {
        "description": "Mirostat — adaptive perplexity-targeting decoding",
        "family": "mirostat",
        "configs": [
            {"name": "mirostat_v2_t3", "version": 2, "target_perplexity": 3.0, "learning_rate": 0.1},
            {"name": "mirostat_v2_t5", "version": 2, "target_perplexity": 5.0, "learning_rate": 0.1},
            {"name": "mirostat_v2_t8", "version": 2, "target_perplexity": 8.0, "learning_rate": 0.1},
            {"name": "mirostat_v2_t12", "version": 2, "target_perplexity": 12.0, "learning_rate": 0.1},
        ],
    },
}

ALL_ALGORITHM_CONFIGS = []
for family_name, family_data in ALGORITHM_FAMILY_CONFIGS.items():
    for cfg in family_data["configs"]:
        ALL_ALGORITHM_CONFIGS.append({
            "family": family_name,
            "description": family_data["description"],
            **cfg,
        })

# ===========================================================================
# SECTION 3: Extended Prompts — 6+ domains, 50+ prompts per domain
# ===========================================================================

EXTENDED_PROMPTS_BY_DOMAIN = {
    "creative_writing": [
        "Once upon a time in a distant kingdom, there lived",
        "The rain fell softly on the old stone bridge as",
        "She opened the letter and read the first line:",
        "In the year 3000, humanity finally discovered",
        "The detective examined the crime scene and noticed",
        "Under the pale moonlight, the garden came alive with",
        "He had been walking for days when he finally reached",
        "The old woman sat by the fire and began to tell",
        "The ship sailed into the harbor just as the sun began to",
        "A strange noise echoed through the abandoned warehouse and",
        "The forest was thick with fog, but she pressed on because",
        "Two strangers met at the crossroads and immediately",
        "The dragon circled overhead, its shadow sweeping across",
        "In a small village at the edge of the desert, there was",
        "The clock struck midnight and the paintings began to",
        "He found a key hidden inside the old book, which unlocked",
        "The lighthouse keeper noticed something strange on the horizon:",
        "After the earthquake, the underground cavern revealed",
        "She whispered a secret that changed everything about",
        "The last train of the night carried only one passenger who",
        "Behind the waterfall, there was a hidden passage that led to",
        "The magician pulled from his hat not a rabbit but",
        "Three sisters inherited a mansion that was said to be",
        "The message in the bottle had been floating for centuries and said",
        "At the top of the mountain, the temple glowed with",
        "The robot opened its eyes for the first time and saw",
        "Deep beneath the ocean, a civilization thrived that had never",
        "The painting in the gallery seemed to watch visitors and",
        "A door appeared in the middle of the field, leading to",
        "The chef discovered an ancient recipe that could",
        "Stars fell from the sky one evening, each one carrying",
        "The mirror showed not a reflection but a window into",
        "On the eve of the great battle, the commander addressed",
        "The violin began to play by itself, filling the room with",
        "She traveled back in time to find that history was",
        "The garden gnome came to life at dusk and immediately",
        "An invisible thread connected two people across continents, and when",
        "The old map showed a route through mountains that no longer",
        "Thunder rumbled in the distance as the village prepared for",
        "The alchemist mixed the final ingredient, and the potion began to",
        "In the library of unwritten books, each volume contained",
        "The astronaut opened the airlock to find the planet covered in",
        "A single feather floated down from the sky, glowing with",
        "The musician played a chord that opened a portal to",
        "Beneath the castle, a network of tunnels connected",
        "The oracle spoke in riddles, her words revealing",
        "A child found a coin that granted not wishes but",
        "The world's last sunset painted the sky in colors never",
        "Two rivals discovered they shared the same dream about",
        "The phantom ship appeared once every century, carrying",
    ],
    "code_generation": [
        "def fibonacci(n):\n    # Return the nth Fibonacci number\n",
        "# Python function to sort a list using merge sort\ndef merge_sort(",
        "class DatabaseConnection:\n    def __init__(self",
        "async def fetch_data(url: str) -> dict:\n    # Fetch JSON data from",
        "def binary_search(arr, target):\n    # Implement binary search\n",
        "class LRUCache:\n    def __init__(self, capacity: int):\n",
        "def parse_csv(filepath: str) -> list:\n    # Parse CSV file and return rows\n",
        "def validate_email(email: str) -> bool:\n    # Check if email is valid\n",
        "class BinaryTree:\n    def insert(self, value):\n        # Insert a value into the BST\n",
        "def quicksort(arr: list) -> list:\n    # Implement quicksort algorithm\n",
        "class MinHeap:\n    def __init__(self):\n        self.heap = []\n",
        "def depth_first_search(graph, start):\n    # DFS traversal\n",
        "def breadth_first_search(graph, start):\n    # BFS traversal\n",
        "class LinkedList:\n    def reverse(self):\n        # Reverse the linked list in-place\n",
        "def matrix_multiply(a, b):\n    # Multiply two matrices\n",
        "class TrieNode:\n    def __init__(self):\n        # Initialize trie node\n",
        "def dijkstra(graph, source):\n    # Shortest path using Dijkstra's algorithm\n",
        "class HashTable:\n    def __init__(self, size=256):\n        # Initialize hash table with chaining\n",
        "async def rate_limiter(max_calls: int, period: float):\n    # Implement a rate limiter\n",
        "def topological_sort(graph):\n    # Topological sort for DAG\n",
        "class BloomFilter:\n    def __init__(self, size, num_hashes):\n",
        "def knapsack(weights, values, capacity):\n    # 0/1 Knapsack problem\n",
        "class AVLTree:\n    def _rotate_left(self, node):\n",
        "def longest_common_subsequence(s1, s2):\n    # Find LCS of two strings\n",
        "class ThreadPool:\n    def __init__(self, num_workers: int):\n",
        "def huffman_encoding(text: str) -> tuple:\n    # Build Huffman tree and encode\n",
        "class Graph:\n    def find_bridges(self):\n        # Find all bridges in undirected graph\n",
        "def regex_match(pattern: str, text: str) -> bool:\n    # Simple regex matcher\n",
        "class SkipList:\n    def __init__(self, max_level=16):\n",
        "def convex_hull(points):\n    # Graham scan for convex hull\n",
        "class BPlusTree:\n    def __init__(self, order=4):\n        # Initialize B+ tree\n",
        "def edit_distance(s1: str, s2: str) -> int:\n    # Levenshtein distance\n",
        "class DisjointSet:\n    def __init__(self, n):\n        # Union-Find with path compression\n",
        "def a_star_search(grid, start, goal):\n    # A* pathfinding algorithm\n",
        "class PriorityQueue:\n    def decrease_key(self, item, new_priority):\n",
        "def json_parser(s: str) -> dict:\n    # Parse a JSON string manually\n",
        "class EventEmitter:\n    def __init__(self):\n        # Pub/sub event system\n",
        "def reservoir_sampling(stream, k):\n    # Select k random items from a stream\n",
        "class CircularBuffer:\n    def __init__(self, capacity: int):\n",
        "def generate_permutations(arr):\n    # Generate all permutations without itertools\n",
        "class RedBlackTree:\n    def _fix_insert(self, node):\n",
        "def kmp_search(text: str, pattern: str) -> list:\n    # KMP string matching\n",
        "class SegmentTree:\n    def __init__(self, data):\n        # Range query segment tree\n",
        "def flood_fill(image, sr, sc, new_color):\n    # Flood fill algorithm\n",
        "class LFUCache:\n    def __init__(self, capacity: int):\n",
        "def longest_increasing_subsequence(arr):\n    # Find LIS with O(n log n)\n",
        "class Trie:\n    def autocomplete(self, prefix: str) -> list:\n",
        "def tarjan_scc(graph):\n    # Find strongly connected components\n",
        "def rabin_karp(text: str, pattern: str) -> list:\n    # Rabin-Karp string matching\n",
        "class MinStack:\n    def __init__(self):\n        # Stack that supports O(1) min queries\n",
        "def counting_sort(arr, max_val):\n    # Counting sort implementation\n",
    ],
    "scientific": [
        "The relationship between quantum entanglement and",
        "Recent advances in CRISPR gene editing have shown that",
        "The standard model of particle physics predicts that",
        "Climate models suggest that by 2050, global temperatures will",
        "The discovery of gravitational waves confirmed that",
        "Photosynthesis in plants converts light energy into",
        "The human immune system responds to viral infections by",
        "Neural plasticity allows the brain to reorganize by",
        "Dark matter constitutes approximately 27% of the universe and",
        "The theory of general relativity describes gravity as",
        "Antibiotic resistance develops when bacteria evolve mechanisms to",
        "Quantum computing uses qubits which differ from classical bits because",
        "The Higgs boson was discovered in 2012, confirming that",
        "Stem cells have the unique ability to differentiate into",
        "The second law of thermodynamics states that entropy in",
        "Black holes form when massive stars collapse and their",
        "DNA replication involves unwinding the double helix and",
        "The greenhouse effect occurs when certain gases in the atmosphere",
        "Superconductivity emerges at low temperatures because",
        "The theory of evolution explains biodiversity through",
        "Plate tectonics drives continental drift because the Earth's mantle",
        "Epigenetics studies heritable changes that do not involve",
        "The Hubble constant measures the rate at which",
        "Mitochondria are often called the powerhouse of the cell because",
        "String theory proposes that fundamental particles are actually",
        "The dopamine reward system in the brain influences behavior by",
        "Ocean acidification results from increased CO2 absorption and",
        "Protein folding determines function because the three-dimensional structure",
        "The uncertainty principle in quantum mechanics states that",
        "Neurogenesis in adults was long thought impossible, but research shows",
        "Catalysts speed up chemical reactions without being consumed by",
        "The cosmic microwave background radiation provides evidence that",
        "Circadian rhythms are regulated by molecular clocks that",
        "The double-slit experiment demonstrates that particles exhibit",
        "Entropy in information theory measures the average information content of",
        "Symbiotic relationships in ecosystems can be mutualistic when",
        "The RNA world hypothesis suggests that early life relied on",
        "Fusion reactions in the sun convert hydrogen into helium by",
        "Prion diseases are caused by misfolded proteins that",
        "The Cambrian explosion approximately 540 million years ago saw",
        "Telomeres protect chromosome ends during replication and their shortening",
        "Bose-Einstein condensates form when atoms are cooled to",
        "The microbiome in the human gut contains trillions of bacteria that",
        "Chaos theory describes systems where small changes in initial conditions",
        "The electromagnetic spectrum ranges from radio waves to gamma rays and",
        "Optogenetics allows researchers to control neurons using light by",
        "The anthropic principle in cosmology observes that physical constants",
        "Graph neural networks extend deep learning to non-Euclidean data by",
        "Transformer architectures have revolutionized NLP through the mechanism of",
        "The P vs NP problem asks whether every problem whose solution can be",
    ],
    "business": [
        "The key to a successful startup strategy is",
        "In today's competitive market, companies must focus on",
        "The quarterly earnings report showed that revenue increased by",
        "To improve customer retention, the most effective approach is",
        "Supply chain disruptions have forced companies to",
        "The merger between the two companies will create",
        "Digital transformation requires organizations to",
        "Sustainable business practices can improve profitability by",
        "The venture capital landscape in 2024 has shifted toward",
        "Remote work policies have fundamentally changed how companies approach",
        "Customer acquisition cost can be reduced by implementing",
        "The board of directors decided to pivot the company's strategy toward",
        "Market segmentation analysis revealed three underserved segments that",
        "The company's competitive moat is built on proprietary technology that",
        "Agile methodology has transformed product development by emphasizing",
        "The annual strategic planning process should incorporate scenario analysis for",
        "Brand loyalty among younger consumers is driven primarily by",
        "Cross-functional teams improve innovation because they bring together",
        "The total addressable market for this product category is estimated at",
        "Subscription-based business models provide recurring revenue streams that",
        "Due diligence in mergers and acquisitions should examine",
        "The company needs to optimize its working capital by",
        "Intellectual property strategy is crucial for startups because",
        "ESG reporting has become mandatory for companies that",
        "The go-to-market strategy for the new product focuses on",
        "Stakeholder management requires balancing the interests of",
        "The return on investment for training programs can be measured by",
        "Pricing strategy should account for elasticity of demand and",
        "The chief technology officer presented a roadmap that includes",
        "Risk management frameworks help organizations identify and mitigate",
        "The sales pipeline shows a conversion rate of 12% which indicates",
        "Operational efficiency can be improved through lean management principles that",
        "The company's valuation is based on a discounted cash flow model that",
        "Employee engagement surveys revealed that the top concern is",
        "The marketing funnel needs optimization at the consideration stage where",
        "International expansion requires adapting to local regulations including",
        "Data-driven decision making has replaced intuition-based management in",
        "The platform economy has disrupted traditional industries by",
        "Corporate governance reforms aim to increase transparency by",
        "The startup's burn rate suggests they have runway for",
        "Customer lifetime value analysis shows that the most profitable segment",
        "The supply chain resilience framework includes diversifying suppliers and",
        "Product-market fit can be validated through metrics such as",
        "The company's net promoter score of 72 suggests that",
        "Strategic partnerships accelerate growth by providing access to",
        "The quarterly business review highlighted three key performance indicators:",
        "Change management frameworks help organizations navigate transformation by",
        "The competitive analysis matrix shows strengths in technology but gaps in",
        "Revenue diversification reduces risk by ensuring no single product exceeds",
        "The organizational restructuring plan aims to flatten the hierarchy by",
    ],
    "dialogue": [
        "Customer: I'd like to return this product.\nAgent:",
        "Student: Can you explain how neural networks work?\nTeacher:",
        "User: What's the best way to learn Python?\nAssistant:",
        "Interviewer: Tell me about your biggest achievement.\nCandidate:",
        "Patient: I've been having trouble sleeping lately.\nDoctor:",
        "Manager: We need to discuss the project timeline.\nDeveloper:",
        "Tourist: Can you recommend a good restaurant nearby?\nLocal:",
        "Parent: My child is struggling with math.\nTutor:",
        "Reporter: What inspired your latest research?\nScientist:",
        "Caller: I think there's a gas leak in my apartment.\nOperator:",
        "Client: I need help planning my retirement investments.\nAdvisor:",
        "Friend: I'm feeling really overwhelmed with work lately.\nFriend:",
        "Guest: What's the story behind this dish?\nChef:",
        "Passenger: How long is the delay expected to last?\nPilot:",
        "Neighbor: Your music has been really loud lately.\nResident:",
        "New Employee: How does the code review process work here?\nMentor:",
        "Customer: This is the third time my order was wrong.\nManager:",
        "Student: I don't understand why we need to learn calculus.\nProfessor:",
        "Author: I'm struggling with writer's block on my novel.\nEditor:",
        "Tenant: The heating has been broken for two weeks now.\nLandlord:",
        "Athlete: I've hit a plateau in my training.\nCoach:",
        "Patient: What are the side effects of this medication?\nPharmacist:",
        "Buyer: Can you go any lower on the price?\nSeller:",
        "Citizen: I'd like to report a pothole on Main Street.\nOfficial:",
        "Developer: The build keeps failing on this test.\nSenior Dev:",
        "Student: How do I choose between grad school and industry?\nAdvisor:",
        "Customer: I was charged twice for the same transaction.\nBank Rep:",
        "Parent: At what age should kids get their first phone?\nPediatrician:",
        "Employee: I'd like to negotiate a raise.\nHR Manager:",
        "Hiker: We seem to be lost. Which trail leads back?\nPark Ranger:",
        "Host: Welcome to the show. What's your take on AI?\nGuest Expert:",
        "Rookie: Any advice for my first day on the job?\nVeteran:",
        "Jury Member: I'm not sure I understand reasonable doubt.\nJudge:",
        "Gamer: I keep dying at this boss fight.\nStreamer:",
        "Homeowner: Should I refinance my mortgage now?\nLoan Officer:",
        "Patient: I'm nervous about the surgery tomorrow.\nSurgeon:",
        "Driver: I was pulled over but I don't know why.\nOfficer:",
        "Student: How do I debug a segmentation fault?\nTA:",
        "Tourist: What cultural customs should I be aware of?\nGuide:",
        "Client: My website loads very slowly.\nDeveloper:",
        "Musician: I want to switch from guitar to piano.\nTeacher:",
        "Voter: How do I verify my registration status?\nElection Worker:",
        "Researcher: My paper got rejected. What should I do?\nAdvisor:",
        "Child: Why is the sky blue?\nParent:",
        "Entrepreneur: How do I protect my business idea?\nLawyer:",
        "Witness: I'm afraid to testify.\nProsecutor:",
        "User: My computer keeps crashing randomly.\nTech Support:",
        "Intern: How can I make a good impression?\nManager:",
        "Patient: How long will physical therapy take?\nTherapist:",
        "Customer: Do you have this in a different size?\nSales Associate:",
    ],
    "mathematical_reasoning": [
        "To prove that the square root of 2 is irrational, we assume",
        "The fundamental theorem of calculus connects differentiation and integration by",
        "A group in abstract algebra must satisfy four axioms:",
        "The pigeonhole principle states that if n items are placed into m containers and n > m, then",
        "Euler's identity e^(iπ) + 1 = 0 connects five fundamental constants because",
        "The Riemann hypothesis concerns the distribution of prime numbers and states that",
        "A topological space is Hausdorff if for any two distinct points",
        "The central limit theorem explains why the normal distribution appears so frequently:",
        "Gödel's incompleteness theorems show that any sufficiently powerful formal system",
        "The Fourier transform decomposes a function into its frequency components by",
        "A vector space must be closed under both addition and scalar multiplication, meaning",
        "The Cantor diagonal argument proves that the real numbers are uncountable by",
        "Lagrange multipliers find the extrema of a function subject to constraints by",
        "The Cauchy-Schwarz inequality states that for any vectors u and v,",
        "A metric space is complete if every Cauchy sequence in the space",
        "The Taylor series approximates a function near a point by expressing it as",
        "Fermat's last theorem states that no three positive integers satisfy",
        "The eigenvalues of a matrix reveal important properties including",
        "A bijection between two sets proves they have the same cardinality because",
        "The method of mathematical induction works by first proving the base case, then",
        "Bayes' theorem relates conditional probabilities through the formula",
        "A ring is a set equipped with two binary operations that generalize",
        "The divergence theorem relates a flux integral over a closed surface to",
        "Zorn's lemma is equivalent to the axiom of choice and states that",
        "The Jordan normal form of a matrix decomposes it into blocks that",
        "A Banach space is a complete normed vector space, which means",
        "The Bolzano-Weierstrass theorem guarantees that every bounded sequence in R^n",
        "The Chinese remainder theorem provides a solution for systems of congruences when",
        "Lebesgue integration extends the Riemann integral by measuring",
        "The compactness theorem in logic states that a set of sentences has a model if",
        "Stochastic processes generalize deterministic dynamics by introducing",
        "The simplex method for linear programming works by moving along",
        "A manifold locally resembles Euclidean space, which means that near any point",
        "The Hahn-Banach theorem allows extending bounded linear functionals from",
        "Galois theory connects field extensions to group theory by associating",
        "The spectral theorem for self-adjoint operators guarantees that",
        "A martingale is a sequence of random variables satisfying the property that",
        "The Poincaré conjecture, proven by Perelman, states that every simply connected",
        "Markov chain Monte Carlo methods sample from probability distributions by",
        "The Stone-Weierstrass theorem states that continuous functions on a compact set can be",
        "An abelian group has the property that the group operation is",
        "The fixed-point theorem of Banach guarantees that a contraction mapping on",
        "Category theory provides a unifying framework for mathematics by studying",
        "The law of large numbers states that the sample mean converges to",
        "A Hilbert space generalizes Euclidean space to infinite dimensions by",
        "The residue theorem in complex analysis evaluates contour integrals by",
        "Ramsey theory studies conditions under which order must appear in",
        "The Yoneda lemma in category theory states that an object is determined by",
        "A p-adic number system extends the rationals using a different notion of",
        "The ergodic theorem relates time averages to space averages for",
    ],
    "instruction_following": [
        "Write a haiku about the ocean at sunset.",
        "List exactly five advantages of renewable energy, numbered 1-5.",
        "Explain photosynthesis to a 10-year-old in three sentences.",
        "Compare and contrast democracy and authoritarianism in a balanced paragraph.",
        "Write a formal email declining a job offer politely.",
        "Describe the water cycle using only words that start with the letter S.",
        "Create a recipe for chocolate chip cookies with metric measurements.",
        "Summarize the plot of Romeo and Juliet in exactly 50 words.",
        "Write instructions for tying a bowline knot in five steps.",
        "Explain the concept of inflation as if speaking to a medieval farmer.",
        "Write a limerick about a programmer debugging code.",
        "List the planets in our solar system from largest to smallest.",
        "Describe the smell of rain using three different metaphors.",
        "Write a product description for a smart water bottle.",
        "Explain machine learning without using the words 'data' or 'algorithm'.",
        "Create a workout routine for beginners with exactly four exercises.",
        "Write a two-sentence horror story set in a library.",
        "Describe the taste of coffee to someone who has never tried it.",
        "Write a persuasive paragraph about why everyone should learn to cook.",
        "Explain blockchain technology using a pizza delivery analogy.",
        "List seven tips for improving public speaking skills.",
        "Write a weather forecast for a fictional planet with two suns.",
        "Describe the process of making paper from trees in reverse order.",
        "Create a short dialogue between a cat and a dog discussing philosophy.",
        "Explain the difference between a virus and a bacterium in simple terms.",
        "Write a motivational quote about perseverance and explain its meaning.",
        "Describe a rainbow to someone who has never seen colors.",
        "Create a timeline of five major events in computer history.",
        "Write a user manual for a time machine in three paragraphs.",
        "Explain why the sky is blue without using scientific jargon.",
        "List five common logical fallacies with brief examples.",
        "Write a news headline and opening paragraph about a discovery on Mars.",
        "Describe the feeling of nostalgia using sensory details.",
        "Create a menu for a restaurant that serves only foods from the year 1900.",
        "Explain the concept of opportunity cost with a everyday example.",
        "Write a short fairy tale with exactly three characters.",
        "Describe the sound of a thunderstorm from inside a tent.",
        "List the steps to change a flat tire in numbered order.",
        "Write an apology letter from a robot to its human owner.",
        "Explain the scientific method in exactly six steps.",
        "Create a travel itinerary for a weekend in a city you've never visited.",
        "Describe gravity to an alien who has never experienced it.",
        "Write a song chorus about friendship in four lines.",
        "Explain why exercise is important without mentioning health or weight.",
        "Create a quiz with five multiple-choice questions about world geography.",
        "Write a thank-you note for a gift you didn't actually want.",
        "Describe what silence sounds like using at least three senses.",
        "Explain compound interest using a snowball rolling down a hill analogy.",
        "Write instructions for an alien on how to make a peanut butter sandwich.",
        "Create a list of pros and cons for living on the moon.",
    ],
}

EXTENDED_ALL_PROMPTS = [
    (domain, prompt)
    for domain, prompts in EXTENDED_PROMPTS_BY_DOMAIN.items()
    for prompt in prompts
]


# ===========================================================================
# SECTION 4: Statistical Analysis Functions
# ===========================================================================

def bootstrap_confidence_intervals(
    data: np.ndarray,
    statistic_fn=np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    use_bca: bool = True,
    random_state: int = 42,
) -> Dict[str, float]:
    """Bootstrap CIs on any statistic with optional BCa correction.

    Parameters
    ----------
    data : array-like of observations
    statistic_fn : callable applied to bootstrap samples
    n_bootstrap : number of bootstrap resamples
    confidence_level : desired CI level (e.g. 0.95 for 95%)
    use_bca : if True, apply BCa (bias-corrected and accelerated) correction
    random_state : seed for reproducibility

    Returns
    -------
    dict with keys: point_estimate, ci_lower, ci_upper, se, method
    """
    from scipy.stats import norm as scipy_norm

    rng = np.random.RandomState(random_state)
    data = np.asarray(data, dtype=float)
    n = len(data)

    if n < 2:
        val = float(statistic_fn(data)) if n == 1 else 0.0
        return {
            "point_estimate": val,
            "ci_lower": val,
            "ci_upper": val,
            "se": 0.0,
            "method": "degenerate",
        }

    point_est = float(statistic_fn(data))

    # Draw bootstrap resamples
    boot_indices = rng.randint(0, n, size=(n_bootstrap, n))
    boot_stats = np.array([float(statistic_fn(data[idx])) for idx in boot_indices])

    se = float(np.std(boot_stats, ddof=1))
    alpha = 1 - confidence_level

    if not use_bca:
        ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        return {
            "point_estimate": point_est,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "se": se,
            "method": "percentile",
        }

    # BCa correction
    # Bias correction factor z0
    prop_below = np.mean(boot_stats < point_est)
    prop_below = np.clip(prop_below, 1e-10, 1.0 - 1e-10)
    z0 = scipy_norm.ppf(prop_below)

    # Acceleration factor a (jackknife)
    jackknife_stats = np.empty(n)
    for i in range(n):
        jack_sample = np.delete(data, i)
        jackknife_stats[i] = float(statistic_fn(jack_sample))
    jack_mean = np.mean(jackknife_stats)
    jack_diff = jack_mean - jackknife_stats
    numerator = np.sum(jack_diff ** 3)
    denominator = 6.0 * (np.sum(jack_diff ** 2) ** 1.5)
    a = numerator / denominator if abs(denominator) > 1e-15 else 0.0

    # Adjusted percentiles
    z_alpha_lower = scipy_norm.ppf(alpha / 2)
    z_alpha_upper = scipy_norm.ppf(1 - alpha / 2)

    def _adjusted_percentile(z_alpha):
        numer = z0 + z_alpha
        denom = 1.0 - a * numer
        if abs(denom) < 1e-15:
            return 0.5
        adjusted = scipy_norm.cdf(z0 + numer / denom)
        return np.clip(adjusted, 0.001, 0.999)

    p_lower = _adjusted_percentile(z_alpha_lower)
    p_upper = _adjusted_percentile(z_alpha_upper)

    ci_lower = float(np.percentile(boot_stats, 100 * p_lower))
    ci_upper = float(np.percentile(boot_stats, 100 * p_upper))

    return {
        "point_estimate": point_est,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
        "method": "bca",
    }


def bayesian_sign_tests(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    prior_strength: float = 1.0,
) -> Dict[str, Any]:
    """Bayesian sign test between two paired sets of scores.

    Uses a Dirichlet-Multinomial model with categories (A wins, tie, B wins).
    The prior is a symmetric Dirichlet with parameter `prior_strength`.

    Parameters
    ----------
    scores_a, scores_b : paired observations
    prior_strength : Dirichlet prior concentration (1.0 = uniform)

    Returns
    -------
    dict with probabilities that A > B, A == B, A < B, plus Bayes factor
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    assert len(scores_a) == len(scores_b), "Paired observations required"

    diffs = scores_a - scores_b
    n_a_wins = int(np.sum(diffs > 0))
    n_ties = int(np.sum(diffs == 0))
    n_b_wins = int(np.sum(diffs < 0))
    n_total = len(diffs)

    # Posterior Dirichlet parameters
    alpha_a = prior_strength + n_a_wins
    alpha_tie = prior_strength + n_ties
    alpha_b = prior_strength + n_b_wins
    alpha_sum = alpha_a + alpha_tie + alpha_b

    # Posterior means
    p_a_wins = alpha_a / alpha_sum
    p_tie = alpha_tie / alpha_sum
    p_b_wins = alpha_b / alpha_sum

    # Monte Carlo estimation of P(θ_A > θ_B)
    rng = np.random.RandomState(42)
    n_mc = 50000
    samples = rng.dirichlet([alpha_a, alpha_tie, alpha_b], size=n_mc)
    prob_a_better = float(np.mean(samples[:, 0] > samples[:, 2]))
    prob_b_better = float(np.mean(samples[:, 2] > samples[:, 0]))

    # Bayes factor for A > B vs B > A
    epsilon = 1e-12
    bayes_factor = (prob_a_better + epsilon) / (prob_b_better + epsilon)

    # Effect size (simple proportion difference)
    effect_size = (n_a_wins - n_b_wins) / max(n_total, 1)

    return {
        "n_a_wins": n_a_wins,
        "n_ties": n_ties,
        "n_b_wins": n_b_wins,
        "n_total": n_total,
        "p_a_wins_posterior": round(p_a_wins, 4),
        "p_tie_posterior": round(p_tie, 4),
        "p_b_wins_posterior": round(p_b_wins, 4),
        "prob_a_better": round(prob_a_better, 4),
        "prob_b_better": round(prob_b_better, 4),
        "bayes_factor_a_over_b": round(bayes_factor, 4),
        "effect_size": round(effect_size, 4),
    }


def bradley_terry_rankings(
    win_matrix: np.ndarray,
    names: List[str],
    n_iterations: int = 1000,
    tol: float = 1e-8,
    n_bootstrap: int = 500,
) -> Dict[str, Any]:
    """Compute Bradley-Terry model rankings with uncertainty via bootstrap.

    Parameters
    ----------
    win_matrix : (n, n) matrix where w[i,j] = number of times i beat j
    names : list of algorithm names
    n_iterations : max iterations for the MM algorithm
    tol : convergence tolerance
    n_bootstrap : number of bootstrap runs for uncertainty

    Returns
    -------
    dict with rankings, scores, uncertainty intervals
    """
    n = len(names)
    assert win_matrix.shape == (n, n)

    def _fit_bt(W):
        """Fit BT model using the MM (minorization-maximization) algorithm."""
        pi = np.ones(n) / n
        for _ in range(n_iterations):
            pi_old = pi.copy()
            for i in range(n):
                numerator = 0.0
                denominator = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    w_ij = W[i, j]
                    w_ji = W[j, i]
                    n_ij = w_ij + w_ji
                    if n_ij == 0:
                        continue
                    numerator += w_ij
                    denominator += n_ij / (pi[i] + pi[j])
                pi[i] = numerator / max(denominator, 1e-15)
            # Normalize
            pi = pi / np.sum(pi)
            if np.max(np.abs(pi - pi_old)) < tol:
                break
        return pi

    # Fit on full data
    scores = _fit_bt(win_matrix.astype(float))

    # Bootstrap for uncertainty
    rng = np.random.RandomState(42)
    boot_scores = np.zeros((n_bootstrap, n))
    total_matches = np.sum(win_matrix)

    for b in range(n_bootstrap):
        W_boot = np.zeros_like(win_matrix, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                n_ij = int(win_matrix[i, j] + win_matrix[j, i])
                if n_ij == 0:
                    continue
                p_i = win_matrix[i, j] / n_ij
                wins_i = rng.binomial(n_ij, p_i)
                W_boot[i, j] = wins_i
                W_boot[j, i] = n_ij - wins_i
        boot_scores[b] = _fit_bt(W_boot)

    # Compute CIs for each algorithm
    rankings = []
    for i in range(n):
        boot_vals = boot_scores[:, i]
        rankings.append({
            "name": names[i],
            "score": round(float(scores[i]), 6),
            "ci_lower": round(float(np.percentile(boot_vals, 2.5)), 6),
            "ci_upper": round(float(np.percentile(boot_vals, 97.5)), 6),
            "se": round(float(np.std(boot_vals)), 6),
        })

    rankings.sort(key=lambda x: x["score"], reverse=True)
    for rank_idx, r in enumerate(rankings):
        r["rank"] = rank_idx + 1

    # Probability that each algorithm is ranked first
    first_place_counts = np.zeros(n)
    for b in range(n_bootstrap):
        best_idx = np.argmax(boot_scores[b])
        first_place_counts[best_idx] += 1
    prob_first = first_place_counts / n_bootstrap

    return {
        "rankings": rankings,
        "prob_first_place": {names[i]: round(float(prob_first[i]), 4) for i in range(n)},
        "n_bootstrap": n_bootstrap,
        "converged": True,
    }


def pareto_analysis(
    points: np.ndarray,
    names: List[str],
    objectives: List[str],
    maximize: List[bool],
) -> Dict[str, Any]:
    """Multi-objective Pareto frontier computation.

    Parameters
    ----------
    points : (n, m) array of objective values
    names : labels for each point
    objectives : names of each objective dimension
    maximize : whether each objective should be maximized

    Returns
    -------
    dict with pareto_front indices, dominated points, hypervolume
    """
    n, m = points.shape
    assert len(names) == n
    assert len(objectives) == m
    assert len(maximize) == m

    # Normalize direction: convert everything to maximization
    normalized = points.copy()
    for j in range(m):
        if not maximize[j]:
            normalized[:, j] = -normalized[:, j]

    # Find Pareto-optimal points
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # Check if j dominates i
            if np.all(normalized[j] >= normalized[i]) and np.any(normalized[j] > normalized[i]):
                is_pareto[i] = False
                break

    pareto_indices = np.where(is_pareto)[0].tolist()
    dominated_indices = np.where(~is_pareto)[0].tolist()

    # Compute approximate hypervolume using the Pareto points
    # Reference point: slightly worse than the worst observed in each objective
    ref_point = np.min(normalized, axis=0) - 0.01 * (np.max(normalized, axis=0) - np.min(normalized, axis=0) + 1e-10)
    pareto_points = normalized[pareto_indices]

    # 2D hypervolume exact computation if m == 2
    hypervolume = 0.0
    if m == 2 and len(pareto_points) > 0:
        sorted_pts = pareto_points[np.argsort(-pareto_points[:, 0])]
        prev_y = ref_point[1]
        for pt in sorted_pts:
            if pt[1] > prev_y:
                hypervolume += (pt[0] - ref_point[0]) * (pt[1] - prev_y)
                prev_y = pt[1]
    elif m > 2 and len(pareto_points) > 0:
        # Monte Carlo hypervolume approximation for higher dimensions
        rng = np.random.RandomState(42)
        bounds_min = ref_point
        bounds_max = np.max(pareto_points, axis=0) + 1e-10
        volume_box = float(np.prod(bounds_max - bounds_min))
        n_mc_samples = 100000
        random_points = rng.uniform(bounds_min, bounds_max, size=(n_mc_samples, m))
        dominated_count = 0
        for rp in random_points:
            for pp in pareto_points:
                if np.all(pp >= rp):
                    dominated_count += 1
                    break
        hypervolume = volume_box * dominated_count / n_mc_samples

    # Crowding distance for Pareto points
    crowding = np.zeros(len(pareto_indices))
    if len(pareto_indices) >= 3:
        pp = normalized[pareto_indices]
        for obj_idx in range(m):
            sorted_order = np.argsort(pp[:, obj_idx])
            crowding[sorted_order[0]] = float("inf")
            crowding[sorted_order[-1]] = float("inf")
            obj_range = pp[sorted_order[-1], obj_idx] - pp[sorted_order[0], obj_idx]
            if obj_range < 1e-15:
                continue
            for k in range(1, len(sorted_order) - 1):
                gap = pp[sorted_order[k + 1], obj_idx] - pp[sorted_order[k - 1], obj_idx]
                crowding[sorted_order[k]] += gap / obj_range

    pareto_details = []
    for idx_in_pareto, global_idx in enumerate(pareto_indices):
        detail = {
            "name": names[global_idx],
            "index": global_idx,
            "values": {objectives[j]: round(float(points[global_idx, j]), 6) for j in range(m)},
            "crowding_distance": round(float(crowding[idx_in_pareto]), 6) if not np.isinf(crowding[idx_in_pareto]) else "inf",
        }
        pareto_details.append(detail)

    return {
        "n_points": n,
        "n_pareto": len(pareto_indices),
        "n_dominated": len(dominated_indices),
        "pareto_fraction": round(len(pareto_indices) / max(n, 1), 4),
        "pareto_front": pareto_details,
        "dominated_indices": dominated_indices,
        "hypervolume": round(hypervolume, 6),
        "objectives": objectives,
    }


def metric_correlation_analysis(
    metric_records: List[Dict],
    metric_names: List[str],
) -> Dict[str, Any]:
    """Compute correlation heatmap data across all metrics.

    Parameters
    ----------
    metric_records : list of dicts, each with metric values
    metric_names : list of metric keys to analyze

    Returns
    -------
    dict with pearson and spearman correlation matrices, plus significance
    """
    from scipy.stats import pearsonr, spearmanr

    n_metrics = len(metric_names)
    n_records = len(metric_records)

    # Build data matrix, filtering out records with missing values
    data_rows = []
    for rec in metric_records:
        row = []
        skip = False
        for m in metric_names:
            val = rec.get(m)
            if val is None:
                skip = True
                break
            row.append(float(val))
        if not skip:
            data_rows.append(row)

    if len(data_rows) < 3:
        return {"error": "Insufficient data for correlation analysis", "n_valid": len(data_rows)}

    data = np.array(data_rows)

    pearson_matrix = np.eye(n_metrics)
    pearson_pval = np.zeros((n_metrics, n_metrics))
    spearman_matrix = np.eye(n_metrics)
    spearman_pval = np.zeros((n_metrics, n_metrics))

    for i in range(n_metrics):
        for j in range(i + 1, n_metrics):
            r_p, p_p = pearsonr(data[:, i], data[:, j])
            r_s, p_s = spearmanr(data[:, i], data[:, j])
            pearson_matrix[i, j] = pearson_matrix[j, i] = r_p
            pearson_pval[i, j] = pearson_pval[j, i] = p_p
            spearman_matrix[i, j] = spearman_matrix[j, i] = r_s
            spearman_pval[i, j] = spearman_pval[j, i] = p_s

    # Identify clusters of highly correlated metrics (|r| > 0.8)
    redundancy_pairs = []
    for i in range(n_metrics):
        for j in range(i + 1, n_metrics):
            if abs(pearson_matrix[i, j]) > 0.8:
                redundancy_pairs.append({
                    "metric_a": metric_names[i],
                    "metric_b": metric_names[j],
                    "pearson_r": round(float(pearson_matrix[i, j]), 4),
                    "spearman_r": round(float(spearman_matrix[i, j]), 4),
                })

    # Determine which metrics are most independent (lowest avg correlation)
    avg_abs_corr = np.mean(np.abs(pearson_matrix), axis=1)
    independence_ranking = sorted(
        [(metric_names[i], round(float(avg_abs_corr[i]), 4)) for i in range(n_metrics)],
        key=lambda x: x[1],
    )

    return {
        "n_records_used": len(data_rows),
        "n_records_total": n_records,
        "pearson_correlation": {
            metric_names[i]: {
                metric_names[j]: round(float(pearson_matrix[i, j]), 4)
                for j in range(n_metrics)
            }
            for i in range(n_metrics)
        },
        "spearman_correlation": {
            metric_names[i]: {
                metric_names[j]: round(float(spearman_matrix[i, j]), 4)
                for j in range(n_metrics)
            }
            for i in range(n_metrics)
        },
        "pearson_pvalues": {
            metric_names[i]: {
                metric_names[j]: round(float(pearson_pval[i, j]), 6)
                for j in range(n_metrics)
            }
            for i in range(n_metrics)
        },
        "spearman_pvalues": {
            metric_names[i]: {
                metric_names[j]: round(float(spearman_pval[i, j]), 6)
                for j in range(n_metrics)
            }
            for i in range(n_metrics)
        },
        "redundancy_pairs": redundancy_pairs,
        "independence_ranking": independence_ranking,
    }


def convergence_analysis(
    metric_records: List[Dict],
    metric_names: List[str],
    sample_fractions: List[float] = None,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Analyze how metrics and rankings converge with increasing sample size.

    For each fraction of the data, compute metrics and measure stability of
    the resulting config ranking compared to the full-data ranking.

    Parameters
    ----------
    metric_records : list of result dicts
    metric_names : metric keys to track
    sample_fractions : fractions of data to test (default: 0.1 to 1.0)
    n_bootstrap : bootstrap repetitions per fraction
    random_state : seed

    Returns
    -------
    dict with convergence curves for each metric
    """
    if sample_fractions is None:
        sample_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    rng = np.random.RandomState(random_state)
    n_total = len(metric_records)

    if n_total < 10:
        return {"error": "Too few records for convergence analysis", "n_records": n_total}

    # Build full-data config means as reference
    from collections import defaultdict
    full_config_means = defaultdict(lambda: defaultdict(list))
    for rec in metric_records:
        cfg = rec.get("config", "unknown")
        for m in metric_names:
            val = rec.get(m)
            if val is not None:
                full_config_means[cfg][m].append(val)

    full_rankings = {}
    for m in metric_names:
        sorted_configs = sorted(
            full_config_means.keys(),
            key=lambda c: np.mean(full_config_means[c].get(m, [0])),
            reverse=True,
        )
        full_rankings[m] = sorted_configs

    convergence = {}
    for m in metric_names:
        fraction_data = []
        for frac in sample_fractions:
            n_sample = max(2, int(n_total * frac))
            tau_values = []
            mean_values = []

            for _ in range(min(n_bootstrap, 50 if frac < 0.3 else n_bootstrap)):
                sample_indices = rng.choice(n_total, size=n_sample, replace=False)
                sample_records = [metric_records[i] for i in sample_indices]

                # Compute config means from sample
                sample_config_means = defaultdict(list)
                for rec in sample_records:
                    cfg = rec.get("config", "unknown")
                    val = rec.get(m)
                    if val is not None:
                        sample_config_means[cfg].append(val)

                # Rank configs
                common_configs = [c for c in full_rankings[m] if c in sample_config_means]
                if len(common_configs) < 3:
                    continue

                full_rank = [full_rankings[m].index(c) for c in common_configs]
                sample_sorted = sorted(
                    common_configs,
                    key=lambda c: np.mean(sample_config_means[c]),
                    reverse=True,
                )
                sample_rank = [sample_sorted.index(c) for c in common_configs]

                tau, _ = kendalltau(full_rank, sample_rank)
                if not np.isnan(tau):
                    tau_values.append(tau)

                # Overall mean of the metric
                all_vals = [v for vals in sample_config_means.values() for v in vals]
                if all_vals:
                    mean_values.append(np.mean(all_vals))

            fraction_data.append({
                "fraction": frac,
                "n_sample": n_sample,
                "rank_stability_tau": round(float(np.mean(tau_values)), 4) if tau_values else None,
                "rank_stability_std": round(float(np.std(tau_values)), 4) if tau_values else None,
                "metric_mean": round(float(np.mean(mean_values)), 4) if mean_values else None,
                "metric_std": round(float(np.std(mean_values)), 4) if mean_values else None,
            })

        convergence[m] = fraction_data

    # Determine minimum sample size for stable rankings (tau > 0.9)
    min_samples = {}
    for m, fdata in convergence.items():
        stable_frac = None
        for fd in fdata:
            if fd["rank_stability_tau"] is not None and fd["rank_stability_tau"] > 0.9:
                stable_frac = fd["fraction"]
                break
        min_samples[m] = {
            "min_fraction_for_tau_0.9": stable_frac,
            "min_n_for_tau_0.9": int(n_total * stable_frac) if stable_frac else None,
        }

    return {
        "n_total_records": n_total,
        "sample_fractions": sample_fractions,
        "convergence_curves": convergence,
        "min_samples_for_stability": min_samples,
    }


# ===========================================================================
# SECTION 5: Result Aggregation and Reporting
# ===========================================================================

def aggregate_results_by_domain(
    metric_records: List[Dict],
    metric_names: List[str],
) -> Dict[str, Any]:
    """Aggregate metric results grouped by task domain.

    Computes per-domain means, standard deviations, and bootstrap CIs
    for each metric. Also computes cross-domain consistency.

    Returns
    -------
    dict mapping domain -> metric -> {mean, std, ci_lower, ci_upper, n}
    """
    from collections import defaultdict

    domain_data = defaultdict(lambda: defaultdict(list))
    for rec in metric_records:
        domain = rec.get("domain", "unknown")
        for m in metric_names:
            val = rec.get(m)
            if val is not None:
                domain_data[domain][m].append(val)

    aggregated = {}
    for domain, metrics in domain_data.items():
        aggregated[domain] = {}
        for m, values in metrics.items():
            vals = np.array(values)
            ci = bootstrap_confidence_intervals(vals, n_bootstrap=2000)
            aggregated[domain][m] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "median": round(float(np.median(vals)), 4),
                "ci_lower": ci["ci_lower"],
                "ci_upper": ci["ci_upper"],
                "n": len(vals),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

    # Cross-domain consistency: coefficient of variation of means
    consistency = {}
    for m in metric_names:
        domain_means = [
            aggregated[d][m]["mean"]
            for d in aggregated
            if m in aggregated[d]
        ]
        if len(domain_means) >= 2:
            cv = float(np.std(domain_means) / max(abs(np.mean(domain_means)), 1e-10))
            consistency[m] = {
                "coefficient_of_variation": round(cv, 4),
                "domain_means": {d: aggregated[d][m]["mean"] for d in aggregated if m in aggregated[d]},
                "is_stable": cv < 0.3,
            }

    return {
        "per_domain": aggregated,
        "cross_domain_consistency": consistency,
        "domains": list(aggregated.keys()),
        "n_domains": len(aggregated),
    }


def aggregate_results_by_algorithm(
    metric_records: List[Dict],
    metric_names: List[str],
) -> Dict[str, Any]:
    """Aggregate metric results grouped by algorithm config.

    Computes per-algorithm means, standard deviations, and identifies
    the best algorithm for each metric.

    Returns
    -------
    dict mapping config -> metric -> {mean, std, n, rank}
    """
    from collections import defaultdict

    algo_data = defaultdict(lambda: defaultdict(list))
    for rec in metric_records:
        config = rec.get("config", "unknown")
        for m in metric_names:
            val = rec.get(m)
            if val is not None:
                algo_data[config][m].append(val)

    aggregated = {}
    for config, metrics in algo_data.items():
        aggregated[config] = {}
        for m, values in metrics.items():
            vals = np.array(values)
            ci = bootstrap_confidence_intervals(vals, n_bootstrap=2000)
            aggregated[config][m] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "ci_lower": ci["ci_lower"],
                "ci_upper": ci["ci_upper"],
                "n": len(vals),
            }

    # Rank algorithms per metric
    rankings = {}
    for m in metric_names:
        reverse = m != "self_bleu"  # lower self_bleu is more diverse
        sorted_configs = sorted(
            [c for c in aggregated if m in aggregated[c]],
            key=lambda c: aggregated[c][m]["mean"],
            reverse=reverse,
        )
        rankings[m] = sorted_configs
        for rank_idx, c in enumerate(sorted_configs):
            aggregated[c][m]["rank"] = rank_idx + 1

    # Find best config per metric
    best_per_metric = {}
    for m, ranked in rankings.items():
        if ranked:
            best_per_metric[m] = {
                "best_config": ranked[0],
                "best_mean": aggregated[ranked[0]][m]["mean"],
                "worst_config": ranked[-1],
                "worst_mean": aggregated[ranked[-1]][m]["mean"],
            }

    # Average rank across metrics (overall best algorithm)
    avg_ranks = {}
    for config in aggregated:
        ranks = [aggregated[config][m].get("rank", len(aggregated)) for m in metric_names if m in aggregated[config]]
        if ranks:
            avg_ranks[config] = round(float(np.mean(ranks)), 2)

    overall_ranking = sorted(avg_ranks.items(), key=lambda x: x[1])

    return {
        "per_algorithm": aggregated,
        "rankings_per_metric": rankings,
        "best_per_metric": best_per_metric,
        "average_rank": dict(overall_ranking),
        "overall_best": overall_ranking[0] if overall_ranking else None,
    }


def compute_effect_sizes(
    metric_records: List[Dict],
    metric_names: List[str],
    baseline_config: str = "temp_1.0",
) -> Dict[str, Any]:
    """Compute Cohen's d and rank-biserial effect sizes vs a baseline.

    Parameters
    ----------
    metric_records : list of result dicts
    metric_names : metrics to analyze
    baseline_config : the reference config for comparisons

    Returns
    -------
    dict mapping (config, metric) -> effect size info
    """
    from collections import defaultdict

    config_data = defaultdict(lambda: defaultdict(list))
    for rec in metric_records:
        config = rec.get("config", "unknown")
        for m in metric_names:
            val = rec.get(m)
            if val is not None:
                config_data[config][m].append(val)

    if baseline_config not in config_data:
        return {"error": f"Baseline config '{baseline_config}' not found"}

    results = {}
    for config in config_data:
        if config == baseline_config:
            continue
        results[config] = {}
        for m in metric_names:
            baseline_vals = np.array(config_data[baseline_config].get(m, []))
            compare_vals = np.array(config_data[config].get(m, []))
            if len(baseline_vals) < 2 or len(compare_vals) < 2:
                continue

            # Cohen's d
            pooled_std = np.sqrt(
                ((len(baseline_vals) - 1) * np.var(baseline_vals, ddof=1)
                 + (len(compare_vals) - 1) * np.var(compare_vals, ddof=1))
                / (len(baseline_vals) + len(compare_vals) - 2)
            )
            if pooled_std < 1e-15:
                cohens_d = 0.0
            else:
                cohens_d = (np.mean(compare_vals) - np.mean(baseline_vals)) / pooled_std

            # Interpret effect size
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                interpretation = "negligible"
            elif abs_d < 0.5:
                interpretation = "small"
            elif abs_d < 0.8:
                interpretation = "medium"
            else:
                interpretation = "large"

            # Rank-biserial correlation (non-parametric effect size)
            from scipy.stats import mannwhitneyu
            try:
                u_stat, p_val = mannwhitneyu(compare_vals, baseline_vals, alternative="two-sided")
                n1, n2 = len(compare_vals), len(baseline_vals)
                rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
            except Exception:
                rank_biserial = 0.0
                p_val = 1.0

            results[config][m] = {
                "cohens_d": round(float(cohens_d), 4),
                "interpretation": interpretation,
                "rank_biserial": round(float(rank_biserial), 4),
                "mann_whitney_p": round(float(p_val), 6),
                "baseline_mean": round(float(np.mean(baseline_vals)), 4),
                "compare_mean": round(float(np.mean(compare_vals)), 4),
                "difference": round(float(np.mean(compare_vals) - np.mean(baseline_vals)), 4),
            }

    return {
        "baseline_config": baseline_config,
        "effect_sizes": results,
    }


def generate_latex_tables(
    algo_agg: Dict[str, Any],
    domain_agg: Dict[str, Any],
    metric_names: List[str],
    output_dir: Path = None,
) -> Dict[str, str]:
    """Generate LaTeX tables for the paper.

    Produces:
    1. Main results table: metrics per algorithm
    2. Domain breakdown table
    3. Ranking table

    Returns
    -------
    dict mapping table name -> LaTeX string
    """
    if output_dir is None:
        output_dir = RESULTS_DIR

    tables = {}

    # --- Table 1: Main results by algorithm ---
    configs = list(algo_agg.get("per_algorithm", {}).keys())
    display_metrics = [m for m in metric_names if any(
        m in algo_agg["per_algorithm"].get(c, {}) for c in configs
    )]

    header = " & ".join(["Config"] + [m.replace("_", "\\_") for m in display_metrics])
    rows = []
    for cfg in sorted(configs):
        vals = []
        for m in display_metrics:
            entry = algo_agg["per_algorithm"].get(cfg, {}).get(m, {})
            mean = entry.get("mean", "---")
            std = entry.get("std", "---")
            if isinstance(mean, float):
                vals.append(f"${mean:.3f}_{{{std:.3f}}}$")
            else:
                vals.append("---")
        row = cfg.replace("_", "\\_") + " & " + " & ".join(vals)
        rows.append(row)

    latex_main = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Diversity metrics across decoding configurations (mean$_{\\text{std}}$).}\n"
        "\\label{tab:main_results}\n"
        f"\\begin{{tabular}}{{l{'c' * len(display_metrics)}}}\n"
        "\\toprule\n"
        f"{header} \\\\\n"
        "\\midrule\n"
        + " \\\\\n".join(rows) + " \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )
    tables["main_results"] = latex_main

    # --- Table 2: Domain breakdown ---
    domains = list(domain_agg.get("per_domain", {}).keys())
    domain_rows = []
    for domain in sorted(domains):
        vals = []
        for m in display_metrics:
            entry = domain_agg["per_domain"].get(domain, {}).get(m, {})
            mean = entry.get("mean", "---")
            if isinstance(mean, float):
                vals.append(f"${mean:.3f}$")
            else:
                vals.append("---")
        row = domain.replace("_", "\\_") + " & " + " & ".join(vals)
        domain_rows.append(row)

    latex_domain = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Diversity metrics averaged across configurations per domain.}\n"
        "\\label{tab:domain_results}\n"
        f"\\begin{{tabular}}{{l{'c' * len(display_metrics)}}}\n"
        "\\toprule\n"
        f"{header.replace('Config', 'Domain')} \\\\\n"
        "\\midrule\n"
        + " \\\\\n".join(domain_rows) + " \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )
    tables["domain_results"] = latex_domain

    # --- Table 3: Rankings ---
    rankings = algo_agg.get("average_rank", {})
    rank_rows = []
    for rank_idx, (cfg, avg_rank) in enumerate(sorted(rankings.items(), key=lambda x: x[1])):
        per_metric_ranks = []
        for m in display_metrics:
            r = algo_agg["per_algorithm"].get(cfg, {}).get(m, {}).get("rank", "---")
            per_metric_ranks.append(str(r))
        row = f"{rank_idx + 1} & {cfg.replace('_', chr(92) + '_')} & {avg_rank:.1f} & " + " & ".join(per_metric_ranks)
        rank_rows.append(row)

    rank_header = " & ".join(
        ["Rank", "Config", "Avg"] + [m[:6].replace("_", "\\_") for m in display_metrics]
    )
    latex_rankings = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Algorithm rankings by diversity metrics.}\n"
        "\\label{tab:rankings}\n"
        f"\\begin{{tabular}}{{rl{'c' * (1 + len(display_metrics))}r}}\n"
        "\\toprule\n"
        f"{rank_header} \\\\\n"
        "\\midrule\n"
        + " \\\\\n".join(rank_rows) + " \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )
    tables["rankings"] = latex_rankings

    # Save to files
    for name, latex in tables.items():
        out_path = output_dir / f"table_{name}.tex"
        with open(out_path, "w") as f:
            f.write(latex)
        logger.info(f"Saved LaTeX table to {out_path}")

    return tables


def generate_summary_report(
    metric_records: List[Dict],
    algo_agg: Dict[str, Any],
    domain_agg: Dict[str, Any],
    effect_sizes: Dict[str, Any],
    correlation_analysis: Dict[str, Any],
    convergence: Dict[str, Any],
    metric_names: List[str],
    output_dir: Path = None,
) -> str:
    """Generate a comprehensive plain-text summary report.

    Returns
    -------
    report string, also saved to output_dir/summary_report.txt
    """
    if output_dir is None:
        output_dir = RESULTS_DIR

    lines = []
    lines.append("=" * 72)
    lines.append("LARGE-SCALE DIVERSITY METRIC TAXONOMY: SUMMARY REPORT")
    lines.append("=" * 72)
    lines.append("")

    # Overview
    lines.append("1. EXPERIMENT OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"   Total metric records: {len(metric_records)}")
    lines.append(f"   Metrics evaluated: {', '.join(metric_names)}")
    n_configs = len(set(r.get('config', '') for r in metric_records))
    n_domains = len(set(r.get('domain', '') for r in metric_records))
    lines.append(f"   Configurations: {n_configs}")
    lines.append(f"   Domains: {n_domains}")
    lines.append("")

    # Best algorithms
    lines.append("2. BEST ALGORITHMS PER METRIC")
    lines.append("-" * 40)
    best = algo_agg.get("best_per_metric", {})
    for m in metric_names:
        if m in best:
            b = best[m]
            lines.append(f"   {m}: {b['best_config']} (mean={b['best_mean']:.4f})")
    lines.append("")

    # Overall ranking
    lines.append("3. OVERALL ALGORITHM RANKING (by average rank)")
    lines.append("-" * 40)
    avg_rank = algo_agg.get("average_rank", {})
    for rank_idx, (cfg, ar) in enumerate(sorted(avg_rank.items(), key=lambda x: x[1])[:10]):
        lines.append(f"   {rank_idx + 1:2d}. {cfg:20s}  avg_rank={ar:.2f}")
    lines.append("")

    # Domain consistency
    lines.append("4. CROSS-DOMAIN CONSISTENCY")
    lines.append("-" * 40)
    consistency = domain_agg.get("cross_domain_consistency", {})
    for m, info in sorted(consistency.items()):
        cv = info.get("coefficient_of_variation", "N/A")
        stable = info.get("is_stable", "N/A")
        lines.append(f"   {m}: CV={cv}, stable={stable}")
    lines.append("")

    # Metric correlations
    lines.append("5. METRIC CORRELATIONS (highly correlated pairs)")
    lines.append("-" * 40)
    redundancy = correlation_analysis.get("redundancy_pairs", [])
    if redundancy:
        for pair in redundancy[:15]:
            lines.append(
                f"   {pair['metric_a']} <-> {pair['metric_b']}: "
                f"pearson={pair['pearson_r']:.3f}, spearman={pair['spearman_r']:.3f}"
            )
    else:
        lines.append("   No highly correlated pairs found (|r| > 0.8)")
    lines.append("")

    # Independence ranking
    lines.append("6. METRIC INDEPENDENCE RANKING (least correlated = most informative)")
    lines.append("-" * 40)
    indep = correlation_analysis.get("independence_ranking", [])
    for m, avg_corr in indep:
        lines.append(f"   {m:20s}  avg|r|={avg_corr:.3f}")
    lines.append("")

    # Effect sizes
    lines.append("7. NOTABLE EFFECT SIZES vs baseline")
    lines.append("-" * 40)
    baseline = effect_sizes.get("baseline_config", "unknown")
    lines.append(f"   Baseline: {baseline}")
    es_data = effect_sizes.get("effect_sizes", {})
    notable = []
    for config, metrics in es_data.items():
        for m, info in metrics.items():
            if info.get("interpretation") in ("medium", "large"):
                notable.append((config, m, info["cohens_d"], info["interpretation"]))
    notable.sort(key=lambda x: abs(x[2]), reverse=True)
    for cfg, m, d, interp in notable[:20]:
        lines.append(f"   {cfg:20s} {m:15s} d={d:+.3f} ({interp})")
    lines.append("")

    # Convergence
    lines.append("8. CONVERGENCE ANALYSIS")
    lines.append("-" * 40)
    min_samples = convergence.get("min_samples_for_stability", {})
    for m, info in min_samples.items():
        frac = info.get("min_fraction_for_tau_0.9")
        n_min = info.get("min_n_for_tau_0.9")
        lines.append(f"   {m}: min_fraction={frac}, min_n={n_min}")
    lines.append("")

    lines.append("=" * 72)
    lines.append("END OF REPORT")
    lines.append("=" * 72)

    report = "\n".join(lines)

    report_path = output_dir / "summary_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved summary report to {report_path}")

    return report


# ===========================================================================
# SECTION 6: Ablation Study Functions
# ===========================================================================

def temperature_ablation(
    metric_records: List[Dict],
    metric_names: List[str],
    temperature_configs: List[str] = None,
) -> Dict[str, Any]:
    """Ablation study: how does temperature affect each diversity metric?

    Isolates temperature as the varying parameter (top_p=1.0 fixed).

    Returns
    -------
    dict with temperature -> metric curves and optimal temperatures
    """
    from collections import defaultdict

    if temperature_configs is None:
        temperature_configs = [
            "temp_0.0", "temp_0.3", "temp_0.5", "temp_0.7",
            "temp_1.0", "temp_1.2", "temp_1.5", "temp_2.0",
        ]

    temp_data = defaultdict(lambda: defaultdict(list))
    for rec in metric_records:
        cfg = rec.get("config", "")
        if cfg in temperature_configs:
            for m in metric_names:
                val = rec.get(m)
                if val is not None:
                    temp_data[cfg][m].append(val)

    # Parse temperature values
    temp_values = {}
    for cfg in temperature_configs:
        try:
            temp_values[cfg] = float(cfg.replace("temp_", ""))
        except ValueError:
            temp_values[cfg] = 0.0

    curves = {}
    for m in metric_names:
        points = []
        for cfg in sorted(temperature_configs, key=lambda c: temp_values.get(c, 0)):
            vals = temp_data[cfg].get(m, [])
            if vals:
                ci = bootstrap_confidence_intervals(np.array(vals), n_bootstrap=1000)
                points.append({
                    "config": cfg,
                    "temperature": temp_values.get(cfg, 0.0),
                    "mean": ci["point_estimate"],
                    "ci_lower": ci["ci_lower"],
                    "ci_upper": ci["ci_upper"],
                    "se": ci["se"],
                    "n": len(vals),
                })
        curves[m] = points

    # Find optimal temperature for each metric
    optimal = {}
    for m, points in curves.items():
        if not points:
            continue
        if m == "self_bleu":
            # Lower is more diverse
            best = min(points, key=lambda p: p["mean"])
        else:
            best = max(points, key=lambda p: p["mean"])
        optimal[m] = {
            "optimal_temperature": best["temperature"],
            "optimal_config": best["config"],
            "optimal_value": best["mean"],
        }

    # Compute monotonicity (is the metric monotonically increasing with temperature?)
    monotonicity = {}
    for m, points in curves.items():
        if len(points) < 3:
            continue
        means = [p["mean"] for p in points]
        diffs = np.diff(means)
        n_increasing = int(np.sum(diffs > 0))
        n_decreasing = int(np.sum(diffs < 0))
        total_transitions = len(diffs)
        monotonicity[m] = {
            "n_increasing": n_increasing,
            "n_decreasing": n_decreasing,
            "is_monotone_increasing": n_increasing == total_transitions,
            "is_monotone_decreasing": n_decreasing == total_transitions,
            "monotonicity_score": max(n_increasing, n_decreasing) / max(total_transitions, 1),
        }

    return {
        "ablation_type": "temperature",
        "n_configs": len(temperature_configs),
        "curves": curves,
        "optimal_temperature": optimal,
        "monotonicity": monotonicity,
    }


def top_p_ablation(
    metric_records: List[Dict],
    metric_names: List[str],
    nucleus_configs: List[str] = None,
) -> Dict[str, Any]:
    """Ablation study: how does top_p (nucleus) threshold affect metrics?

    Isolates top_p as the varying parameter (temperature=1.0 fixed).

    Returns
    -------
    dict with top_p -> metric curves and optimal thresholds
    """
    from collections import defaultdict

    if nucleus_configs is None:
        nucleus_configs = [
            "nucleus_0.3", "nucleus_0.5", "nucleus_0.7",
            "nucleus_0.9", "nucleus_0.95",
        ]

    nuc_data = defaultdict(lambda: defaultdict(list))
    for rec in metric_records:
        cfg = rec.get("config", "")
        if cfg in nucleus_configs:
            for m in metric_names:
                val = rec.get(m)
                if val is not None:
                    nuc_data[cfg][m].append(val)

    # Parse top_p values
    p_values = {}
    for cfg in nucleus_configs:
        try:
            p_values[cfg] = float(cfg.replace("nucleus_", ""))
        except ValueError:
            p_values[cfg] = 0.5

    curves = {}
    for m in metric_names:
        points = []
        for cfg in sorted(nucleus_configs, key=lambda c: p_values.get(c, 0)):
            vals = nuc_data[cfg].get(m, [])
            if vals:
                ci = bootstrap_confidence_intervals(np.array(vals), n_bootstrap=1000)
                points.append({
                    "config": cfg,
                    "top_p": p_values.get(cfg, 0.5),
                    "mean": ci["point_estimate"],
                    "ci_lower": ci["ci_lower"],
                    "ci_upper": ci["ci_upper"],
                    "se": ci["se"],
                    "n": len(vals),
                })
        curves[m] = points

    # Optimal top_p per metric
    optimal = {}
    for m, points in curves.items():
        if not points:
            continue
        if m == "self_bleu":
            best = min(points, key=lambda p: p["mean"])
        else:
            best = max(points, key=lambda p: p["mean"])
        optimal[m] = {
            "optimal_top_p": best["top_p"],
            "optimal_config": best["config"],
            "optimal_value": best["mean"],
        }

    # Sensitivity: how much does each metric change across the top_p range?
    sensitivity = {}
    for m, points in curves.items():
        if len(points) < 2:
            continue
        means = [p["mean"] for p in points]
        sensitivity[m] = {
            "range": round(float(max(means) - min(means)), 4),
            "relative_range": round(float((max(means) - min(means)) / max(abs(np.mean(means)), 1e-10)), 4),
            "std_across_configs": round(float(np.std(means)), 4),
        }

    return {
        "ablation_type": "top_p",
        "n_configs": len(nucleus_configs),
        "curves": curves,
        "optimal_top_p": optimal,
        "sensitivity": sensitivity,
    }


def sample_size_ablation(
    metric_records: List[Dict],
    metric_names: List[str],
    sample_sizes: List[int] = None,
    n_trials: int = 100,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Ablation: how do metric values change with different numbers of samples?

    For each sample size k, randomly draw k texts from each generation group
    and recompute metrics. Measures metric stability.

    Returns
    -------
    dict with sample_size -> metric stability data
    """
    if sample_sizes is None:
        sample_sizes = [2, 3, 5, 7, 10]

    rng = np.random.RandomState(random_state)

    # Group records by config to get raw texts back
    # Since we may not have raw texts, simulate with metric values
    # Use per-record analysis with subsampling of records per config
    from collections import defaultdict

    config_records = defaultdict(list)
    for rec in metric_records:
        config_records[rec.get("config", "unknown")].append(rec)

    results_by_size = {}
    for k in sample_sizes:
        metric_stats = defaultdict(list)

        for config, recs in config_records.items():
            if len(recs) < k:
                continue
            for trial in range(min(n_trials, 50)):
                sample_indices = rng.choice(len(recs), size=k, replace=False)
                sample = [recs[i] for i in sample_indices]

                for m in metric_names:
                    vals = [s.get(m) for s in sample if s.get(m) is not None]
                    if vals:
                        metric_stats[m].append(np.mean(vals))

        size_result = {}
        for m in metric_names:
            vals = metric_stats.get(m, [])
            if len(vals) >= 2:
                size_result[m] = {
                    "mean_of_means": round(float(np.mean(vals)), 4),
                    "std_of_means": round(float(np.std(vals)), 4),
                    "cv": round(float(np.std(vals) / max(abs(np.mean(vals)), 1e-10)), 4),
                    "n_trials": len(vals),
                }

        results_by_size[k] = size_result

    # Determine minimum sample size for stability (CV < 0.1)
    min_k = {}
    for m in metric_names:
        found = None
        for k in sorted(sample_sizes):
            info = results_by_size.get(k, {}).get(m, {})
            cv = info.get("cv", float("inf"))
            if cv < 0.1:
                found = k
                break
        min_k[m] = found

    return {
        "ablation_type": "sample_size",
        "sample_sizes": sample_sizes,
        "results_by_size": results_by_size,
        "min_k_for_cv_0.1": min_k,
    }


def prompt_sensitivity_analysis(
    metric_records: List[Dict],
    metric_names: List[str],
) -> Dict[str, Any]:
    """Analyze how sensitive metric values are to prompt choice.

    Computes within-prompt vs between-prompt variance to assess whether
    prompts or algorithms explain more variance in each metric.

    Returns
    -------
    dict with variance decomposition and sensitivity scores per metric
    """
    from collections import defaultdict

    # Group by prompt and config
    prompt_config_data = defaultdict(lambda: defaultdict(list))
    prompt_data = defaultdict(lambda: defaultdict(list))
    config_data = defaultdict(lambda: defaultdict(list))

    for rec in metric_records:
        prompt = rec.get("prompt", "unknown")
        config = rec.get("config", "unknown")
        for m in metric_names:
            val = rec.get(m)
            if val is not None:
                prompt_config_data[(prompt, config)][m].append(val)
                prompt_data[prompt][m].append(val)
                config_data[config][m].append(val)

    results = {}
    for m in metric_names:
        # Total variance
        all_vals = []
        for rec in metric_records:
            val = rec.get(m)
            if val is not None:
                all_vals.append(val)
        if len(all_vals) < 5:
            continue
        total_var = float(np.var(all_vals))

        # Between-prompt variance
        prompt_means = [np.mean(prompt_data[p][m]) for p in prompt_data if m in prompt_data[p] and len(prompt_data[p][m]) >= 2]
        between_prompt_var = float(np.var(prompt_means)) if len(prompt_means) >= 2 else 0.0

        # Between-config variance
        config_means = [np.mean(config_data[c][m]) for c in config_data if m in config_data[c] and len(config_data[c][m]) >= 2]
        between_config_var = float(np.var(config_means)) if len(config_means) >= 2 else 0.0

        # Within-group variance (residual)
        within_var = max(0.0, total_var - between_prompt_var - between_config_var)

        # Eta-squared (proportion of variance explained)
        eta_sq_prompt = between_prompt_var / max(total_var, 1e-15)
        eta_sq_config = between_config_var / max(total_var, 1e-15)

        # Most and least sensitive prompts
        prompt_variability = {}
        for p in prompt_data:
            vals = prompt_data[p].get(m, [])
            if len(vals) >= 3:
                prompt_variability[p] = float(np.std(vals))

        sorted_prompts = sorted(prompt_variability.items(), key=lambda x: x[1], reverse=True)

        results[m] = {
            "total_variance": round(total_var, 6),
            "between_prompt_variance": round(between_prompt_var, 6),
            "between_config_variance": round(between_config_var, 6),
            "within_variance": round(within_var, 6),
            "eta_squared_prompt": round(eta_sq_prompt, 4),
            "eta_squared_config": round(eta_sq_config, 4),
            "prompt_explains_more": eta_sq_prompt > eta_sq_config,
            "most_sensitive_prompts": [
                {"prompt": p[:60], "std": round(s, 4)}
                for p, s in sorted_prompts[:5]
            ],
            "least_sensitive_prompts": [
                {"prompt": p[:60], "std": round(s, 4)}
                for p, s in sorted_prompts[-5:]
            ] if len(sorted_prompts) >= 5 else [],
        }

    return {
        "ablation_type": "prompt_sensitivity",
        "n_prompts": len(prompt_data),
        "n_configs": len(config_data),
        "variance_decomposition": results,
    }


# ===========================================================================
# SECTION 7: Comprehensive main() and orchestration
# ===========================================================================

def run_extended_analysis(metric_records: List[Dict], metric_names: List[str]) -> Dict[str, Any]:
    """Run all statistical analyses on computed metric records.

    This is the central analysis pipeline that invokes all analysis functions
    and collects their results.

    Parameters
    ----------
    metric_records : list of dicts with metric values, config, domain, etc.
    metric_names : list of metric key names

    Returns
    -------
    dict with all analysis results
    """
    logger.info("=" * 60)
    logger.info("RUNNING EXTENDED ANALYSIS PIPELINE")
    logger.info("=" * 60)

    all_results = {}

    # 5a. Aggregate by domain
    logger.info("[1/8] Aggregating results by domain...")
    domain_agg = aggregate_results_by_domain(metric_records, metric_names)
    all_results["domain_aggregation"] = domain_agg
    logger.info(f"  Domains: {domain_agg['n_domains']}")

    # 5b. Aggregate by algorithm
    logger.info("[2/8] Aggregating results by algorithm...")
    algo_agg = aggregate_results_by_algorithm(metric_records, metric_names)
    all_results["algorithm_aggregation"] = algo_agg
    overall_best = algo_agg.get("overall_best")
    if overall_best:
        logger.info(f"  Overall best: {overall_best[0]} (avg rank {overall_best[1]})")

    # 5c. Effect sizes
    logger.info("[3/8] Computing effect sizes...")
    effect_sizes = compute_effect_sizes(metric_records, metric_names, baseline_config="temp_1.0")
    all_results["effect_sizes"] = effect_sizes

    # 4a. Metric correlations
    logger.info("[4/8] Computing metric correlations...")
    corr = metric_correlation_analysis(metric_records, metric_names)
    all_results["metric_correlations"] = corr
    n_redundant = len(corr.get("redundancy_pairs", []))
    logger.info(f"  Redundant metric pairs (|r|>0.8): {n_redundant}")

    # 4b. Convergence analysis
    logger.info("[5/8] Running convergence analysis...")
    conv = convergence_analysis(metric_records, metric_names)
    all_results["convergence"] = conv

    # 6a. Temperature ablation
    logger.info("[6/8] Running temperature ablation...")
    temp_abl = temperature_ablation(metric_records, metric_names)
    all_results["temperature_ablation"] = temp_abl

    # 6b. Top-p ablation
    logger.info("[7/8] Running top-p ablation...")
    topp_abl = top_p_ablation(metric_records, metric_names)
    all_results["top_p_ablation"] = topp_abl

    # 6c. Sample size ablation
    logger.info("[8/8] Running sample size ablation...")
    ss_abl = sample_size_ablation(metric_records, metric_names)
    all_results["sample_size_ablation"] = ss_abl

    return all_results


def run_pairwise_comparisons(
    metric_records: List[Dict],
    metric_names: List[str],
    comparison_metric: str = "tfidf_epd",
) -> Dict[str, Any]:
    """Run Bayesian sign tests and Bradley-Terry rankings across configs.

    Performs pairwise comparisons between all configurations on a given metric,
    then fits a BT model to the resulting win matrix.

    Parameters
    ----------
    metric_records : list of result dicts
    metric_names : available metrics
    comparison_metric : metric used for pairwise comparison

    Returns
    -------
    dict with sign test results and BT rankings
    """
    from collections import defaultdict

    if comparison_metric not in metric_names:
        return {"error": f"Metric {comparison_metric} not available"}

    logger.info(f"Running pairwise comparisons on {comparison_metric}...")

    # Group by (prompt, seed) to get paired observations
    ps_groups = defaultdict(dict)
    for rec in metric_records:
        ps_key = (rec.get("prompt", ""), rec.get("seed", 0))
        config = rec.get("config", "unknown")
        val = rec.get(comparison_metric)
        if val is not None:
            ps_groups[ps_key][config] = val

    configs = sorted(set(rec.get("config", "") for rec in metric_records))
    n_configs = len(configs)
    config_idx = {c: i for i, c in enumerate(configs)}

    # Build win matrix and run sign tests for all pairs
    win_matrix = np.zeros((n_configs, n_configs))
    sign_test_results = {}
    higher_is_better = comparison_metric != "self_bleu"

    for i, c1 in enumerate(configs):
        for j, c2 in enumerate(configs):
            if j <= i:
                continue

            scores_1 = []
            scores_2 = []
            for ps_key, config_vals in ps_groups.items():
                if c1 in config_vals and c2 in config_vals:
                    scores_1.append(config_vals[c1])
                    scores_2.append(config_vals[c2])

            if len(scores_1) < 5:
                continue

            s1 = np.array(scores_1)
            s2 = np.array(scores_2)

            if higher_is_better:
                sign_result = bayesian_sign_tests(s1, s2)
            else:
                sign_result = bayesian_sign_tests(s2, s1)

            sign_test_results[f"{c1}_vs_{c2}"] = sign_result

            # Update win matrix
            if higher_is_better:
                win_matrix[i, j] = sign_result["n_a_wins"]
                win_matrix[j, i] = sign_result["n_b_wins"]
            else:
                win_matrix[i, j] = sign_result["n_b_wins"]
                win_matrix[j, i] = sign_result["n_a_wins"]

    # Fit Bradley-Terry model
    bt_results = bradley_terry_rankings(win_matrix, configs, n_bootstrap=500)

    return {
        "comparison_metric": comparison_metric,
        "n_configs": n_configs,
        "sign_tests": sign_test_results,
        "bradley_terry": bt_results,
    }


def run_pareto_on_configs(
    metric_records: List[Dict],
    objective_metrics: List[str] = None,
    maximize: List[bool] = None,
) -> Dict[str, Any]:
    """Compute Pareto frontier over algorithm configs on selected objectives.

    Parameters
    ----------
    metric_records : result dicts
    objective_metrics : metrics to use as objectives
    maximize : direction for each objective

    Returns
    -------
    Pareto analysis results
    """
    from collections import defaultdict

    if objective_metrics is None:
        objective_metrics = ["distinct_2", "tfidf_epd", "entropy"]
    if maximize is None:
        maximize = [True] * len(objective_metrics)

    config_means = defaultdict(lambda: defaultdict(list))
    for rec in metric_records:
        config = rec.get("config", "unknown")
        for m in objective_metrics:
            val = rec.get(m)
            if val is not None:
                config_means[config][m].append(val)

    # Build point matrix
    configs = sorted(config_means.keys())
    valid_configs = []
    points = []
    for c in configs:
        vals = []
        skip = False
        for m in objective_metrics:
            if m not in config_means[c] or len(config_means[c][m]) == 0:
                skip = True
                break
            vals.append(np.mean(config_means[c][m]))
        if not skip:
            valid_configs.append(c)
            points.append(vals)

    if len(valid_configs) < 2:
        return {"error": "Not enough valid configs for Pareto analysis"}

    points_arr = np.array(points)
    return pareto_analysis(points_arr, valid_configs, objective_metrics, maximize)


def run_prompt_sensitivity(
    metric_records: List[Dict],
    metric_names: List[str],
) -> Dict[str, Any]:
    """Run the prompt sensitivity ablation and return results."""
    logger.info("Running prompt sensitivity analysis...")
    return prompt_sensitivity_analysis(metric_records, metric_names)


def save_all_results(results: Dict[str, Any], output_dir: Path = None):
    """Save all analysis results to JSON files."""
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main extended analysis
    extended_path = output_dir / "extended_analysis.json"
    with open(extended_path, "w") as f:
        json.dump(results.get("extended_analysis", {}), f, indent=2, default=str)
    logger.info(f"Saved extended analysis to {extended_path}")

    # Save pairwise comparisons
    pairwise_path = output_dir / "pairwise_comparisons.json"
    with open(pairwise_path, "w") as f:
        json.dump(results.get("pairwise_comparisons", {}), f, indent=2, default=str)
    logger.info(f"Saved pairwise comparisons to {pairwise_path}")

    # Save Pareto analysis
    pareto_path = output_dir / "pareto_analysis.json"
    with open(pareto_path, "w") as f:
        json.dump(results.get("pareto_analysis", {}), f, indent=2, default=str)
    logger.info(f"Saved Pareto analysis to {pareto_path}")

    # Save prompt sensitivity
    prompt_sens_path = output_dir / "prompt_sensitivity.json"
    with open(prompt_sens_path, "w") as f:
        json.dump(results.get("prompt_sensitivity", {}), f, indent=2, default=str)
    logger.info(f"Saved prompt sensitivity to {prompt_sens_path}")

    # Save ablation results
    ablation_path = output_dir / "ablation_results.json"
    ablation_data = {
        "temperature_ablation": results.get("extended_analysis", {}).get("temperature_ablation", {}),
        "top_p_ablation": results.get("extended_analysis", {}).get("top_p_ablation", {}),
        "sample_size_ablation": results.get("extended_analysis", {}).get("sample_size_ablation", {}),
    }
    with open(ablation_path, "w") as f:
        json.dump(ablation_data, f, indent=2, default=str)
    logger.info(f"Saved ablation results to {ablation_path}")


def comprehensive_main():
    """Comprehensive main function that orchestrates all phases of the experiment.

    Phase 1: Generate texts using the API (or load from cache)
    Phase 2: Get semantic embeddings (or load from cache)
    Phase 3: Compute metrics for all generation groups
    Phase 4: Run extended statistical analyses
    Phase 5: Run pairwise comparisons and Bradley-Terry rankings
    Phase 6: Run Pareto frontier analysis
    Phase 7: Run ablation studies
    Phase 8: Generate LaTeX tables and summary report
    """
    t0 = time.time()

    logger.info("=" * 72)
    logger.info("COMPREHENSIVE LARGE-SCALE DIVERSITY EXPERIMENT")
    logger.info("=" * 72)
    logger.info(f"Extended prompts: {len(EXTENDED_ALL_PROMPTS)} across "
                f"{len(EXTENDED_PROMPTS_BY_DOMAIN)} domains")
    logger.info(f"Algorithm families: {len(ALGORITHM_FAMILY_CONFIGS)}")
    logger.info(f"Total algorithm configs: {len(ALL_ALGORITHM_CONFIGS)}")
    logger.info(f"Model configs available: {len(MODEL_CONFIGS)}")
    logger.info("")

    # Phase 1: Generate texts
    logger.info("PHASE 1: Text Generation")
    logger.info("-" * 40)
    gen_results = run_generation_phase()
    logger.info(f"  Loaded/generated {len(gen_results)} generation groups")

    # Phase 2: Embeddings
    logger.info("PHASE 2: Semantic Embeddings")
    logger.info("-" * 40)
    text_to_emb = run_embedding_phase(gen_results)
    logger.info(f"  Loaded/computed {len(text_to_emb)} embeddings")

    # Phase 3: Compute metrics (original analysis)
    logger.info("PHASE 3: Metric Computation and Original Analysis")
    logger.info("-" * 40)
    original_results = run_analysis(gen_results, text_to_emb)

    # Load metric records for extended analysis
    metric_records_path = RESULTS_DIR / "metric_records.json"
    if metric_records_path.exists():
        with open(metric_records_path) as f:
            metric_records = json.load(f)
    else:
        logger.warning("metric_records.json not found, skipping extended analysis")
        print_summary(original_results)
        print(f"\nTotal time: {time.time() - t0:.1f}s")
        return original_results

    metric_names = ["distinct_2", "self_bleu", "jaccard", "entropy",
                    "tfidf_epd", "tfidf_vendi", "semantic_epd", "semantic_vendi"]

    # Phase 4: Extended statistical analysis
    logger.info("PHASE 4: Extended Statistical Analysis")
    logger.info("-" * 40)
    extended = run_extended_analysis(metric_records, metric_names)

    # Phase 5: Pairwise comparisons
    logger.info("PHASE 5: Pairwise Comparisons and Bradley-Terry")
    logger.info("-" * 40)
    pairwise_metrics = ["tfidf_epd", "distinct_2", "entropy"]
    pairwise_results = {}
    for pm in pairwise_metrics:
        if any(rec.get(pm) is not None for rec in metric_records):
            pairwise_results[pm] = run_pairwise_comparisons(metric_records, metric_names, pm)
            bt = pairwise_results[pm].get("bradley_terry", {})
            rankings = bt.get("rankings", [])
            if rankings:
                top3 = ", ".join(f"{r['name']}({r['score']:.4f})" for r in rankings[:3])
                logger.info(f"  BT top-3 on {pm}: {top3}")

    # Phase 6: Pareto frontier
    logger.info("PHASE 6: Pareto Frontier Analysis")
    logger.info("-" * 40)
    pareto_configs = [
        {
            "name": "diversity_tradeoff",
            "objectives": ["distinct_2", "tfidf_epd"],
            "maximize": [True, True],
        },
        {
            "name": "quality_diversity_tradeoff",
            "objectives": ["distinct_2", "entropy", "tfidf_vendi"],
            "maximize": [True, True, True],
        },
    ]
    pareto_results = {}
    for pc in pareto_configs:
        available = [o for o in pc["objectives"] if any(r.get(o) is not None for r in metric_records)]
        avail_max = [pc["maximize"][pc["objectives"].index(o)] for o in available]
        if len(available) >= 2:
            pareto_results[pc["name"]] = run_pareto_on_configs(
                metric_records, available, avail_max,
            )
            n_pareto = pareto_results[pc["name"]].get("n_pareto", 0)
            n_total = pareto_results[pc["name"]].get("n_points", 0)
            logger.info(f"  {pc['name']}: {n_pareto}/{n_total} on Pareto front")

    # Phase 7: Prompt sensitivity
    logger.info("PHASE 7: Prompt Sensitivity")
    logger.info("-" * 40)
    prompt_sens = run_prompt_sensitivity(metric_records, metric_names)
    vd = prompt_sens.get("variance_decomposition", {})
    for m in metric_names[:3]:
        if m in vd:
            eta_p = vd[m].get("eta_squared_prompt", 0)
            eta_c = vd[m].get("eta_squared_config", 0)
            logger.info(f"  {m}: η²_prompt={eta_p:.3f}, η²_config={eta_c:.3f}")

    # Phase 8: Generate tables and report
    logger.info("PHASE 8: Tables and Report Generation")
    logger.info("-" * 40)
    algo_agg = extended.get("algorithm_aggregation", {})
    domain_agg = extended.get("domain_aggregation", {})
    effect_sizes = extended.get("effect_sizes", {})
    corr = extended.get("metric_correlations", {})
    conv = extended.get("convergence", {})

    latex_tables = generate_latex_tables(algo_agg, domain_agg, metric_names)
    logger.info(f"  Generated {len(latex_tables)} LaTeX tables")

    report = generate_summary_report(
        metric_records, algo_agg, domain_agg, effect_sizes, corr, conv, metric_names,
    )

    # Collect all results
    all_results = {
        "original_analysis": original_results,
        "extended_analysis": extended,
        "pairwise_comparisons": pairwise_results,
        "pareto_analysis": pareto_results,
        "prompt_sensitivity": prompt_sens,
    }

    # Save everything
    save_all_results(all_results)

    # Print summary
    print_summary(original_results)

    # Print extended highlights
    print("\n" + "=" * 72)
    print("EXTENDED ANALYSIS HIGHLIGHTS")
    print("=" * 72)

    # Overall best algorithm
    if algo_agg.get("overall_best"):
        cfg, rank = algo_agg["overall_best"]
        print(f"\nOverall best algorithm: {cfg} (avg rank {rank})")

    # Bradley-Terry rankings
    for pm, pr in pairwise_results.items():
        bt = pr.get("bradley_terry", {})
        rankings = bt.get("rankings", [])
        if rankings:
            print(f"\nBradley-Terry rankings ({pm}):")
            for r in rankings[:5]:
                ci = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
                print(f"  #{r['rank']} {r['name']:20s} score={r['score']:.4f} {ci}")

    # Pareto summary
    for name, pr in pareto_results.items():
        n_pareto = pr.get("n_pareto", 0)
        n_total = pr.get("n_points", 0)
        hv = pr.get("hypervolume", 0)
        print(f"\nPareto ({name}): {n_pareto}/{n_total} configs on frontier, HV={hv:.4f}")
        for pf in pr.get("pareto_front", [])[:5]:
            vals = ", ".join(f"{k}={v:.3f}" for k, v in pf.get("values", {}).items())
            print(f"  {pf['name']}: {vals}")

    # Ablation highlights
    temp_abl = extended.get("temperature_ablation", {})
    opt_temps = temp_abl.get("optimal_temperature", {})
    if opt_temps:
        print("\nOptimal temperatures per metric:")
        for m, info in sorted(opt_temps.items()):
            print(f"  {m}: T*={info['optimal_temperature']}, val={info['optimal_value']:.4f}")

    topp_abl = extended.get("top_p_ablation", {})
    opt_topp = topp_abl.get("optimal_top_p", {})
    if opt_topp:
        print("\nOptimal top-p per metric:")
        for m, info in sorted(opt_topp.items()):
            print(f"  {m}: p*={info['optimal_top_p']}, val={info['optimal_value']:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal experiment time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    logger.info(f"Experiment complete in {elapsed:.1f}s")

    return all_results


if __name__ == "__main__":
    t0 = time.time()

    # Phase 1: Generate texts
    gen_results = run_generation_phase()

    # Phase 2: Get semantic embeddings
    text_to_emb = run_embedding_phase(gen_results)

    # Phase 3: Analyze
    results = run_analysis(gen_results, text_to_emb)

    # Print summary
    print_summary(results)

    print(f"\nTotal time: {time.time()-t0:.1f}s")

    # Run the comprehensive analysis if --extended flag is passed
    if "--extended" in sys.argv:
        comprehensive_main()
