#!/usr/bin/env python3
"""
Human evaluation proxy using LLM-as-judge for diversity assessment.

Addresses the critique: "no human evaluation"

Approach:
  - Uses an LLM (via OpenAI API or local model) as a diversity judge
  - Presents pairs of text sets and asks which is more diverse
  - Computes Kendall τ between LLM judge rankings and each automatic metric
  - Serves as a proxy for human evaluation

Run from implementation/ directory:
    PYTHONPATH=. python3 experiments/human_eval_proxy.py

Requires: openai (optional, falls back to heuristic judge)
"""

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics.diversity import (
    SelfBLEU,
    DistinctN,
    NGramEntropy,
    EmbeddingPairwiseDistance,
    VendiScore,
    tokenize_simple,
)
from src.metrics.neural_diversity import (
    MAUVE,
    BERTScoreDiversity,
    STSDiversity,
    CompressionRatioDiversity,
)

RESULTS_DIR = Path(__file__).parent / "human_eval_results"
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an expert judge evaluating the diversity of text sets.

I will show you a set of generated texts. Rate the DIVERSITY of this set on a scale from 1 to 10, where:
- 1 = All texts are essentially identical or paraphrases of each other
- 5 = Moderate diversity with some variety in content and style
- 10 = Extremely diverse with very different topics, styles, and structures

Consider these dimensions of diversity:
1. Lexical diversity: Different word choices and vocabulary
2. Semantic diversity: Different meanings and topics
3. Structural diversity: Different sentence structures and formats
4. Stylistic diversity: Different tones, registers, and writing styles

Here are the texts:

{texts}

Respond with ONLY a JSON object: {{"score": <number>, "reasoning": "<brief explanation>"}}"""


class LLMJudge:
    """LLM-based diversity judge."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1-nano"):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model
        self._available = False
        if self._api_key:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self._api_key)
                self._available = True
            except ImportError:
                logger.warning("openai package not installed, using heuristic judge")

    @property
    def available(self) -> bool:
        return self._available

    def judge(self, texts: List[str], max_texts: int = 10) -> Dict[str, Any]:
        """Judge diversity of a text set. Returns {"score": float, "reasoning": str}."""
        # Subsample if too many
        if len(texts) > max_texts:
            rng = np.random.default_rng(SEED)
            indices = rng.choice(len(texts), size=max_texts, replace=False)
            texts = [texts[i] for i in indices]

        if self._available:
            return self._llm_judge(texts)
        return self._heuristic_judge(texts)

    def _llm_judge(self, texts: List[str]) -> Dict[str, Any]:
        """Use LLM API for judgment."""
        numbered = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
        prompt = JUDGE_PROMPT.format(texts=numbered)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            content = response.choices[0].message.content.strip()
            # Parse JSON response
            result = json.loads(content)
            return {
                "score": float(result.get("score", 5)),
                "reasoning": result.get("reasoning", ""),
                "method": "llm",
            }
        except Exception as e:
            logger.warning("LLM judge failed: %s, falling back to heuristic", e)
            return self._heuristic_judge(texts)

    def _heuristic_judge(self, texts: List[str]) -> Dict[str, Any]:
        """Multi-signal heuristic diversity judge (fallback).

        Combines multiple diversity signals to approximate human judgment:
        - Lexical diversity (unique token ratio)
        - Pairwise Jaccard distance
        - Sentence length variation
        - Vocabulary richness
        """
        if len(texts) < 2:
            return {"score": 1.0, "reasoning": "Only one text", "method": "heuristic"}

        # 1. Token-level diversity
        all_tokens = []
        token_sets = []
        lengths = []
        for t in texts:
            tokens = tokenize_simple(t)
            all_tokens.extend(tokens)
            token_sets.append(set(tokens))
            lengths.append(len(tokens))

        unique_ratio = len(set(all_tokens)) / max(len(all_tokens), 1)

        # 2. Pairwise Jaccard distance
        jaccard_dists = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                union = len(token_sets[i] | token_sets[j])
                inter = len(token_sets[i] & token_sets[j])
                jaccard_dists.append(1.0 - inter / max(union, 1))
        mean_jaccard = np.mean(jaccard_dists) if jaccard_dists else 0.0

        # 3. Length variation (coefficient of variation)
        length_cv = np.std(lengths) / max(np.mean(lengths), 1)

        # 4. Unique text ratio
        unique_texts = len(set(texts)) / len(texts)

        # 5. First-word diversity
        first_words = [tokenize_simple(t)[0] if tokenize_simple(t) else "" for t in texts]
        first_word_div = len(set(first_words)) / len(first_words)

        # Combine signals (weighted sum, scaled to 1-10)
        raw_score = (
            0.30 * mean_jaccard +
            0.25 * unique_ratio +
            0.15 * min(length_cv, 1.0) +
            0.20 * unique_texts +
            0.10 * first_word_div
        )
        score = 1.0 + 9.0 * min(max(raw_score, 0.0), 1.0)

        return {
            "score": round(score, 2),
            "reasoning": (
                f"jaccard={mean_jaccard:.3f}, unique_ratio={unique_ratio:.3f}, "
                f"length_cv={length_cv:.3f}, unique_texts={unique_texts:.2f}"
            ),
            "method": "heuristic",
        }


# ---------------------------------------------------------------------------
# Generate evaluation data
# ---------------------------------------------------------------------------

def generate_evaluation_sets(seed: int = SEED) -> List[Dict[str, Any]]:
    """Create text sets with varying diversity for evaluation.

    Returns list of {"texts": [...], "config": {...}, "expected_diversity": str}
    """
    rng = np.random.default_rng(seed)
    sets = []

    # 1. Identical texts (very low diversity)
    sets.append({
        "texts": ["The cat sat on the mat and watched the birds fly by."] * 8,
        "config": "identical",
        "expected_diversity": "very_low",
    })

    # 2. Minor paraphrases (low diversity)
    sets.append({
        "texts": [
            "The cat sat on the mat and watched the birds.",
            "The cat was sitting on the mat, watching birds.",
            "A cat sat on a mat and observed the birds.",
            "The cat rested on the mat, looking at birds.",
            "On the mat, the cat sat watching the birds fly.",
            "The cat was on the mat, watching birds go by.",
            "Sitting on the mat, the cat watched some birds.",
            "The cat, on the mat, observed birds flying by.",
        ],
        "config": "paraphrases",
        "expected_diversity": "low",
    })

    # 3. Same topic, varied style (moderate diversity)
    sets.append({
        "texts": [
            "Cats are fascinating creatures with remarkable hunting instincts.",
            "The domestic feline: a study in evolutionary adaptation and grace.",
            "My cat Whiskers always brings me dead mice as gifts.",
            "According to recent research, cats can recognize their owners' voices.",
            "In ancient Egypt, cats were revered as sacred animals.",
            "The stray cat population in urban areas has been growing rapidly.",
            "Cat cafes have become popular in Japan and are spreading worldwide.",
            "Training a cat requires patience, treats, and understanding their nature.",
        ],
        "config": "same_topic_varied",
        "expected_diversity": "moderate",
    })

    # 4. Different topics, same style (moderate-high diversity)
    sets.append({
        "texts": [
            "The quantum computer processed information at unprecedented speeds.",
            "She prepared an elaborate five-course French dinner for the guests.",
            "The ancient ruins revealed secrets of a lost civilization.",
            "Climate change is rapidly transforming Arctic ecosystems and wildlife.",
            "The stock market experienced its largest single-day gain in years.",
            "Astronomers discovered a potentially habitable exoplanet orbiting a red dwarf.",
            "The new education policy aims to reduce inequality in schools.",
            "Deep sea exploration uncovered previously unknown species near hydrothermal vents.",
        ],
        "config": "different_topics",
        "expected_diversity": "high",
    })

    # 5. Fully diverse (different topics, styles, lengths)
    sets.append({
        "texts": [
            "Once upon a time, a dragon befriended a mouse.",
            "BREAKING: Major earthquake hits coastal region, evacuations underway.",
            "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
            "Dear Sir/Madam, I am writing to express my concern regarding...",
            "lol this movie was SO bad but also kinda funny ngl 😂",
            "The mitochondria is the powerhouse of the cell, converting nutrients.",
            "Roses are red, violets are blue, algorithms are fun, and so are you.",
            "In conclusion, the evidence strongly supports the hypothesis that...",
        ],
        "config": "fully_diverse",
        "expected_diversity": "very_high",
    })

    # 6-10: Synthetic temperature-like variations
    base_words = [
        "machine", "learning", "neural", "network", "deep", "training",
        "model", "data", "prediction", "optimization", "gradient", "loss",
        "architecture", "transformer", "attention", "embedding",
    ]
    for temp_label, variance in [
        ("synth_low_var", 0.1),
        ("synth_med_var", 0.4),
        ("synth_high_var", 0.7),
        ("synth_very_high_var", 0.9),
        ("synth_max_var", 1.0),
    ]:
        texts = []
        for i in range(8):
            n_words = rng.integers(8, 20)
            if variance < 0.5:
                # Low variance: mostly same words
                pool = base_words[:5]
            else:
                pool = base_words
            words = [pool[rng.integers(len(pool))] for _ in range(n_words)]
            # Add some randomness proportional to variance
            extra_words = ["amazing", "complex", "novel", "efficient", "robust",
                          "scalable", "innovative", "groundbreaking"]
            n_extra = int(variance * 5)
            for _ in range(n_extra):
                pos = rng.integers(len(words))
                words[pos] = extra_words[rng.integers(len(extra_words))]
            texts.append(" ".join(words) + ".")
        sets.append({
            "texts": texts,
            "config": temp_label,
            "expected_diversity": temp_label,
        })

    return sets


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_metrics_for_set(texts: List[str]) -> Dict[str, float]:
    """Compute diversity metrics for a single text set."""
    metrics = {}
    instances = [
        ("distinct_2", DistinctN(n=2)),
        ("self_bleu", SelfBLEU(max_order=4)),
        ("ngram_entropy", NGramEntropy(n=2)),
        ("embedding_distance", EmbeddingPairwiseDistance()),
        ("vendi_score", VendiScore()),
        ("mauve", MAUVE(n_clusters=min(20, len(texts) // 2), backend="tfidf")),
        ("bertscore_div", BERTScoreDiversity(backend="tfidf")),
        ("sts_div", STSDiversity(backend="tfidf")),
        ("crd", CompressionRatioDiversity()),
    ]
    for name, metric in instances:
        try:
            metrics[name] = metric.compute(texts)
        except Exception as e:
            metrics[name] = float("nan")
    return metrics


def kendall_tau(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi = x[i] - x[j]
            yi = y[i] - y[j]
            if xi * yi > 0:
                c += 1
            elif xi * yi < 0:
                d += 1
    total = c + d
    return (c - d) / total if total > 0 else 0.0


def run_human_eval_proxy():
    """Main human evaluation proxy pipeline."""
    logger.info("=" * 60)
    logger.info("HUMAN EVALUATION PROXY (LLM-as-Judge)")
    logger.info("=" * 60)

    # Initialize judge
    judge = LLMJudge()
    judge_method = "LLM" if judge.available else "heuristic"
    logger.info("Using %s judge", judge_method)

    # Generate evaluation sets
    eval_sets = generate_evaluation_sets()
    logger.info("Created %d evaluation sets", len(eval_sets))

    # Compute judge scores and automatic metrics
    results = []
    for i, es in enumerate(eval_sets):
        texts = es["texts"]
        config = es["config"]
        logger.info(
            "Evaluating set %d/%d: %s (%d texts)",
            i + 1, len(eval_sets), config, len(texts),
        )

        # LLM/heuristic judge
        judgment = judge.judge(texts)
        judge_score = judgment["score"]

        # Automatic metrics
        auto_metrics = compute_metrics_for_set(texts)

        results.append({
            "config": config,
            "expected_diversity": es["expected_diversity"],
            "judge_score": judge_score,
            "judge_reasoning": judgment.get("reasoning", ""),
            "judge_method": judgment.get("method", "unknown"),
            "auto_metrics": auto_metrics,
            "n_texts": len(texts),
        })

    # Compute correlation between judge and each metric
    judge_scores = [r["judge_score"] for r in results]
    metric_names = sorted(results[0]["auto_metrics"].keys())

    correlations = {}
    for mn in metric_names:
        metric_vals = [r["auto_metrics"].get(mn, float("nan")) for r in results]
        valid = [
            (j, m) for j, m in zip(judge_scores, metric_vals)
            if not math.isnan(m)
        ]
        if len(valid) >= 3:
            js, ms = zip(*valid)
            tau = kendall_tau(list(js), list(ms))
            # For self_bleu, negate since lower = more diverse
            if mn == "self_bleu":
                tau = -tau
            correlations[mn] = tau
        else:
            correlations[mn] = float("nan")

    # Sort by correlation
    sorted_corr = sorted(
        correlations.items(), key=lambda x: abs(x[1]) if not math.isnan(x[1]) else 0,
        reverse=True,
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS: Judge-Metric Correlation (Kendall τ)")
    logger.info("=" * 60)
    logger.info(f"{'Metric':>25s} | {'τ with judge':>12s} | {'|τ|':>6s}")
    logger.info("-" * 50)
    for mn, tau in sorted_corr:
        logger.info(f"{mn:>25s} | {tau:>12.4f} | {abs(tau):>6.3f}")

    # Judge scores per set
    logger.info("\n" + "=" * 60)
    logger.info("JUDGE SCORES PER SET")
    logger.info("=" * 60)
    for r in sorted(results, key=lambda x: x["judge_score"]):
        logger.info(
            "  %s: judge=%.1f (%s) | %s",
            r["config"],
            r["judge_score"],
            r["expected_diversity"],
            r["judge_reasoning"][:60],
        )

    # Save results
    output = {
        "judge_method": judge_method,
        "n_sets": len(eval_sets),
        "correlations": correlations,
        "sorted_correlations": sorted_corr,
        "per_set_results": results,
    }
    out_path = RESULTS_DIR / "human_eval_proxy_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("\nSaved results to %s", out_path)

    return output


if __name__ == "__main__":
    run_human_eval_proxy()
