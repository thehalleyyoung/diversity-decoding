"""
Reference-based metrics for the Diversity Decoding Arena.

Comprehensive implementations of metrics that compare generated text against
reference texts. All algorithms implemented from scratch using numpy and
the standard library — no external NLP libraries required.

Metrics included:
    - BLEU (1–4 gram, corpus & sentence level)
    - ROUGE (N, L, W, S variants)
    - METEOR (exact + stem matching with chunk penalty)
    - CIDEr (TF-IDF weighted n-gram consensus)
    - BERTScore approximation (TF-IDF greedy matching)
    - ChromaticScore (diversity-aware reference metric)
    - ReferenceMetricSuite (unified runner & correlation analysis)
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WHITESPACE_RE = re.compile(r"\s+")


def tokenize(text: str, lower: bool = True) -> List[str]:
    """Simple whitespace + punctuation tokeniser."""
    if lower:
        text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.split()


# ---- simple suffix-stripping stemmer ------------------------------------

_SUFFIX_RULES: List[Tuple[str, str]] = [
    ("ational", "ate"),
    ("tional", "tion"),
    ("encies", "ence"),
    ("ances", "ance"),
    ("izers", "ize"),
    ("ising", "ise"),
    ("izing", "ize"),
    ("ating", "ate"),
    ("ition", "ite"),
    ("alism", "al"),
    ("iness", "y"),
    ("ously", "ous"),
    ("ively", "ive"),
    ("ement", "e"),
    ("ment", ""),
    ("ness", ""),
    ("ling", "le"),
    ("ting", "te"),
    ("able", ""),
    ("ible", ""),
    ("ies", "y"),
    ("ing", ""),
    ("eed", "ee"),
    ("ion", ""),
    ("ful", ""),
    ("ous", ""),
    ("ive", ""),
    ("ant", ""),
    ("ent", ""),
    ("ly", ""),
    ("ed", ""),
    ("er", ""),
    ("es", "e"),
    ("s", ""),
]


def stem(word: str) -> str:
    """Rule-based suffix stripping (lightweight Porter-like stemmer)."""
    if len(word) <= 3:
        return word
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix):
            candidate = word[: -len(suffix)] + replacement
            if len(candidate) >= 2:
                return candidate
    return word


# ---- n-gram helpers ------------------------------------------------------


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Return list of n-gram tuples from *tokens*."""
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def ngram_counts(tokens: List[str], n: int) -> Counter:
    """Return a Counter of n-gram tuples."""
    return Counter(ngrams(tokens, n))


def multi_ngram_counts(tokens: List[str], max_n: int = 4) -> Dict[int, Counter]:
    """Return {n: Counter} for n = 1 .. max_n."""
    return {n: ngram_counts(tokens, n) for n in range(1, max_n + 1)}


# ---- LCS helpers ---------------------------------------------------------


def lcs_length(x: Sequence, y: Sequence) -> int:
    """Length of the longest common subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(cur[j - 1], prev[j])
        prev = cur
    return prev[n]


def lcs_table(x: Sequence, y: Sequence) -> List[List[int]]:
    """Full LCS dynamic-programming table (m+1 × n+1)."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp


def _wlcs_table(x: Sequence, y: Sequence, weight_fn: Callable[[int], float]) -> Tuple[List[List[float]], List[List[int]]]:
    """Weighted LCS table returning (score, consecutive-match-length) matrices."""
    m, n = len(x), len(y)
    c = [[0.0] * (n + 1) for _ in range(m + 1)]
    w = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                k = w[i - 1][j - 1] + 1
                c[i][j] = c[i - 1][j - 1] + weight_fn(k) - weight_fn(k - 1)
                w[i][j] = k
            else:
                if c[i - 1][j] >= c[i][j - 1]:
                    c[i][j] = c[i - 1][j]
                    w[i][j] = 0
                else:
                    c[i][j] = c[i][j - 1]
                    w[i][j] = 0
    return c, w


# ---- skip-bigrams -------------------------------------------------------


def skip_bigrams(tokens: List[str], max_skip: int = -1) -> Counter:
    """Generate skip-bigram counts.

    If *max_skip* < 0 every pair is considered (no skip limit).
    """
    counts: Counter = Counter()
    n = len(tokens)
    for i in range(n):
        upper = n if max_skip < 0 else min(n, i + max_skip + 2)
        for j in range(i + 1, upper):
            counts[(tokens[i], tokens[j])] += 1
    return counts


# ---- TF-IDF helpers ------------------------------------------------------


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Term-frequency (raw count / total)."""
    total = len(tokens)
    if total == 0:
        return {}
    counts = Counter(tokens)
    return {t: c / total for t, c in counts.items()}


def compute_idf(documents: List[List[str]], smooth: bool = True) -> Dict[str, float]:
    """Inverse document frequency across a corpus.

    Uses log( (N+1) / (df+1) ) + 1 when *smooth* is True for stability.
    """
    n_docs = len(documents)
    df: Counter = Counter()
    for doc in documents:
        for tok in set(doc):
            df[tok] += 1
    idf: Dict[str, float] = {}
    for tok, doc_freq in df.items():
        if smooth:
            idf[tok] = math.log((n_docs + 1) / (doc_freq + 1)) + 1.0
        else:
            idf[tok] = math.log(n_docs / doc_freq) if doc_freq > 0 else 0.0
    return idf


def tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """TF-IDF vector for a single document."""
    tf = compute_tf(tokens)
    return {t: tf_val * idf.get(t, 0.0) for t, tf_val in tf.items()}


def tfidf_matrix(
    documents: List[List[str]], idf: Dict[str, float], vocab: List[str],
) -> np.ndarray:
    """Build a dense (n_docs × vocab_size) TF-IDF matrix."""
    tok2idx = {t: i for i, t in enumerate(vocab)}
    mat = np.zeros((len(documents), len(vocab)), dtype=np.float64)
    for d, doc in enumerate(documents):
        vec = tfidf_vector(doc, idf)
        for tok, val in vec.items():
            if tok in tok2idx:
                mat[d, tok2idx[tok]] = val
    return mat


# =========================================================================
# 1. BLEUScore
# =========================================================================


class BLEUScore:
    """BLEU (Bilingual Evaluation Understudy) scorer.

    Supports corpus-level and sentence-level BLEU with three smoothing
    strategies:
        - ``chen_cherry``: method 1 from Chen & Cherry (2014)
        - ``floor``:       replace zero counts with a small floor value
        - ``add_epsilon``: add a small epsilon to both numerator and denominator
    """

    SMOOTHING_METHODS = ("none", "chen_cherry", "floor", "add_epsilon")

    def __init__(
        self,
        max_n: int = 4,
        weights: Optional[List[float]] = None,
        smoothing: str = "chen_cherry",
        floor_value: float = 0.01,
        epsilon: float = 0.1,
    ) -> None:
        self.max_n = max_n
        self.weights = weights or [1.0 / max_n] * max_n
        if len(self.weights) != max_n:
            raise ValueError("len(weights) must equal max_n")
        if smoothing not in self.SMOOTHING_METHODS:
            raise ValueError(f"Unknown smoothing method: {smoothing}")
        self.smoothing = smoothing
        self.floor_value = floor_value
        self.epsilon = epsilon

    # -- modified precision ------------------------------------------------

    @staticmethod
    def _modified_precision(
        hypothesis_tokens: List[str],
        references_tokens: List[List[str]],
        n: int,
    ) -> Tuple[int, int]:
        """Clipped n-gram precision counts (numerator, denominator)."""
        hyp_counts = ngram_counts(hypothesis_tokens, n)
        if not hyp_counts:
            return 0, 0

        max_ref_counts: Counter = Counter()
        for ref_tokens in references_tokens:
            ref_c = ngram_counts(ref_tokens, n)
            for ng, cnt in ref_c.items():
                max_ref_counts[ng] = max(max_ref_counts[ng], cnt)

        clipped = {ng: min(cnt, max_ref_counts.get(ng, 0)) for ng, cnt in hyp_counts.items()}
        numerator = sum(clipped.values())
        denominator = sum(hyp_counts.values())
        return numerator, denominator

    # -- brevity penalty ---------------------------------------------------

    @staticmethod
    def _brevity_penalty(hyp_len: int, closest_ref_len: int) -> float:
        if hyp_len == 0:
            return 0.0
        ratio = closest_ref_len / hyp_len
        if ratio <= 1.0:
            return 1.0
        return math.exp(1.0 - ratio)

    @staticmethod
    def _closest_ref_length(hyp_len: int, ref_lens: List[int]) -> int:
        """Reference length closest to hypothesis length."""
        return min(ref_lens, key=lambda r: (abs(r - hyp_len), r))

    # -- smoothing ---------------------------------------------------------

    def _apply_smoothing(
        self, precisions: List[Tuple[int, int]],
    ) -> List[float]:
        """Return smoothed precision values for each n-gram order."""
        smoothed: List[float] = []
        for idx, (num, den) in enumerate(precisions):
            n = idx + 1
            if den == 0:
                smoothed.append(0.0)
                continue
            if num == 0:
                if self.smoothing == "none":
                    smoothed.append(0.0)
                elif self.smoothing == "chen_cherry":
                    # Incrementally increasing pseudo-count: 1/(2^n)
                    smoothed.append(1.0 / (2 ** n) / den)
                elif self.smoothing == "floor":
                    smoothed.append(self.floor_value / den)
                elif self.smoothing == "add_epsilon":
                    smoothed.append(self.epsilon / (den + self.epsilon))
                else:
                    smoothed.append(0.0)
            else:
                if self.smoothing == "add_epsilon":
                    smoothed.append((num + self.epsilon) / (den + self.epsilon))
                else:
                    smoothed.append(num / den)
        return smoothed

    # -- sentence BLEU -----------------------------------------------------

    def sentence_bleu(
        self,
        hypothesis: str,
        references: List[str],
    ) -> float:
        """Compute sentence-level BLEU."""
        hyp_tokens = tokenize(hypothesis)
        refs_tokens = [tokenize(r) for r in references]
        if not hyp_tokens:
            return 0.0

        precisions: List[Tuple[int, int]] = []
        for n in range(1, self.max_n + 1):
            precisions.append(self._modified_precision(hyp_tokens, refs_tokens, n))

        smoothed = self._apply_smoothing(precisions)
        # Weighted geometric mean in log space
        log_avg = 0.0
        for w, p in zip(self.weights, smoothed):
            if p <= 0:
                return 0.0
            log_avg += w * math.log(p)

        ref_lens = [len(r) for r in refs_tokens]
        bp = self._brevity_penalty(
            len(hyp_tokens), self._closest_ref_length(len(hyp_tokens), ref_lens),
        )
        return bp * math.exp(log_avg)

    # -- corpus BLEU -------------------------------------------------------

    def corpus_bleu(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> float:
        """Compute corpus-level BLEU (micro-averaged precision)."""
        if len(hypotheses) != len(references):
            raise ValueError("hypotheses and references must have equal length")
        if not hypotheses:
            return 0.0

        total_num = [0] * self.max_n
        total_den = [0] * self.max_n
        total_hyp_len = 0
        total_ref_len = 0

        for hyp_str, refs_str in zip(hypotheses, references):
            hyp_tokens = tokenize(hyp_str)
            refs_tokens = [tokenize(r) for r in refs_str]
            total_hyp_len += len(hyp_tokens)
            ref_lens = [len(r) for r in refs_tokens]
            if ref_lens:
                total_ref_len += self._closest_ref_length(len(hyp_tokens), ref_lens)

            for n in range(1, self.max_n + 1):
                num, den = self._modified_precision(hyp_tokens, refs_tokens, n)
                total_num[n - 1] += num
                total_den[n - 1] += den

        precisions_raw = list(zip(total_num, total_den))
        smoothed = self._apply_smoothing(precisions_raw)

        log_avg = 0.0
        for w, p in zip(self.weights, smoothed):
            if p <= 0:
                return 0.0
            log_avg += w * math.log(p)

        bp = self._brevity_penalty(total_hyp_len, total_ref_len)
        return bp * math.exp(log_avg)

    # -- convenience -------------------------------------------------------

    def compute(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> float:
        """Compute corpus-level BLEU (primary interface)."""
        return self.corpus_bleu(hypotheses, references)

    def compute_detailed(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict[str, Any]:
        """Return per-sentence and corpus BLEU with component details."""
        sentence_scores = [
            self.sentence_bleu(h, refs) for h, refs in zip(hypotheses, references)
        ]
        corpus = self.corpus_bleu(hypotheses, references)
        return {
            "corpus_bleu": corpus,
            "sentence_bleu": sentence_scores,
            "mean_sentence_bleu": float(np.mean(sentence_scores)) if sentence_scores else 0.0,
            "max_n": self.max_n,
            "smoothing": self.smoothing,
        }


# =========================================================================
# 2. ROUGEScore
# =========================================================================


class ROUGEScore:
    """ROUGE (Recall-Oriented Understudy for Gisting Evaluation).

    Implements ROUGE-N (N=1,2,3), ROUGE-L, ROUGE-W, ROUGE-S.
    """

    def __init__(
        self,
        max_n: int = 3,
        rouge_w_weight: float = 1.2,
        skip_bigram_max_skip: int = -1,
        alpha: float = 0.5,
    ) -> None:
        self.max_n = max_n
        self.rouge_w_weight = rouge_w_weight
        self.skip_bigram_max_skip = skip_bigram_max_skip
        # alpha: balance between precision and recall in F-measure
        # alpha=0.5 -> standard F1; alpha closer to 0 favours recall
        self.alpha = alpha

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _f_measure(precision: float, recall: float, alpha: float = 0.5) -> float:
        if precision == 0.0 and recall == 0.0:
            return 0.0
        denom = alpha * precision + (1.0 - alpha) * recall
        if denom == 0.0:
            return 0.0
        return (precision * recall) / denom

    # -- ROUGE-N -----------------------------------------------------------

    def _rouge_n_single(
        self,
        hyp_tokens: List[str],
        ref_tokens: List[str],
        n: int,
    ) -> Dict[str, float]:
        hyp_ng = ngram_counts(hyp_tokens, n)
        ref_ng = ngram_counts(ref_tokens, n)

        overlap = 0
        for ng, cnt in ref_ng.items():
            overlap += min(cnt, hyp_ng.get(ng, 0))

        total_ref = sum(ref_ng.values())
        total_hyp = sum(hyp_ng.values())

        recall = overlap / total_ref if total_ref > 0 else 0.0
        precision = overlap / total_hyp if total_hyp > 0 else 0.0
        f1 = self._f_measure(precision, recall, self.alpha)
        return {"precision": precision, "recall": recall, "f1": f1}

    def rouge_n(
        self,
        hypothesis: str,
        references: List[str],
        n: int,
    ) -> Dict[str, float]:
        """ROUGE-N against multiple references (best score kept)."""
        hyp_tokens = tokenize(hypothesis)
        best: Dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for ref in references:
            ref_tokens = tokenize(ref)
            scores = self._rouge_n_single(hyp_tokens, ref_tokens, n)
            if scores["f1"] > best["f1"]:
                best = scores
        return best

    # -- ROUGE-L -----------------------------------------------------------

    def _rouge_l_single(
        self,
        hyp_tokens: List[str],
        ref_tokens: List[str],
    ) -> Dict[str, float]:
        lcs_len = lcs_length(ref_tokens, hyp_tokens)
        m = len(ref_tokens)
        n = len(hyp_tokens)
        recall = lcs_len / m if m > 0 else 0.0
        precision = lcs_len / n if n > 0 else 0.0
        f1 = self._f_measure(precision, recall, self.alpha)
        return {"precision": precision, "recall": recall, "f1": f1}

    def rouge_l(
        self,
        hypothesis: str,
        references: List[str],
    ) -> Dict[str, float]:
        hyp_tokens = tokenize(hypothesis)
        best: Dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for ref in references:
            ref_tokens = tokenize(ref)
            scores = self._rouge_l_single(hyp_tokens, ref_tokens)
            if scores["f1"] > best["f1"]:
                best = scores
        return best

    # -- ROUGE-W (weighted LCS) -------------------------------------------

    def _rouge_w_single(
        self,
        hyp_tokens: List[str],
        ref_tokens: List[str],
    ) -> Dict[str, float]:
        w = self.rouge_w_weight

        def weight_fn(k: int) -> float:
            return k ** w

        def inv_weight_fn(val: float) -> float:
            if val <= 0:
                return 0.0
            return val ** (1.0 / w)

        c, _ = _wlcs_table(ref_tokens, hyp_tokens, weight_fn)
        wlcs = c[len(ref_tokens)][len(hyp_tokens)]

        m = len(ref_tokens)
        n = len(hyp_tokens)

        recall = inv_weight_fn(wlcs / weight_fn(m)) if m > 0 else 0.0
        precision = inv_weight_fn(wlcs / weight_fn(n)) if n > 0 else 0.0
        f1 = self._f_measure(precision, recall, self.alpha)
        return {"precision": precision, "recall": recall, "f1": f1}

    def rouge_w(
        self,
        hypothesis: str,
        references: List[str],
    ) -> Dict[str, float]:
        hyp_tokens = tokenize(hypothesis)
        best: Dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for ref in references:
            ref_tokens = tokenize(ref)
            scores = self._rouge_w_single(hyp_tokens, ref_tokens)
            if scores["f1"] > best["f1"]:
                best = scores
        return best

    # -- ROUGE-S (skip-bigram) ---------------------------------------------

    def _rouge_s_single(
        self,
        hyp_tokens: List[str],
        ref_tokens: List[str],
    ) -> Dict[str, float]:
        hyp_sb = skip_bigrams(hyp_tokens, self.skip_bigram_max_skip)
        ref_sb = skip_bigrams(ref_tokens, self.skip_bigram_max_skip)

        overlap = 0
        for bg, cnt in ref_sb.items():
            overlap += min(cnt, hyp_sb.get(bg, 0))

        total_ref = sum(ref_sb.values())
        total_hyp = sum(hyp_sb.values())

        recall = overlap / total_ref if total_ref > 0 else 0.0
        precision = overlap / total_hyp if total_hyp > 0 else 0.0
        f1 = self._f_measure(precision, recall, self.alpha)
        return {"precision": precision, "recall": recall, "f1": f1}

    def rouge_s(
        self,
        hypothesis: str,
        references: List[str],
    ) -> Dict[str, float]:
        hyp_tokens = tokenize(hypothesis)
        best: Dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for ref in references:
            ref_tokens = tokenize(ref)
            scores = self._rouge_s_single(hyp_tokens, ref_tokens)
            if scores["f1"] > best["f1"]:
                best = scores
        return best

    # -- corpus helpers ----------------------------------------------------

    def _corpus_aggregate(
        self,
        score_fn: Callable,
        hypotheses: List[str],
        references: List[List[str]],
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Macro-average a per-sentence scoring function across a corpus."""
        all_p, all_r, all_f = [], [], []
        for hyp, refs in zip(hypotheses, references):
            s = score_fn(hyp, refs, **kwargs)
            all_p.append(s["precision"])
            all_r.append(s["recall"])
            all_f.append(s["f1"])
        return {
            "precision": float(np.mean(all_p)) if all_p else 0.0,
            "recall": float(np.mean(all_r)) if all_r else 0.0,
            "f1": float(np.mean(all_f)) if all_f else 0.0,
        }

    # -- main interface ----------------------------------------------------

    def compute(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute all ROUGE variants and return a nested dict."""
        if len(hypotheses) != len(references):
            raise ValueError("hypotheses and references must have equal length")
        result: Dict[str, Dict[str, float]] = {}
        for n in range(1, self.max_n + 1):
            result[f"rouge-{n}"] = self._corpus_aggregate(
                self.rouge_n, hypotheses, references, n=n,
            )
        result["rouge-l"] = self._corpus_aggregate(
            self.rouge_l, hypotheses, references,
        )
        result["rouge-w"] = self._corpus_aggregate(
            self.rouge_w, hypotheses, references,
        )
        result["rouge-s"] = self._corpus_aggregate(
            self.rouge_s, hypotheses, references,
        )
        return result

    def compute_sentence(
        self, hypothesis: str, references: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """All ROUGE variants for a single sentence."""
        result: Dict[str, Dict[str, float]] = {}
        for n in range(1, self.max_n + 1):
            result[f"rouge-{n}"] = self.rouge_n(hypothesis, references, n)
        result["rouge-l"] = self.rouge_l(hypothesis, references)
        result["rouge-w"] = self.rouge_w(hypothesis, references)
        result["rouge-s"] = self.rouge_s(hypothesis, references)
        return result


# =========================================================================
# 3. METEORScore
# =========================================================================


class METEORScore:
    """METEOR (Metric for Evaluation of Translation with Explicit ORdering).

    Implements exact matching, stem matching, a simple chunk-based penalty,
    and the standard weighted harmonic mean.
    """

    def __init__(
        self,
        alpha: float = 0.9,
        beta: float = 3.0,
        gamma: float = 0.5,
        exact_weight: float = 1.0,
        stem_weight: float = 0.6,
    ) -> None:
        # alpha: weight for recall in harmonic mean
        self.alpha = alpha
        # beta: exponent in penalty
        self.beta = beta
        # gamma: penalty weight
        self.gamma = gamma
        self.exact_weight = exact_weight
        self.stem_weight = stem_weight

    # -- alignment ---------------------------------------------------------

    @staticmethod
    def _exact_align(
        hyp_tokens: List[str], ref_tokens: List[str],
    ) -> List[Tuple[int, int]]:
        """Greedily align by exact match (left-to-right)."""
        used_ref: Set[int] = set()
        alignment: List[Tuple[int, int]] = []
        for h_idx, h_tok in enumerate(hyp_tokens):
            for r_idx, r_tok in enumerate(ref_tokens):
                if r_idx not in used_ref and h_tok == r_tok:
                    alignment.append((h_idx, r_idx))
                    used_ref.add(r_idx)
                    break
        return alignment

    @staticmethod
    def _stem_align(
        hyp_tokens: List[str],
        ref_tokens: List[str],
        existing: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Additional alignments via stem matching (not already exact-aligned)."""
        used_hyp = {h for h, _ in existing}
        used_ref = {r for _, r in existing}
        extra: List[Tuple[int, int]] = []
        for h_idx, h_tok in enumerate(hyp_tokens):
            if h_idx in used_hyp:
                continue
            h_stem = stem(h_tok)
            for r_idx, r_tok in enumerate(ref_tokens):
                if r_idx in used_ref:
                    continue
                if h_stem == stem(r_tok):
                    extra.append((h_idx, r_idx))
                    used_hyp.add(h_idx)
                    used_ref.add(r_idx)
                    break
        return extra

    # -- chunks / penalty --------------------------------------------------

    @staticmethod
    def _count_chunks(alignment: List[Tuple[int, int]]) -> int:
        """Count the number of contiguous chunks in the alignment.

        A chunk is a maximal sequence of aligned unigrams that are adjacent
        in both hypothesis and reference.
        """
        if not alignment:
            return 0
        # Sort by hypothesis position
        sorted_align = sorted(alignment, key=lambda x: x[0])
        chunks = 1
        for i in range(1, len(sorted_align)):
            prev_h, prev_r = sorted_align[i - 1]
            cur_h, cur_r = sorted_align[i]
            if cur_h != prev_h + 1 or cur_r != prev_r + 1:
                chunks += 1
        return chunks

    # -- scoring -----------------------------------------------------------

    def _score_pair(
        self, hyp_tokens: List[str], ref_tokens: List[str],
    ) -> Dict[str, float]:
        exact_align = self._exact_align(hyp_tokens, ref_tokens)
        stem_extra = self._stem_align(hyp_tokens, ref_tokens, exact_align)

        # Weighted matches
        matches = (
            len(exact_align) * self.exact_weight
            + len(stem_extra) * self.stem_weight
        )

        total_matches_count = len(exact_align) + len(stem_extra)

        hyp_len = len(hyp_tokens)
        ref_len = len(ref_tokens)
        precision = matches / hyp_len if hyp_len > 0 else 0.0
        recall = matches / ref_len if ref_len > 0 else 0.0

        if precision == 0.0 and recall == 0.0:
            f_mean = 0.0
        else:
            f_mean = (precision * recall) / (
                self.alpha * precision + (1.0 - self.alpha) * recall
            )

        # Chunk penalty
        full_alignment = exact_align + stem_extra
        chunks = self._count_chunks(full_alignment)
        if total_matches_count > 0:
            frag = chunks / total_matches_count
        else:
            frag = 0.0
        penalty = self.gamma * (frag ** self.beta)

        score = f_mean * (1.0 - penalty)
        return {
            "score": max(score, 0.0),
            "precision": precision,
            "recall": recall,
            "f_mean": f_mean,
            "penalty": penalty,
            "chunks": chunks,
            "matches": total_matches_count,
        }

    def sentence_meteor(
        self, hypothesis: str, references: List[str],
    ) -> Dict[str, float]:
        """Score a single hypothesis against multiple references (best kept)."""
        hyp_tokens = tokenize(hypothesis)
        best: Dict[str, float] = {"score": 0.0}
        for ref in references:
            ref_tokens = tokenize(ref)
            result = self._score_pair(hyp_tokens, ref_tokens)
            if result["score"] > best.get("score", 0.0):
                best = result
        return best

    def compute(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> float:
        """Corpus-level METEOR (macro average)."""
        if len(hypotheses) != len(references):
            raise ValueError("hypotheses and references must have equal length")
        if not hypotheses:
            return 0.0
        scores = [
            self.sentence_meteor(h, refs)["score"]
            for h, refs in zip(hypotheses, references)
        ]
        return float(np.mean(scores))

    def compute_detailed(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict[str, Any]:
        sentence_scores = [
            self.sentence_meteor(h, refs) for h, refs in zip(hypotheses, references)
        ]
        score_vals = [s["score"] for s in sentence_scores]
        return {
            "corpus_meteor": float(np.mean(score_vals)) if score_vals else 0.0,
            "sentence_scores": sentence_scores,
            "std": float(np.std(score_vals)) if score_vals else 0.0,
        }


# =========================================================================
# 4. CIDErScore
# =========================================================================


class CIDErScore:
    """CIDEr (Consensus-based Image Description Evaluation).

    Uses TF-IDF weighted n-gram similarity measured via cosine, averaged
    over n-gram orders 1..max_n.
    """

    def __init__(self, max_n: int = 4, sigma: float = 6.0) -> None:
        self.max_n = max_n
        self.sigma = sigma  # Gaussian length penalty sigma

    # -- build document-frequency from references --------------------------

    @staticmethod
    def _doc_freq(
        references: List[List[str]], n: int,
    ) -> Counter:
        """Document frequency of each n-gram across references."""
        df: Counter = Counter()
        for refs in references:
            seen: Set[Tuple[str, ...]] = set()
            for ref in refs:
                ref_tokens = tokenize(ref)
                for ng in ngrams(ref_tokens, n):
                    seen.add(ng)
            for ng in seen:
                df[ng] += 1
        return df

    # -- TF-IDF vectors for a single sentence ------------------------------

    @staticmethod
    def _tfidf_vec(
        tokens: List[str],
        n: int,
        df: Counter,
        num_docs: int,
    ) -> Counter:
        counts = ngram_counts(tokens, n)
        total = sum(counts.values())
        if total == 0:
            return Counter()
        vec: Counter = Counter()
        for ng, cnt in counts.items():
            tf = cnt / total
            idf_val = math.log(max((num_docs) / (1.0 + df.get(ng, 0)), 1.0))
            vec[ng] = tf * idf_val
        return vec

    @staticmethod
    def _cosine(a: Counter, b: Counter) -> float:
        keys = set(a.keys()) | set(b.keys())
        if not keys:
            return 0.0
        dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
        norm_a = math.sqrt(sum(v ** 2 for v in a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _length_penalty(self, hyp_len: int, ref_len: int) -> float:
        """Gaussian length penalty."""
        diff = hyp_len - ref_len
        return math.exp(-(diff ** 2) / (2 * self.sigma ** 2))

    # -- per-sentence CIDEr-n ----------------------------------------------

    def _cider_n(
        self,
        hyp_tokens: List[str],
        refs_tokens: List[List[str]],
        n: int,
        df: Counter,
        num_docs: int,
    ) -> float:
        hyp_vec = self._tfidf_vec(hyp_tokens, n, df, num_docs)
        scores = []
        for ref_tokens in refs_tokens:
            ref_vec = self._tfidf_vec(ref_tokens, n, df, num_docs)
            cos = self._cosine(hyp_vec, ref_vec)
            lp = self._length_penalty(len(hyp_tokens), len(ref_tokens))
            scores.append(cos * lp)
        return float(np.mean(scores)) if scores else 0.0

    # -- main interface ----------------------------------------------------

    def compute(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> float:
        """Corpus-level CIDEr score (average over sentences and n-gram orders)."""
        if len(hypotheses) != len(references):
            raise ValueError("hypotheses and references must have equal length")
        if not hypotheses:
            return 0.0
        num_docs = len(references)

        # Pre-compute document frequencies for each n
        dfs: Dict[int, Counter] = {}
        for n in range(1, self.max_n + 1):
            dfs[n] = self._doc_freq(references, n)

        sentence_scores: List[float] = []
        for hyp_str, refs_str in zip(hypotheses, references):
            hyp_tokens = tokenize(hyp_str)
            refs_tokens = [tokenize(r) for r in refs_str]
            n_scores = []
            for n in range(1, self.max_n + 1):
                n_scores.append(
                    self._cider_n(hyp_tokens, refs_tokens, n, dfs[n], num_docs)
                )
            # CIDEr: equal weight across n-gram orders, scaled by 10
            sentence_scores.append(10.0 * float(np.mean(n_scores)))

        return float(np.mean(sentence_scores))

    def compute_detailed(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict[str, Any]:
        if len(hypotheses) != len(references):
            raise ValueError("hypotheses and references must have equal length")
        if not hypotheses:
            return {"corpus_cider": 0.0, "sentence_scores": [], "per_n": {}}
        num_docs = len(references)

        dfs: Dict[int, Counter] = {}
        for n in range(1, self.max_n + 1):
            dfs[n] = self._doc_freq(references, n)

        per_n_scores: Dict[int, List[float]] = {n: [] for n in range(1, self.max_n + 1)}
        sentence_scores: List[float] = []

        for hyp_str, refs_str in zip(hypotheses, references):
            hyp_tokens = tokenize(hyp_str)
            refs_tokens = [tokenize(r) for r in refs_str]
            n_scores = []
            for n in range(1, self.max_n + 1):
                s = self._cider_n(hyp_tokens, refs_tokens, n, dfs[n], num_docs)
                per_n_scores[n].append(s)
                n_scores.append(s)
            sentence_scores.append(10.0 * float(np.mean(n_scores)))

        return {
            "corpus_cider": float(np.mean(sentence_scores)),
            "sentence_scores": sentence_scores,
            "per_n": {
                f"cider-{n}": float(np.mean(vals)) if vals else 0.0
                for n, vals in per_n_scores.items()
            },
        }


# =========================================================================
# 5. BERTScoreApproximation
# =========================================================================


class BERTScoreApproximation:
    """Approximate BERTScore using TF-IDF token embeddings.

    Real BERTScore uses contextual embeddings from a pre-trained
    transformer. Here we approximate by constructing a high-dimensional
    TF-IDF vector for each token (based on the character n-grams of
    the word and its corpus IDF) and performing greedy matching.
    """

    def __init__(
        self,
        char_ngram_range: Tuple[int, int] = (3, 6),
        idf_weighting: bool = True,
    ) -> None:
        self.char_ngram_range = char_ngram_range
        self.idf_weighting = idf_weighting
        self._idf: Optional[Dict[str, float]] = None
        self._vocab: Optional[List[str]] = None
        self._char_vocab: Optional[Dict[str, int]] = None

    # -- character n-gram embedding ----------------------------------------

    def _char_ngrams(self, word: str) -> List[str]:
        padded = f"<{word}>"
        cng: List[str] = []
        lo, hi = self.char_ngram_range
        for n in range(lo, hi + 1):
            for i in range(len(padded) - n + 1):
                cng.append(padded[i: i + n])
        return cng

    def _build_char_vocab(self, all_tokens: Set[str]) -> Dict[str, int]:
        char_set: Set[str] = set()
        for tok in all_tokens:
            char_set.update(self._char_ngrams(tok))
        return {c: i for i, c in enumerate(sorted(char_set))}

    def _token_embedding(self, token: str) -> np.ndarray:
        """Sparse-to-dense character n-gram vector for a token."""
        assert self._char_vocab is not None
        dim = len(self._char_vocab)
        vec = np.zeros(dim, dtype=np.float64)
        for cng in self._char_ngrams(token):
            idx = self._char_vocab.get(cng)
            if idx is not None:
                vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> "BERTScoreApproximation":
        """Compute IDF and char-ngram vocabulary from the evaluation corpus."""
        all_docs: List[List[str]] = []
        all_token_set: Set[str] = set()
        for hyp in hypotheses:
            tokens = tokenize(hyp)
            all_docs.append(tokens)
            all_token_set.update(tokens)
        for refs in references:
            for ref in refs:
                tokens = tokenize(ref)
                all_docs.append(tokens)
                all_token_set.update(tokens)

        self._idf = compute_idf(all_docs, smooth=True)
        self._char_vocab = self._build_char_vocab(all_token_set)
        return self

    # -- greedy matching ---------------------------------------------------

    def _embed_tokens(self, tokens: List[str]) -> np.ndarray:
        """Matrix of shape (n_tokens, dim)."""
        if not tokens:
            return np.zeros((0, len(self._char_vocab) if self._char_vocab else 1))
        vecs = [self._token_embedding(t) for t in tokens]
        return np.stack(vecs)

    @staticmethod
    def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pairwise cosine similarity (len_a × len_b)."""
        if a.shape[0] == 0 or b.shape[0] == 0:
            return np.zeros((a.shape[0], b.shape[0]))
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True)
        norm_a = np.where(norm_a == 0, 1.0, norm_a)
        norm_b = np.where(norm_b == 0, 1.0, norm_b)
        return (a / norm_a) @ (b / norm_b).T

    def _idf_weights(self, tokens: List[str]) -> np.ndarray:
        if self._idf is None:
            return np.ones(len(tokens))
        return np.array([self._idf.get(t, 1.0) for t in tokens])

    def _greedy_match(
        self,
        hyp_tokens: List[str],
        ref_tokens: List[str],
    ) -> Tuple[float, float, float]:
        """Return (precision, recall, F1) via greedy cosine matching."""
        if not hyp_tokens or not ref_tokens:
            return (0.0, 0.0, 0.0)

        hyp_emb = self._embed_tokens(hyp_tokens)
        ref_emb = self._embed_tokens(ref_tokens)
        sim = self._cosine_matrix(hyp_emb, ref_emb)

        if self.idf_weighting:
            hyp_w = self._idf_weights(hyp_tokens)
            ref_w = self._idf_weights(ref_tokens)
        else:
            hyp_w = np.ones(len(hyp_tokens))
            ref_w = np.ones(len(ref_tokens))

        # Precision: for each hyp token, max similarity to any ref token
        max_sim_hyp = sim.max(axis=1)
        precision = float(np.sum(max_sim_hyp * hyp_w) / np.sum(hyp_w))

        # Recall: for each ref token, max similarity to any hyp token
        max_sim_ref = sim.max(axis=0)
        recall = float(np.sum(max_sim_ref * ref_w) / np.sum(ref_w))

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    # -- main interface ----------------------------------------------------

    def score_pair(
        self, hypothesis: str, references: List[str],
    ) -> Dict[str, float]:
        hyp_tokens = tokenize(hypothesis)
        best_f1 = -1.0
        best_result: Dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for ref in references:
            ref_tokens = tokenize(ref)
            p, r, f = self._greedy_match(hyp_tokens, ref_tokens)
            if f > best_f1:
                best_f1 = f
                best_result = {"precision": p, "recall": r, "f1": f}
        return best_result

    def compute(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """Corpus-level approximate BERTScore."""
        if len(hypotheses) != len(references):
            raise ValueError("hypotheses and references must have equal length")
        self.fit(hypotheses, references)

        all_p, all_r, all_f = [], [], []
        for hyp, refs in zip(hypotheses, references):
            s = self.score_pair(hyp, refs)
            all_p.append(s["precision"])
            all_r.append(s["recall"])
            all_f.append(s["f1"])

        return {
            "precision": float(np.mean(all_p)) if all_p else 0.0,
            "recall": float(np.mean(all_r)) if all_r else 0.0,
            "f1": float(np.mean(all_f)) if all_f else 0.0,
        }


# =========================================================================
# 6. ChromaticScore
# =========================================================================


class ChromaticScore:
    """Diversity-aware reference metric.

    Measures both *accuracy* (how well hypotheses match references) and
    *variety* (how diverse the hypotheses are relative to the reference set).

    The final score blends:
        chromatic = lambda_ * accuracy + (1 - lambda_) * variety
    """

    def __init__(
        self,
        max_n: int = 2,
        lambda_: float = 0.6,
        diversity_method: str = "type_token",
    ) -> None:
        self.max_n = max_n
        self.lambda_ = lambda_
        self.diversity_method = diversity_method

    # -- accuracy component ------------------------------------------------

    def _accuracy(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> float:
        """Average best-match ROUGE-L F1 across hypotheses."""
        rouge = ROUGEScore(max_n=1)
        scores = []
        for hyp, refs in zip(hypotheses, references):
            s = rouge.rouge_l(hyp, refs)
            scores.append(s["f1"])
        return float(np.mean(scores)) if scores else 0.0

    # -- variety component -------------------------------------------------

    def _type_token_diversity(self, hypotheses: List[str]) -> float:
        """Type-token ratio across all hypothesis n-grams."""
        all_ngrams: List[Tuple[str, ...]] = []
        for hyp in hypotheses:
            tokens = tokenize(hyp)
            for n in range(1, self.max_n + 1):
                all_ngrams.extend(ngrams(tokens, n))
        if not all_ngrams:
            return 0.0
        return len(set(all_ngrams)) / len(all_ngrams)

    def _pairwise_diversity(self, hypotheses: List[str]) -> float:
        """1 - average pairwise Jaccard similarity of unigram sets."""
        if len(hypotheses) < 2:
            return 0.0
        token_sets = [set(tokenize(h)) for h in hypotheses]
        sims = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                inter = len(token_sets[i] & token_sets[j])
                union = len(token_sets[i] | token_sets[j])
                sims.append(inter / union if union > 0 else 0.0)
        return 1.0 - float(np.mean(sims))

    def _self_bleu_diversity(self, hypotheses: List[str]) -> float:
        """1 - average self-BLEU (lower self-BLEU = higher diversity)."""
        if len(hypotheses) < 2:
            return 0.0
        bleu = BLEUScore(max_n=self.max_n, smoothing="chen_cherry")
        scores = []
        for i, hyp in enumerate(hypotheses):
            refs = [h for j, h in enumerate(hypotheses) if j != i]
            scores.append(bleu.sentence_bleu(hyp, refs))
        return 1.0 - float(np.mean(scores))

    def _variety(self, hypotheses: List[str]) -> float:
        if self.diversity_method == "type_token":
            return self._type_token_diversity(hypotheses)
        elif self.diversity_method == "pairwise":
            return self._pairwise_diversity(hypotheses)
        elif self.diversity_method == "self_bleu":
            return self._self_bleu_diversity(hypotheses)
        raise ValueError(f"Unknown diversity method: {self.diversity_method}")

    # -- reference coverage ------------------------------------------------

    def _reference_coverage(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> float:
        """Fraction of unique reference n-grams covered by hypotheses."""
        ref_ngrams: Set[Tuple[str, ...]] = set()
        for refs in references:
            for ref in refs:
                tokens = tokenize(ref)
                for n in range(1, self.max_n + 1):
                    ref_ngrams.update(ngrams(tokens, n))

        hyp_ngrams: Set[Tuple[str, ...]] = set()
        for hyp in hypotheses:
            tokens = tokenize(hyp)
            for n in range(1, self.max_n + 1):
                hyp_ngrams.update(ngrams(tokens, n))

        if not ref_ngrams:
            return 0.0
        return len(ref_ngrams & hyp_ngrams) / len(ref_ngrams)

    # -- main interface ----------------------------------------------------

    def compute(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        if len(hypotheses) != len(references):
            raise ValueError("hypotheses and references must have equal length")
        acc = self._accuracy(hypotheses, references)
        var = self._variety(hypotheses)
        cov = self._reference_coverage(hypotheses, references)
        chromatic = self.lambda_ * acc + (1.0 - self.lambda_) * var
        return {
            "chromatic": chromatic,
            "accuracy": acc,
            "variety": var,
            "coverage": cov,
            "lambda": self.lambda_,
        }


# =========================================================================
# 7. ReferenceMetricSuite
# =========================================================================


class ReferenceMetricSuite:
    """Compute all reference-based metrics in one pass with optional
    correlation analysis between metric scores."""

    def __init__(
        self,
        bleu_kwargs: Optional[Dict[str, Any]] = None,
        rouge_kwargs: Optional[Dict[str, Any]] = None,
        meteor_kwargs: Optional[Dict[str, Any]] = None,
        cider_kwargs: Optional[Dict[str, Any]] = None,
        bertscore_kwargs: Optional[Dict[str, Any]] = None,
        chromatic_kwargs: Optional[Dict[str, Any]] = None,
        enabled_metrics: Optional[List[str]] = None,
    ) -> None:
        self.bleu = BLEUScore(**(bleu_kwargs or {}))
        self.rouge = ROUGEScore(**(rouge_kwargs or {}))
        self.meteor = METEORScore(**(meteor_kwargs or {}))
        self.cider = CIDErScore(**(cider_kwargs or {}))
        self.bertscore = BERTScoreApproximation(**(bertscore_kwargs or {}))
        self.chromatic = ChromaticScore(**(chromatic_kwargs or {}))

        all_names = ["bleu", "rouge", "meteor", "cider", "bertscore", "chromatic"]
        self.enabled = enabled_metrics or all_names

    # -- compute -----------------------------------------------------------

    def compute(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        if "bleu" in self.enabled:
            results["bleu"] = self.bleu.compute_detailed(hypotheses, references)

        if "rouge" in self.enabled:
            results["rouge"] = self.rouge.compute(hypotheses, references)

        if "meteor" in self.enabled:
            results["meteor"] = self.meteor.compute_detailed(hypotheses, references)

        if "cider" in self.enabled:
            results["cider"] = self.cider.compute_detailed(hypotheses, references)

        if "bertscore" in self.enabled:
            results["bertscore"] = self.bertscore.compute(hypotheses, references)

        if "chromatic" in self.enabled:
            results["chromatic"] = self.chromatic.compute(hypotheses, references)

        return results

    # -- aggregation -------------------------------------------------------

    def summary(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """Single scalar per metric for easy comparison."""
        full = self.compute(hypotheses, references)
        summary: Dict[str, float] = {}

        if "bleu" in full:
            summary["bleu"] = full["bleu"]["corpus_bleu"]
        if "rouge" in full:
            for key, val in full["rouge"].items():
                summary[f"{key}_f1"] = val["f1"]
        if "meteor" in full:
            summary["meteor"] = full["meteor"]["corpus_meteor"]
        if "cider" in full:
            summary["cider"] = full["cider"]["corpus_cider"]
        if "bertscore" in full:
            summary["bertscore_f1"] = full["bertscore"]["f1"]
        if "chromatic" in full:
            summary["chromatic"] = full["chromatic"]["chromatic"]

        return summary

    # -- per-sentence scores (for correlation) -----------------------------

    def _per_sentence_vectors(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict[str, np.ndarray]:
        """Collect per-sentence scalar scores for each metric."""
        n = len(hypotheses)
        vectors: Dict[str, np.ndarray] = {}

        if "bleu" in self.enabled:
            vectors["bleu"] = np.array([
                self.bleu.sentence_bleu(h, refs)
                for h, refs in zip(hypotheses, references)
            ])

        if "rouge" in self.enabled:
            rouge_l_vals = []
            for h, refs in zip(hypotheses, references):
                s = self.rouge.rouge_l(h, refs)
                rouge_l_vals.append(s["f1"])
            vectors["rouge-l"] = np.array(rouge_l_vals)

        if "meteor" in self.enabled:
            vectors["meteor"] = np.array([
                self.meteor.sentence_meteor(h, refs)["score"]
                for h, refs in zip(hypotheses, references)
            ])

        if "bertscore" in self.enabled:
            self.bertscore.fit(hypotheses, references)
            vectors["bertscore"] = np.array([
                self.bertscore.score_pair(h, refs)["f1"]
                for h, refs in zip(hypotheses, references)
            ])

        return vectors

    # -- correlation analysis ----------------------------------------------

    @staticmethod
    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2:
            return 0.0
        x_m = x - np.mean(x)
        y_m = y - np.mean(y)
        denom = np.sqrt(np.sum(x_m ** 2) * np.sum(y_m ** 2))
        if denom == 0:
            return 0.0
        return float(np.sum(x_m * y_m) / denom)

    @staticmethod
    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        """Spearman rank correlation (using average ranks for ties)."""
        if len(x) < 2:
            return 0.0

        def _rank(arr: np.ndarray) -> np.ndarray:
            order = np.argsort(arr)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
            # Handle ties with average rank
            sorted_arr = arr[order]
            i = 0
            while i < len(sorted_arr):
                j = i
                while j < len(sorted_arr) and sorted_arr[j] == sorted_arr[i]:
                    j += 1
                if j > i + 1:
                    avg_rank = np.mean(ranks[order[i:j]])
                    ranks[order[i:j]] = avg_rank
                i = j
            return ranks

        rx = _rank(x)
        ry = _rank(y)
        return ReferenceMetricSuite._pearson(rx, ry)

    @staticmethod
    def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
        """Kendall's tau-b rank correlation."""
        n = len(x)
        if n < 2:
            return 0.0
        concordant = 0
        discordant = 0
        ties_x = 0
        ties_y = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx == 0 and dy == 0:
                    ties_x += 1
                    ties_y += 1
                elif dx == 0:
                    ties_x += 1
                elif dy == 0:
                    ties_y += 1
                elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                    concordant += 1
                else:
                    discordant += 1
        total_pairs = n * (n - 1) / 2
        denom = math.sqrt(
            (total_pairs - ties_x) * (total_pairs - ties_y)
        )
        if denom == 0:
            return 0.0
        return (concordant - discordant) / denom

    def correlation_analysis(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict[str, Any]:
        """Pairwise correlation between all enabled metrics.

        Returns Pearson, Spearman, and Kendall-τ matrices.
        """
        vectors = self._per_sentence_vectors(hypotheses, references)
        metric_names = sorted(vectors.keys())
        n_metrics = len(metric_names)

        pearson_mat = np.zeros((n_metrics, n_metrics))
        spearman_mat = np.zeros((n_metrics, n_metrics))
        kendall_mat = np.zeros((n_metrics, n_metrics))

        for i, m1 in enumerate(metric_names):
            for j, m2 in enumerate(metric_names):
                x, y = vectors[m1], vectors[m2]
                pearson_mat[i, j] = self._pearson(x, y)
                spearman_mat[i, j] = self._spearman(x, y)
                kendall_mat[i, j] = self._kendall_tau(x, y)

        def _mat_to_dict(mat: np.ndarray) -> Dict[str, Dict[str, float]]:
            return {
                m1: {m2: float(mat[i, j]) for j, m2 in enumerate(metric_names)}
                for i, m1 in enumerate(metric_names)
            }

        return {
            "metrics": metric_names,
            "pearson": _mat_to_dict(pearson_mat),
            "spearman": _mat_to_dict(spearman_mat),
            "kendall_tau": _mat_to_dict(kendall_mat),
        }

    # -- report ------------------------------------------------------------

    def report(
        self,
        hypotheses: List[str],
        references: List[List[str]],
        include_correlation: bool = True,
    ) -> Dict[str, Any]:
        """Full report: all metrics + optional correlation analysis."""
        result: Dict[str, Any] = {
            "scores": self.compute(hypotheses, references),
            "summary": self.summary(hypotheses, references),
            "num_hypotheses": len(hypotheses),
        }
        if include_correlation and len(hypotheses) >= 3:
            result["correlation"] = self.correlation_analysis(
                hypotheses, references,
            )
        return result


# =========================================================================
# Convenience factory & quick-score functions
# =========================================================================


def bleu(hypotheses: List[str], references: List[List[str]], **kw: Any) -> float:
    """Quick corpus BLEU."""
    return BLEUScore(**kw).compute(hypotheses, references)


def rouge(hypotheses: List[str], references: List[List[str]], **kw: Any) -> Dict[str, Dict[str, float]]:
    """Quick ROUGE (all variants)."""
    return ROUGEScore(**kw).compute(hypotheses, references)


def meteor(hypotheses: List[str], references: List[List[str]], **kw: Any) -> float:
    """Quick corpus METEOR."""
    return METEORScore(**kw).compute(hypotheses, references)


def cider(hypotheses: List[str], references: List[List[str]], **kw: Any) -> float:
    """Quick corpus CIDEr."""
    return CIDErScore(**kw).compute(hypotheses, references)


def bertscore_approx(
    hypotheses: List[str], references: List[List[str]], **kw: Any,
) -> Dict[str, float]:
    """Quick approximate BERTScore."""
    return BERTScoreApproximation(**kw).compute(hypotheses, references)


def all_metrics(
    hypotheses: List[str],
    references: List[List[str]],
    **kw: Any,
) -> Dict[str, Any]:
    """Compute every reference metric and return a unified report."""
    return ReferenceMetricSuite(**kw).report(hypotheses, references)
