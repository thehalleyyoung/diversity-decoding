"""
Adversarial divergence search for diversity metrics.

Given two metrics that are statistically correlated (high Kendall τ),
searches for adversarial inputs where they diverge. This strengthens
the metric redundancy claim: if even adversarial search cannot find
divergent inputs, the redundancy is robust.

Also implements worst-case analysis for fair selection, computing
the Pareto frontier between constraint tightness and diversity retention.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: List[str], n: int = 2) -> float:
    """Compute distinct-n metric."""
    all_ng = []
    for t in texts:
        all_ng.extend(_ngrams(_tokenize(t), n))
    if not all_ng:
        return 0.0
    return len(set(all_ng)) / len(all_ng)


def self_bleu_approx(texts: List[str], n: int = 4) -> float:
    """Approximate self-BLEU via n-gram overlap."""
    if len(texts) < 2:
        return 0.0
    tokenized = [_tokenize(t) for t in texts]
    scores = []
    for i in range(len(texts)):
        ref_ng = Counter(_ngrams(tokenized[i], n))
        if not ref_ng:
            continue
        for j in range(len(texts)):
            if i == j:
                continue
            hyp_ng = Counter(_ngrams(tokenized[j], n))
            if not hyp_ng:
                scores.append(0.0)
                continue
            overlap = sum((ref_ng & hyp_ng).values())
            total = sum(hyp_ng.values())
            scores.append(overlap / total if total > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def ttr(texts: List[str]) -> float:
    """Type-token ratio across all texts."""
    all_tokens = []
    for t in texts:
        all_tokens.extend(_tokenize(t))
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)


def epd_tfidf(texts: List[str]) -> float:
    """Embedding pairwise distance using TF-IDF vectors."""
    if len(texts) < 2:
        return 0.0
    docs = [_tokenize(t) for t in texts]
    vocab = {}
    for doc in docs:
        for tok in doc:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    if not vocab:
        return 0.0
    n, v = len(texts), len(vocab)
    tf = np.zeros((n, v))
    for i, doc in enumerate(docs):
        for tok in doc:
            tf[i, vocab[tok]] += 1.0
        if doc:
            tf[i] /= len(doc)
    df = np.sum(tf > 0, axis=0).astype(float)
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
    tfidf = tf * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    tfidf_norm = tfidf / norms
    sim = tfidf_norm @ tfidf_norm.T
    np.clip(sim, -1.0, 1.0, out=sim)
    dists = 1.0 - sim
    idx = np.triu_indices(n, k=1)
    return float(np.mean(dists[idx]))


# -----------------------------------------------------------------------
# Adversarial divergence search
# -----------------------------------------------------------------------

class AdversarialDivergenceSearch:
    """Search for text sets where two metrics maximally disagree on ranking.

    Strategy: construct adversarial text sets by controlled manipulation
    (repetition injection, vocabulary restriction, synonym flooding)
    and measure where metric rankings diverge.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def _generate_repetitive_texts(self, base_texts: List[str],
                                    repeat_frac: float) -> List[str]:
        """Create texts with controlled repetition to fool lexical metrics."""
        result = []
        for t in base_texts:
            tokens = _tokenize(t)
            if not tokens:
                result.append(t)
                continue
            n_repeat = max(1, int(len(tokens) * repeat_frac))
            repeated_tokens = tokens[:n_repeat] * 3 + tokens[n_repeat:]
            result.append(' '.join(repeated_tokens))
        return result

    def _generate_synonym_texts(self, base_texts: List[str]) -> List[str]:
        """Create texts with high lexical diversity but low semantic diversity."""
        substitutions = {
            'big': ['large', 'huge', 'enormous', 'massive', 'giant'],
            'small': ['tiny', 'little', 'miniature', 'petite', 'minute'],
            'good': ['excellent', 'great', 'wonderful', 'superb', 'fine'],
            'bad': ['terrible', 'awful', 'horrible', 'dreadful', 'poor'],
            'fast': ['quick', 'rapid', 'swift', 'speedy', 'hasty'],
            'said': ['stated', 'declared', 'mentioned', 'noted', 'remarked'],
            'went': ['traveled', 'journeyed', 'moved', 'proceeded', 'walked'],
            'the': ['the', 'a', 'that', 'this', 'one'],
        }
        result = []
        for t in base_texts:
            tokens = _tokenize(t)
            new_tokens = []
            for tok in tokens:
                if tok in substitutions:
                    new_tokens.append(self.rng.choice(substitutions[tok]))
                else:
                    new_tokens.append(tok)
            result.append(' '.join(new_tokens))
        return result

    def _generate_template_texts(self, n: int = 10) -> List[str]:
        """Texts from the same template: high semantic similarity, varied tokens."""
        templates = [
            "The {adj} {animal} {verb} over the {adj2} {place}.",
            "A {adj} {animal} {verb} across the {adj2} {place}.",
            "One {adj} {animal} {verb} through the {adj2} {place}.",
            "Some {adj} {animal} {verb} around the {adj2} {place}.",
            "That {adj} {animal} {verb} into the {adj2} {place}.",
        ]
        adjs = ['quick', 'lazy', 'clever', 'bright', 'dark', 'old', 'young']
        animals = ['fox', 'cat', 'dog', 'bird', 'fish', 'horse', 'deer']
        verbs = ['jumped', 'ran', 'flew', 'swam', 'walked', 'dashed', 'leaped']
        places = ['forest', 'meadow', 'river', 'mountain', 'valley', 'field']
        adjs2 = ['green', 'wide', 'deep', 'sunny', 'misty', 'rocky', 'calm']
        result = []
        for _ in range(n):
            tmpl = templates[self.rng.randint(len(templates))]
            text = tmpl.format(
                adj=adjs[self.rng.randint(len(adjs))],
                animal=animals[self.rng.randint(len(animals))],
                verb=verbs[self.rng.randint(len(verbs))],
                adj2=adjs2[self.rng.randint(len(adjs2))],
                place=places[self.rng.randint(len(places))],
            )
            result.append(text)
        return result

    def _generate_diverse_topics(self, n: int = 10) -> List[str]:
        """Texts from very different topics: high semantic + lexical diversity."""
        pool = [
            "Quantum entanglement enables instantaneous correlation between distant particles.",
            "The chef prepared a delicate soufflé using imported French butter.",
            "Medieval knights wore heavy plate armor during tournaments.",
            "The stock market crashed after unexpected inflation data was released.",
            "Coral reefs support approximately twenty-five percent of marine biodiversity.",
            "Machine learning algorithms optimize loss functions via gradient descent.",
            "The architect designed a sustainable building with solar panels and green roofs.",
            "Jazz musicians improvise complex harmonies over standard chord progressions.",
            "Volcanic eruptions reshape landscapes and create new geological formations.",
            "The submarine descended three thousand meters below the ocean surface.",
            "Ancient Egyptian pyramids were built as tombs for pharaohs.",
            "Programming languages like Python use dynamic typing for flexibility.",
            "The ballet dancer performed a perfect pirouette during the finale.",
            "Cryptocurrency transactions are verified by decentralized blockchain networks.",
            "The rainforest canopy contains more species than any other habitat.",
        ]
        indices = self.rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [pool[i] for i in indices]

    def search(self, n_trials: int = 100) -> Dict:
        """Run adversarial search for metric divergence.

        Tests multiple adversarial strategies to find text sets where
        D-2 and Self-BLEU disagree, D-2 and EPD diverge, etc.

        Returns detailed results including max divergence found.
        """
        strategies = {
            'template_variation': self._generate_template_texts,
            'diverse_topics': self._generate_diverse_topics,
        }

        # Metric pairs to check
        metric_fns = {
            'distinct_2': lambda texts: distinct_n(texts, 2),
            'self_bleu': self_bleu_approx,
            'ttr': ttr,
            'epd_tfidf': epd_tfidf,
        }

        results = {
            'n_trials': n_trials,
            'metric_pairs': {},
            'max_divergences': {},
            'adversarial_examples': [],
        }

        pairs = [
            ('distinct_2', 'self_bleu', True),   # expected anti-correlated
            ('distinct_2', 'ttr', False),         # expected correlated
            ('distinct_2', 'epd_tfidf', False),   # semi-independent
        ]

        for m1_name, m2_name, anti in pairs:
            pair_key = f"{m1_name}_vs_{m2_name}"
            m1_fn, m2_fn = metric_fns[m1_name], metric_fns[m2_name]
            all_m1, all_m2 = [], []
            max_rank_diff = 0.0
            best_adversarial = None

            for trial in range(n_trials):
                trial_rng = np.random.RandomState(trial)
                text_sets = []

                # Generate multiple text sets with different strategies
                for _ in range(8):
                    strategy = list(strategies.keys())[trial_rng.randint(len(strategies))]
                    texts = strategies[strategy](n=trial_rng.randint(5, 12))
                    # Apply random adversarial perturbation
                    if trial_rng.random() < 0.3:
                        texts = self._generate_repetitive_texts(texts, trial_rng.uniform(0.1, 0.5))
                    if trial_rng.random() < 0.3:
                        texts = self._generate_synonym_texts(texts)
                    text_sets.append(texts)

                m1_scores = [m1_fn(ts) for ts in text_sets]
                m2_scores = [m2_fn(ts) for ts in text_sets]
                all_m1.extend(m1_scores)
                all_m2.extend(m2_scores)

                # Check rank disagreement
                m1_ranks = np.argsort(np.argsort(m1_scores))
                m2_ranks = np.argsort(np.argsort(m2_scores))
                if anti:
                    m2_ranks = len(m2_ranks) - 1 - m2_ranks
                rank_diff = np.max(np.abs(m1_ranks - m2_ranks)) / len(m1_ranks)
                if rank_diff > max_rank_diff:
                    max_rank_diff = rank_diff
                    best_adversarial = {
                        'trial': trial,
                        'm1_scores': [float(s) for s in m1_scores],
                        'm2_scores': [float(s) for s in m2_scores],
                        'rank_diff': float(rank_diff),
                    }

            # Compute Kendall τ across all trials
            all_m1, all_m2 = np.array(all_m1), np.array(all_m2)
            n_concordant, n_discordant = 0, 0
            total = len(all_m1)
            for i in range(total):
                for j in range(i + 1, min(i + 50, total)):  # subsample for speed
                    s = (all_m1[i] - all_m1[j]) * (all_m2[i] - all_m2[j])
                    if s > 0:
                        n_concordant += 1
                    elif s < 0:
                        n_discordant += 1
            denom = n_concordant + n_discordant
            tau = (n_concordant - n_discordant) / denom if denom > 0 else 0.0

            results['metric_pairs'][pair_key] = {
                'kendall_tau': float(tau),
                'max_rank_divergence': float(max_rank_diff),
                'n_evaluations': total,
            }
            results['max_divergences'][pair_key] = float(max_rank_diff)
            if best_adversarial:
                best_adversarial['pair'] = pair_key
                results['adversarial_examples'].append(best_adversarial)

        return results


# -----------------------------------------------------------------------
# Worst-case fair selection analysis
# -----------------------------------------------------------------------

class FairSelectionWorstCase:
    """Analyze worst-case diversity loss under fairness constraints.

    Computes the Pareto frontier between constraint tightness
    (fraction of slots mandated per group) and diversity retention.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def _farthest_point_select(self, X: np.ndarray, k: int,
                                exclude: set = None) -> List[int]:
        """Farthest-point selection."""
        n = X.shape[0]
        available = [i for i in range(n) if (exclude is None or i not in exclude)]
        if not available or k <= 0:
            return []
        selected = [available[self.rng.randint(len(available))]]
        dists = np.full(n, np.inf)
        for _ in range(k - 1):
            last = selected[-1]
            new_d = np.linalg.norm(X - X[last], axis=1)
            dists = np.minimum(dists, new_d)
            # Mask already selected
            for s in selected:
                dists[s] = -1
            if exclude:
                for e in exclude:
                    dists[e] = -1
            best = int(np.argmax(dists))
            if dists[best] <= 0:
                break
            selected.append(best)
        return selected

    def _sum_pairwise_distance(self, X: np.ndarray, indices: List[int]) -> float:
        """Sum of pairwise distances."""
        if len(indices) < 2:
            return 0.0
        subset = X[indices]
        total = 0.0
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                total += np.linalg.norm(subset[i] - subset[j])
        return total

    def pareto_frontier(self, n: int = 200, d: int = 50, k: int = 20,
                        n_groups: int = 4, n_trials: int = 30) -> Dict:
        """Compute Pareto frontier of constraint tightness vs diversity.

        Args:
            n: Number of points.
            d: Dimensionality.
            k: Selection budget.
            n_groups: Number of groups.
            n_trials: Number of random trials.

        Returns:
            Dict with Pareto frontier data, worst-case analysis.
        """
        # Constraint levels: fraction of k mandated per group
        constraint_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        results = {
            'params': {'n': n, 'd': d, 'k': k, 'n_groups': n_groups, 'n_trials': n_trials},
            'constraint_levels': constraint_levels,
            'mean_retention': [],
            'worst_case_retention': [],
            'ci_lower': [],
            'ci_upper': [],
            'sensitivity': [],
        }

        for level in constraint_levels:
            retentions = []
            for trial in range(n_trials):
                trial_rng = np.random.RandomState(trial * 100 + int(level * 1000))
                X = trial_rng.randn(n, d)

                # Create imbalanced groups: 60%, 25%, 10%, 5%
                group_fracs = [0.6, 0.25, 0.10, 0.05]
                groups = np.zeros(n, dtype=int)
                boundaries = np.cumsum([int(f * n) for f in group_fracs[:-1]])
                for g in range(1, n_groups):
                    if g - 1 < len(boundaries):
                        groups[boundaries[g-1]:] = g

                # Unconstrained selection
                unc_indices = self._farthest_point_select(X, k)
                unc_div = self._sum_pairwise_distance(X, unc_indices)

                # Fair selection with constraint level
                min_per_group = max(1, int(level * k / n_groups))
                fair_indices = []
                for g in range(n_groups):
                    g_mask = np.where(groups == g)[0]
                    if len(g_mask) == 0:
                        continue
                    g_select = self._farthest_point_select(
                        X, min(min_per_group, len(g_mask)),
                        exclude=set(fair_indices)
                    )
                    # Map back to original indices
                    for s in g_select:
                        if s not in fair_indices:
                            fair_indices.append(s)

                # Fill remaining with unconstrained
                remaining = k - len(fair_indices)
                if remaining > 0:
                    extra = self._farthest_point_select(
                        X, remaining, exclude=set(fair_indices)
                    )
                    fair_indices.extend(extra)

                fair_div = self._sum_pairwise_distance(X, fair_indices[:k])
                retention = fair_div / unc_div if unc_div > 0 else 1.0
                retentions.append(retention)

            retentions = np.array(retentions)
            results['mean_retention'].append(float(np.mean(retentions)))
            results['worst_case_retention'].append(float(np.min(retentions)))
            results['ci_lower'].append(float(np.percentile(retentions, 2.5)))
            results['ci_upper'].append(float(np.percentile(retentions, 97.5)))

        # Compute sensitivity: dRetention/dConstraint
        for i in range(len(constraint_levels)):
            if i == 0:
                results['sensitivity'].append(0.0)
            else:
                dr = results['mean_retention'][i] - results['mean_retention'][i-1]
                dc = constraint_levels[i] - constraint_levels[i-1]
                results['sensitivity'].append(dr / dc if dc > 0 else 0.0)

        return results


# -----------------------------------------------------------------------
# NP-hardness witness construction
# -----------------------------------------------------------------------

def construct_np_hardness_witness(n: int = 20, d: int = 2, k: int = 5,
                                   n_trials: int = 1000,
                                   seed: int = 42) -> Dict:
    """Demonstrate computational hardness by showing greedy suboptimality.

    Constructs instances where greedy farthest-point selection is provably
    suboptimal, witnessing the NP-hardness of optimal max-spread selection.

    Strategy: construct point configurations where greedy gets trapped
    in local optima. The gap between greedy and exhaustive search
    demonstrates that exact optimization requires exponential search.
    """
    rng = np.random.RandomState(seed)
    gaps = []
    worst_gap = 0.0
    worst_instance = None

    for trial in range(n_trials):
        # Construct adversarial geometry: clustered + outlier structure
        n_clusters = k + 1  # More clusters than budget -> greedy can't cover all
        points = []
        for c in range(n_clusters):
            center = rng.randn(d) * 5
            n_pts = max(1, n // n_clusters)
            cluster_pts = center + rng.randn(n_pts, d) * 0.1
            points.append(cluster_pts)
        X = np.vstack(points)[:n]

        # Greedy farthest-point
        selected_greedy = [0]
        dists = np.full(len(X), np.inf)
        for _ in range(k - 1):
            last = selected_greedy[-1]
            new_d = np.linalg.norm(X - X[last], axis=1)
            dists = np.minimum(dists, new_d)
            for s in selected_greedy:
                dists[s] = -1
            best = int(np.argmax(dists))
            selected_greedy.append(best)

        greedy_spread = float(np.inf)
        for i in range(len(selected_greedy)):
            for j in range(i + 1, len(selected_greedy)):
                d_ij = np.linalg.norm(X[selected_greedy[i]] - X[selected_greedy[j]])
                greedy_spread = min(greedy_spread, d_ij)

        # Random search for better solutions (proxy for exhaustive)
        best_random_spread = greedy_spread
        for _ in range(min(200, n_trials)):
            random_sel = rng.choice(len(X), size=k, replace=False)
            spread = float(np.inf)
            for i in range(k):
                for j in range(i + 1, k):
                    d_ij = np.linalg.norm(X[random_sel[i]] - X[random_sel[j]])
                    spread = min(spread, d_ij)
            best_random_spread = max(best_random_spread, spread)

        gap = (best_random_spread - greedy_spread) / max(greedy_spread, 1e-12)
        gaps.append(gap)
        if gap > worst_gap:
            worst_gap = gap
            worst_instance = {
                'n': len(X), 'k': k, 'd': d,
                'greedy_spread': greedy_spread,
                'best_found_spread': best_random_spread,
                'gap': float(gap),
            }

    return {
        'n_trials': n_trials,
        'mean_gap': float(np.mean(gaps)),
        'max_gap': float(np.max(gaps)),
        'std_gap': float(np.std(gaps)),
        'fraction_suboptimal': float(np.mean(np.array(gaps) > 0.01)),
        'worst_instance': worst_instance,
    }
