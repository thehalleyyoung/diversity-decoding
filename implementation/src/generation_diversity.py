"""
Control diversity during text generation: temperature, top-k, nucleus, typical,
contrastive search, diverse beam search, stochastic beam search, controllable diversity.
"""

import numpy as np
from scipy.special import softmax, log_softmax
from scipy.spatial.distance import cosine, cdist, pdist, squareform
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import math


@dataclass
class GeneratedText:
    """A generated text (as token sequence) with metadata."""
    tokens: List[int]
    score: float = 0.0
    log_prob: float = 0.0
    diversity_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """Configuration for diverse generation."""
    max_length: int = 50
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    typical_p: float = 0.9
    num_beams: int = 4
    num_beam_groups: int = 2
    diversity_penalty: float = 1.0
    repetition_penalty: float = 1.2
    contrastive_alpha: float = 0.6
    contrastive_k: int = 5
    seed: int = 42


class VocabSimulator:
    """Simulates a vocabulary and language model logits."""

    def __init__(self, vocab_size: int = 500, embed_dim: int = 64, seed: int = 42):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.rng = np.random.RandomState(seed)
        self.embeddings = self.rng.randn(vocab_size, embed_dim) * 0.1
        self.embeddings = self.embeddings / (np.linalg.norm(
            self.embeddings, axis=1, keepdims=True) + 1e-10)
        self.transition_matrix = self.rng.randn(embed_dim, embed_dim) * 0.3
        self.output_projection = self.rng.randn(embed_dim, vocab_size) * 0.2

    def get_logits(self, context_tokens: List[int]) -> np.ndarray:
        """Get next-token logits given context."""
        if not context_tokens:
            return self.rng.randn(self.vocab_size) * 0.5

        context_embed = np.mean(
            [self.embeddings[t % self.vocab_size] for t in context_tokens], axis=0
        )
        hidden = np.tanh(context_embed @ self.transition_matrix)
        logits = hidden @ self.output_projection
        return logits

    def get_embedding(self, token: int) -> np.ndarray:
        return self.embeddings[token % self.vocab_size]

    def get_sequence_embedding(self, tokens: List[int]) -> np.ndarray:
        if not tokens:
            return np.zeros(self.embed_dim)
        return np.mean([self.get_embedding(t) for t in tokens], axis=0)


class TemperatureSampler:
    """Temperature-based sampling for diversity control."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def sample(self, logits: np.ndarray, temperature: float = 1.0) -> int:
        """Sample token with temperature scaling."""
        if temperature <= 0:
            return int(np.argmax(logits))
        scaled = logits / temperature
        probs = softmax(scaled)
        return int(self.rng.choice(len(probs), p=probs))

    def sample_n(self, logits: np.ndarray, n: int, temperature: float = 1.0) -> List[int]:
        """Sample n tokens."""
        if temperature <= 0:
            return [int(np.argmax(logits))] * n
        scaled = logits / temperature
        probs = softmax(scaled)
        return self.rng.choice(len(probs), size=n, p=probs, replace=True).tolist()

    def adaptive_temperature(self, logits: np.ndarray,
                             target_entropy: float) -> float:
        """Find temperature that achieves target entropy."""
        def entropy_at_temp(t):
            if t <= 0:
                return 0.0
            probs = softmax(logits / t)
            probs = np.clip(probs, 1e-15, 1.0)
            return float(-np.sum(probs * np.log2(probs)))

        low, high = 0.01, 10.0
        for _ in range(50):
            mid = (low + high) / 2
            ent = entropy_at_temp(mid)
            if ent < target_entropy:
                low = mid
            else:
                high = mid
        return (low + high) / 2


class TopKSampler:
    """Top-k sampling with diversity enforcement."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def sample(self, logits: np.ndarray, k: int = 50,
               temperature: float = 1.0) -> int:
        """Standard top-k sampling."""
        top_k_indices = np.argsort(logits)[-k:]
        top_k_logits = logits[top_k_indices]
        probs = softmax(top_k_logits / max(temperature, 1e-10))
        idx = self.rng.choice(len(probs), p=probs)
        return int(top_k_indices[idx])

    def sample_diverse(self, logits: np.ndarray, k: int, embeddings: np.ndarray,
                       previous_tokens: List[int], similarity_threshold: float = 0.8,
                       temperature: float = 1.0) -> int:
        """Top-k with diversity: reject candidates too similar to recent tokens."""
        top_k_indices = np.argsort(logits)[-k:]

        if previous_tokens:
            prev_embeds = np.array([
                embeddings[t % len(embeddings)] for t in previous_tokens[-5:]
            ])
            candidate_embeds = embeddings[top_k_indices]

            similarities = np.max(
                candidate_embeds @ prev_embeds.T /
                (np.linalg.norm(candidate_embeds, axis=1, keepdims=True) *
                 np.linalg.norm(prev_embeds, axis=1, keepdims=True).T + 1e-10),
                axis=1
            )

            diverse_mask = similarities < similarity_threshold
            if np.any(diverse_mask):
                diverse_indices = top_k_indices[diverse_mask]
                diverse_logits = logits[diverse_indices]
            else:
                diverse_indices = top_k_indices
                diverse_logits = logits[top_k_indices]
        else:
            diverse_indices = top_k_indices
            diverse_logits = logits[top_k_indices]

        probs = softmax(diverse_logits / max(temperature, 1e-10))
        idx = self.rng.choice(len(probs), p=probs)
        return int(diverse_indices[idx])


class NucleusSampler:
    """Nucleus (top-p) sampling."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def sample(self, logits: np.ndarray, p: float = 0.9,
               temperature: float = 1.0) -> int:
        """Top-p nucleus sampling."""
        scaled = logits / max(temperature, 1e-10)
        probs = softmax(scaled)

        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        cumsum = np.cumsum(sorted_probs)
        mask = cumsum <= p
        mask[0] = True
        if not np.any(mask[1:]):
            mask[1] = True

        nucleus_indices = sorted_indices[mask]
        nucleus_probs = probs[nucleus_indices]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()

        idx = self.rng.choice(len(nucleus_probs), p=nucleus_probs)
        return int(nucleus_indices[idx])

    def nucleus_size(self, logits: np.ndarray, p: float = 0.9,
                     temperature: float = 1.0) -> int:
        """How many tokens are in the nucleus?"""
        scaled = logits / max(temperature, 1e-10)
        probs = softmax(scaled)
        sorted_probs = np.sort(probs)[::-1]
        cumsum = np.cumsum(sorted_probs)
        return int(np.searchsorted(cumsum, p)) + 1


class TypicalSampler:
    """Typical sampling: select tokens with information close to expected."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def sample(self, logits: np.ndarray, typical_p: float = 0.9,
               temperature: float = 1.0) -> int:
        """Typical sampling: prefer tokens with -log p(x) close to entropy."""
        scaled = logits / max(temperature, 1e-10)
        probs = softmax(scaled)
        probs = np.clip(probs, 1e-15, 1.0)

        neg_log_probs = -np.log(probs)
        entropy_val = float(np.sum(probs * neg_log_probs))

        surprisal_deviation = np.abs(neg_log_probs - entropy_val)

        sorted_indices = np.argsort(surprisal_deviation)
        sorted_probs = probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)

        mask = cumsum <= typical_p
        mask[0] = True

        typical_indices = sorted_indices[mask]
        typical_probs = probs[typical_indices]
        typical_probs = typical_probs / typical_probs.sum()

        idx = self.rng.choice(len(typical_probs), p=typical_probs)
        return int(typical_indices[idx])

    def information_content(self, logits: np.ndarray) -> Dict[str, float]:
        """Analyze information content of distribution."""
        probs = softmax(logits)
        probs = np.clip(probs, 1e-15, 1.0)
        neg_log = -np.log2(probs)
        entropy_val = float(np.sum(probs * neg_log))

        return {
            'entropy': entropy_val,
            'max_info': float(np.max(neg_log)),
            'min_info': float(np.min(neg_log[probs > 0.001])) if np.any(probs > 0.001) else 0.0,
            'mean_info': float(np.mean(neg_log[probs > 0.001])) if np.any(probs > 0.001) else 0.0,
            'std_info': float(np.std(neg_log[probs > 0.001])) if np.any(probs > 0.001) else 0.0,
        }


class ContrastiveSearch:
    """Contrastive search: balance confidence and degeneration penalty."""

    def __init__(self, vocab_sim: VocabSimulator, seed: int = 42):
        self.vocab_sim = vocab_sim
        self.rng = np.random.RandomState(seed)

    def search(self, context: List[int], max_length: int = 50,
               alpha: float = 0.6, k: int = 5) -> GeneratedText:
        """Contrastive search: score = (1-alpha)*p(v) - alpha*max_sim(v, context)."""
        tokens = list(context)
        total_log_prob = 0.0

        for _ in range(max_length):
            logits = self.vocab_sim.get_logits(tokens)
            probs = softmax(logits)

            top_k_indices = np.argsort(probs)[-k:]
            top_k_probs = probs[top_k_indices]

            if len(tokens) > 0:
                context_embeds = np.array([
                    self.vocab_sim.get_embedding(t) for t in tokens[-10:]
                ])

                scores = np.zeros(len(top_k_indices))
                for i, idx in enumerate(top_k_indices):
                    candidate_embed = self.vocab_sim.get_embedding(idx)
                    sims = context_embeds @ candidate_embed / (
                        np.linalg.norm(context_embeds, axis=1) *
                        np.linalg.norm(candidate_embed) + 1e-10
                    )
                    max_sim = float(np.max(sims))
                    scores[i] = (1 - alpha) * top_k_probs[i] - alpha * max_sim
            else:
                scores = top_k_probs

            best_idx = top_k_indices[np.argmax(scores)]
            tokens.append(int(best_idx))
            total_log_prob += np.log(max(probs[best_idx], 1e-15))

        return GeneratedText(
            tokens=tokens[len(context):],
            score=float(np.max(scores)),
            log_prob=total_log_prob,
            details={'method': 'contrastive_search', 'alpha': alpha, 'k': k}
        )


class DiverseBeamSearch:
    """Diverse beam search: group beams, enforce inter-group diversity."""

    def __init__(self, vocab_sim: VocabSimulator, seed: int = 42):
        self.vocab_sim = vocab_sim
        self.rng = np.random.RandomState(seed)

    def search(self, context: List[int], num_beams: int = 6,
               num_groups: int = 3, diversity_penalty: float = 1.0,
               max_length: int = 30) -> List[GeneratedText]:
        """Diverse beam search with hamming diversity penalty between groups."""
        beams_per_group = num_beams // num_groups

        groups = []
        for g in range(num_groups):
            group_beams = []
            for b in range(beams_per_group):
                group_beams.append({
                    'tokens': list(context),
                    'score': 0.0,
                    'group': g
                })
            groups.append(group_beams)

        for step in range(max_length):
            for g in range(num_groups):
                all_candidates = []

                previously_selected = set()
                for prev_g in range(g):
                    for beam in groups[prev_g]:
                        if len(beam['tokens']) > len(context) + step:
                            previously_selected.add(beam['tokens'][-1])

                for beam in groups[g]:
                    logits = self.vocab_sim.get_logits(beam['tokens'])
                    probs = softmax(logits)

                    top_indices = np.argsort(probs)[-(beams_per_group * 3):]

                    for idx in top_indices:
                        penalty = diversity_penalty if idx in previously_selected else 0.0
                        score = beam['score'] + np.log(max(probs[idx], 1e-15)) - penalty

                        all_candidates.append({
                            'tokens': beam['tokens'] + [int(idx)],
                            'score': score,
                            'group': g
                        })

                all_candidates.sort(key=lambda x: x['score'], reverse=True)
                groups[g] = all_candidates[:beams_per_group]

        results = []
        for g in range(num_groups):
            for beam in groups[g]:
                gen_tokens = beam['tokens'][len(context):]
                results.append(GeneratedText(
                    tokens=gen_tokens,
                    score=beam['score'],
                    log_prob=beam['score'],
                    details={'group': g, 'method': 'diverse_beam_search'}
                ))

        if len(results) > 1:
            embeddings = np.array([
                self.vocab_sim.get_sequence_embedding(r.tokens) for r in results
            ])
            dists = pdist(embeddings)
            avg_dist = float(np.mean(dists)) if len(dists) > 0 else 0.0
            for r in results:
                r.diversity_score = avg_dist

        return results


class StochasticBeamSearch:
    """Stochastic beam search: Gumbel-top-k for unbiased diverse samples."""

    def __init__(self, vocab_sim: VocabSimulator, seed: int = 42):
        self.vocab_sim = vocab_sim
        self.rng = np.random.RandomState(seed)

    def gumbel_noise(self, shape) -> np.ndarray:
        """Sample Gumbel(0,1) noise."""
        u = self.rng.uniform(1e-10, 1.0 - 1e-10, size=shape)
        return -np.log(-np.log(u))

    def gumbel_top_k(self, log_probs: np.ndarray, k: int) -> np.ndarray:
        """Gumbel-top-k: add Gumbel noise, take top-k."""
        perturbed = log_probs + self.gumbel_noise(log_probs.shape)
        return np.argsort(perturbed)[-k:]

    def search(self, context: List[int], num_beams: int = 4,
               max_length: int = 30) -> List[GeneratedText]:
        """Stochastic beam search using Gumbel-top-k at each step."""
        beams = [{'tokens': list(context), 'score': 0.0} for _ in range(num_beams)]

        for step in range(max_length):
            all_candidates = []

            for beam in beams:
                logits = self.vocab_sim.get_logits(beam['tokens'])
                log_probs = log_softmax(logits)

                selected = self.gumbel_top_k(log_probs, num_beams)

                for idx in selected:
                    all_candidates.append({
                        'tokens': beam['tokens'] + [int(idx)],
                        'score': beam['score'] + float(log_probs[idx])
                    })

            perturbed_scores = np.array([c['score'] for c in all_candidates])
            perturbed_scores += self.gumbel_noise(len(perturbed_scores)) * 0.1
            top_indices = np.argsort(perturbed_scores)[-num_beams:]
            beams = [all_candidates[i] for i in top_indices]

        results = []
        for beam in beams:
            gen_tokens = beam['tokens'][len(context):]
            results.append(GeneratedText(
                tokens=gen_tokens,
                score=beam['score'],
                log_prob=beam['score'],
                details={'method': 'stochastic_beam_search'}
            ))

        return results


class ControllableDiversity:
    """Interpolate between deterministic and random generation."""

    def __init__(self, vocab_sim: VocabSimulator, seed: int = 42):
        self.vocab_sim = vocab_sim
        self.rng = np.random.RandomState(seed)
        self.temp_sampler = TemperatureSampler(seed)
        self.topk_sampler = TopKSampler(seed)
        self.nucleus_sampler = NucleusSampler(seed)
        self.typical_sampler = TypicalSampler(seed)

    def generate(self, context: List[int], max_length: int = 50,
                 diversity_level: float = 0.5,
                 method: str = 'interpolated') -> GeneratedText:
        """Generate with controllable diversity level [0=deterministic, 1=random]."""
        if method == 'temperature':
            return self._generate_temperature(context, max_length, diversity_level)
        elif method == 'nucleus':
            return self._generate_nucleus(context, max_length, diversity_level)
        elif method == 'typical':
            return self._generate_typical(context, max_length, diversity_level)
        elif method == 'interpolated':
            return self._generate_interpolated(context, max_length, diversity_level)
        else:
            return self._generate_temperature(context, max_length, diversity_level)

    def _generate_temperature(self, context: List[int], max_length: int,
                               diversity_level: float) -> GeneratedText:
        """Control diversity via temperature."""
        temperature = 0.1 + diversity_level * 1.9
        tokens = list(context)
        total_log_prob = 0.0

        for _ in range(max_length):
            logits = self.vocab_sim.get_logits(tokens)
            token = self.temp_sampler.sample(logits, temperature)
            tokens.append(token)
            probs = softmax(logits)
            total_log_prob += np.log(max(probs[token], 1e-15))

        gen_tokens = tokens[len(context):]
        return GeneratedText(
            tokens=gen_tokens,
            log_prob=total_log_prob,
            details={'method': 'temperature', 'temperature': temperature}
        )

    def _generate_nucleus(self, context: List[int], max_length: int,
                           diversity_level: float) -> GeneratedText:
        """Control diversity via nucleus size."""
        p = 0.1 + diversity_level * 0.89
        tokens = list(context)
        total_log_prob = 0.0

        for _ in range(max_length):
            logits = self.vocab_sim.get_logits(tokens)
            token = self.nucleus_sampler.sample(logits, p=p)
            tokens.append(token)
            probs = softmax(logits)
            total_log_prob += np.log(max(probs[token], 1e-15))

        gen_tokens = tokens[len(context):]
        return GeneratedText(
            tokens=gen_tokens,
            log_prob=total_log_prob,
            details={'method': 'nucleus', 'p': p}
        )

    def _generate_typical(self, context: List[int], max_length: int,
                           diversity_level: float) -> GeneratedText:
        """Control diversity via typical sampling mass."""
        typical_p = 0.1 + diversity_level * 0.89
        tokens = list(context)
        total_log_prob = 0.0

        for _ in range(max_length):
            logits = self.vocab_sim.get_logits(tokens)
            token = self.typical_sampler.sample(logits, typical_p=typical_p)
            tokens.append(token)
            probs = softmax(logits)
            total_log_prob += np.log(max(probs[token], 1e-15))

        gen_tokens = tokens[len(context):]
        return GeneratedText(
            tokens=gen_tokens,
            log_prob=total_log_prob,
            details={'method': 'typical', 'typical_p': typical_p}
        )

    def _generate_interpolated(self, context: List[int], max_length: int,
                                diversity_level: float) -> GeneratedText:
        """Interpolate between greedy and uniform random."""
        tokens = list(context)
        total_log_prob = 0.0

        for _ in range(max_length):
            logits = self.vocab_sim.get_logits(tokens)
            probs = softmax(logits)

            uniform = np.ones_like(probs) / len(probs)
            mixed = (1 - diversity_level) * probs + diversity_level * uniform
            mixed = mixed / mixed.sum()

            token = int(self.rng.choice(len(mixed), p=mixed))
            tokens.append(token)
            total_log_prob += np.log(max(probs[token], 1e-15))

        gen_tokens = tokens[len(context):]
        return GeneratedText(
            tokens=gen_tokens,
            log_prob=total_log_prob,
            details={'method': 'interpolated', 'diversity_level': diversity_level}
        )

    def measure_generation_diversity(self, context: List[int], n: int,
                                     diversity_level: float,
                                     method: str = 'temperature') -> Dict[str, float]:
        """Generate n samples and measure diversity."""
        samples = []
        for _ in range(n):
            result = self.generate(context, max_length=30,
                                   diversity_level=diversity_level, method=method)
            samples.append(result.tokens)

        embeddings = np.array([
            self.vocab_sim.get_sequence_embedding(s) for s in samples
        ])

        if len(embeddings) < 2:
            return {'pairwise_diversity': 0.0, 'unique_ratio': 1.0}

        dists = pdist(embeddings)
        unique_seqs = len(set(tuple(s) for s in samples))

        return {
            'pairwise_diversity': float(np.mean(dists)),
            'min_pairwise': float(np.min(dists)) if len(dists) > 0 else 0.0,
            'max_pairwise': float(np.max(dists)) if len(dists) > 0 else 0.0,
            'unique_ratio': unique_seqs / n,
            'n_unique': unique_seqs,
        }


class DiverseGenerator:
    """Main class: generate diverse text outputs."""

    def __init__(self, vocab_size: int = 500, embed_dim: int = 64, seed: int = 42):
        self.vocab_sim = VocabSimulator(vocab_size, embed_dim, seed)
        self.rng = np.random.RandomState(seed)
        self.controllable = ControllableDiversity(self.vocab_sim, seed)
        self.contrastive = ContrastiveSearch(self.vocab_sim, seed)
        self.diverse_beam = DiverseBeamSearch(self.vocab_sim, seed)
        self.stochastic_beam = StochasticBeamSearch(self.vocab_sim, seed)
        self.config = GenerationConfig(seed=seed)

    def generate(self, prompt_embedding: np.ndarray, n: int,
                 diversity_target: float = 0.5,
                 method: str = 'mixed') -> List[GeneratedText]:
        """Generate n diverse texts given prompt embedding."""
        context = self._embed_to_tokens(prompt_embedding)

        if method == 'temperature':
            return self._generate_temperature_set(context, n, diversity_target)
        elif method == 'diverse_beam':
            return self._generate_diverse_beam(context, n, diversity_target)
        elif method == 'contrastive':
            return self._generate_contrastive_set(context, n, diversity_target)
        elif method == 'stochastic_beam':
            return self._generate_stochastic_beam(context, n)
        elif method == 'mixed':
            return self._generate_mixed(context, n, diversity_target)
        else:
            return self._generate_temperature_set(context, n, diversity_target)

    def _embed_to_tokens(self, embedding: np.ndarray) -> List[int]:
        """Convert prompt embedding to initial context tokens."""
        sims = self.vocab_sim.embeddings @ embedding
        top_tokens = np.argsort(sims)[-3:]
        return top_tokens.tolist()

    def _generate_temperature_set(self, context: List[int], n: int,
                                   diversity_target: float) -> List[GeneratedText]:
        """Generate n samples using temperature, varying around diversity_target."""
        results = []
        for i in range(n):
            level = diversity_target + (i / max(n - 1, 1) - 0.5) * 0.3
            level = max(0.05, min(0.95, level))
            result = self.controllable.generate(
                context, max_length=self.config.max_length,
                diversity_level=level, method='temperature'
            )
            results.append(result)
        self._compute_set_diversity(results)
        return results

    def _generate_diverse_beam(self, context: List[int], n: int,
                                diversity_target: float) -> List[GeneratedText]:
        """Use diverse beam search."""
        num_groups = max(2, n // 2)
        results = self.diverse_beam.search(
            context, num_beams=n, num_groups=num_groups,
            diversity_penalty=diversity_target * 2,
            max_length=self.config.max_length
        )
        return results[:n]

    def _generate_contrastive_set(self, context: List[int], n: int,
                                   diversity_target: float) -> List[GeneratedText]:
        """Generate n samples using contrastive search with varying alpha."""
        results = []
        for i in range(n):
            alpha = 0.3 + (i / max(n - 1, 1)) * 0.5
            result = self.contrastive.search(
                context, max_length=self.config.max_length,
                alpha=alpha, k=self.config.contrastive_k
            )
            results.append(result)
        self._compute_set_diversity(results)
        return results

    def _generate_stochastic_beam(self, context: List[int],
                                   n: int) -> List[GeneratedText]:
        """Use stochastic beam search."""
        return self.stochastic_beam.search(
            context, num_beams=n, max_length=self.config.max_length
        )

    def _generate_mixed(self, context: List[int], n: int,
                         diversity_target: float) -> List[GeneratedText]:
        """Mix methods for maximum diversity."""
        results = []
        per_method = max(1, n // 4)

        for i in range(per_method):
            level = diversity_target + self.rng.uniform(-0.2, 0.2)
            level = max(0.05, min(0.95, level))
            r = self.controllable.generate(context, method='temperature',
                                           diversity_level=level)
            results.append(r)

        for i in range(per_method):
            r = self.contrastive.search(context, alpha=0.3 + i * 0.15)
            results.append(r)

        beam_results = self.diverse_beam.search(
            context, num_beams=max(2, per_method),
            num_groups=max(1, per_method // 2)
        )
        results.extend(beam_results[:per_method])

        stoch_results = self.stochastic_beam.search(
            context, num_beams=max(2, n - len(results))
        )
        results.extend(stoch_results[:n - len(results)])

        self._compute_set_diversity(results[:n])
        return results[:n]

    def _compute_set_diversity(self, results: List[GeneratedText]) -> None:
        """Compute and annotate diversity scores for a set of generated texts."""
        if len(results) < 2:
            return
        embeddings = np.array([
            self.vocab_sim.get_sequence_embedding(r.tokens) for r in results
        ])
        dists = squareform(pdist(embeddings))
        for i, r in enumerate(results):
            other_dists = [dists[i, j] for j in range(len(results)) if j != i]
            r.diversity_score = float(np.mean(other_dists)) if other_dists else 0.0
