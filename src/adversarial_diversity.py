"""
Adversarial approaches to diversity: adversarial filtering, GAN-style diversity,
contrastive diversity, adversarial deduplication, robustness testing, diversity attack.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field
import math


@dataclass
class AdversarialDiversityResult:
    """Result of adversarial diversity operation."""
    selected_indices: List[int]
    diversity_score: float = 0.0
    adversarial_score: float = 0.0
    n_filtered: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobustnessResult:
    """Result of robustness testing."""
    original_diversity: float = 0.0
    perturbed_diversity: float = 0.0
    max_decrease: float = 0.0
    perturbation_norm: float = 0.0
    robustness_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class SimpleDiscriminator:
    """Simple discriminator network (linear + nonlinearity)."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.w1 = self.rng.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = self.rng.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros(1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict probability of being 'real' (from existing set)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        h = np.maximum(0, x @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        return 1.0 / (1.0 + np.exp(-logits.flatten()))

    def train_step(self, real: np.ndarray, fake: np.ndarray,
                   lr: float = 0.01) -> float:
        """One training step of discriminator."""
        h_real = np.maximum(0, real @ self.w1 + self.b1)
        logits_real = h_real @ self.w2 + self.b2
        probs_real = 1.0 / (1.0 + np.exp(-logits_real.flatten()))

        h_fake = np.maximum(0, fake @ self.w1 + self.b1)
        logits_fake = h_fake @ self.w2 + self.b2
        probs_fake = 1.0 / (1.0 + np.exp(-logits_fake.flatten()))

        loss = -np.mean(np.log(probs_real + 1e-10)) - np.mean(np.log(1 - probs_fake + 1e-10))

        grad_real = (probs_real - 1).reshape(-1, 1)
        grad_fake = probs_fake.reshape(-1, 1)

        dw2_real = h_real.T @ grad_real / len(real)
        dw2_fake = h_fake.T @ grad_fake / len(fake)
        dw2 = dw2_real + dw2_fake
        db2 = np.mean(np.vstack([grad_real, grad_fake]), axis=0)

        grad_h_real = grad_real @ self.w2.T * (h_real > 0)
        grad_h_fake = grad_fake @ self.w2.T * (h_fake > 0)

        dw1 = (real.T @ grad_h_real + fake.T @ grad_h_fake) / (len(real) + len(fake))
        db1 = np.mean(np.vstack([grad_h_real, grad_h_fake]), axis=0)

        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2

        return float(loss)


class AdversarialFilter:
    """Remove items that discriminator can distinguish from existing set."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def filter(self, existing: np.ndarray, candidates: np.ndarray,
               threshold: float = 0.5,
               n_train_steps: int = 100) -> AdversarialDiversityResult:
        """Filter out candidates that are too similar to existing items."""
        dim = existing.shape[1]
        disc = SimpleDiscriminator(dim, seed=self.rng.randint(10000))

        for _ in range(n_train_steps):
            disc.train_step(existing, candidates, lr=0.01)

        scores = disc.predict(candidates)

        diverse_mask = scores < threshold
        kept_indices = np.where(diverse_mask)[0].tolist()
        filtered_indices = np.where(~diverse_mask)[0].tolist()

        diversity = 0.0
        if len(kept_indices) > 1:
            kept_items = candidates[kept_indices]
            diversity = float(np.mean(pdist(kept_items)))

        return AdversarialDiversityResult(
            selected_indices=kept_indices,
            diversity_score=diversity,
            adversarial_score=float(np.mean(scores)),
            n_filtered=len(filtered_indices),
            details={
                'threshold': threshold,
                'mean_disc_score': float(np.mean(scores)),
                'filtered_indices': filtered_indices
            }
        )

    def iterative_filter(self, existing: np.ndarray, candidates: np.ndarray,
                          n_rounds: int = 5, threshold: float = 0.5,
                          n_train_steps: int = 50) -> AdversarialDiversityResult:
        """Iteratively filter, retraining discriminator each round."""
        current_candidates = candidates.copy()
        all_kept_indices = list(range(len(candidates)))
        total_filtered = 0

        for round_idx in range(n_rounds):
            if len(current_candidates) <= 1:
                break

            result = self.filter(existing, current_candidates,
                                threshold=threshold,
                                n_train_steps=n_train_steps)

            if not result.selected_indices:
                break

            kept_global = [all_kept_indices[i] for i in result.selected_indices]
            all_kept_indices = kept_global
            current_candidates = candidates[all_kept_indices]
            total_filtered += result.n_filtered

        diversity = 0.0
        if len(all_kept_indices) > 1:
            diversity = float(np.mean(pdist(candidates[all_kept_indices])))

        return AdversarialDiversityResult(
            selected_indices=all_kept_indices,
            diversity_score=diversity,
            n_filtered=total_filtered,
            details={'n_rounds': n_rounds}
        )


class GANDiversity:
    """GAN-style diversity: generator produces diverse, discriminator detects similarity."""

    def __init__(self, dim: int, latent_dim: int = 16, seed: int = 42):
        self.dim = dim
        self.latent_dim = latent_dim
        self.rng = np.random.RandomState(seed)

        self.g_w1 = self.rng.randn(latent_dim, 32) * 0.1
        self.g_b1 = np.zeros(32)
        self.g_w2 = self.rng.randn(32, dim) * 0.1
        self.g_b2 = np.zeros(dim)

        self.discriminator = SimpleDiscriminator(dim, seed=seed)

    def generate(self, n: int) -> np.ndarray:
        """Generate n diverse items."""
        z = self.rng.randn(n, self.latent_dim)
        h = np.maximum(0, z @ self.g_w1 + self.g_b1)
        output = np.tanh(h @ self.g_w2 + self.g_b2)
        return output

    def train(self, real_data: np.ndarray, n_epochs: int = 100,
              batch_size: int = 32, lr: float = 0.01) -> Dict[str, List[float]]:
        """Train GAN for diverse generation."""
        d_losses = []
        g_losses = []
        diversities = []

        n = len(real_data)

        for epoch in range(n_epochs):
            idx = self.rng.choice(n, size=min(batch_size, n), replace=False)
            real_batch = real_data[idx]

            fake_batch = self.generate(len(real_batch))
            d_loss = self.discriminator.train_step(real_batch, fake_batch, lr)
            d_losses.append(d_loss)

            # Generator step: push fakes to look different from real
            fake = self.generate(min(batch_size, n))
            disc_scores = self.discriminator.predict(fake)

            # Want discriminator to think these are diverse (score near 0.5)
            g_loss = float(np.mean((disc_scores - 0.5) ** 2))

            z = self.rng.randn(min(batch_size, n), self.latent_dim)
            h = np.maximum(0, z @ self.g_w1 + self.g_b1)

            # Add diversity penalty to generator
            if len(fake) > 1:
                pairwise = pdist(fake)
                diversity_penalty = -float(np.mean(pairwise)) * 0.1
            else:
                diversity_penalty = 0.0

            noise_scale = lr * 0.1
            self.g_w1 += self.rng.randn(*self.g_w1.shape) * noise_scale
            self.g_w2 += self.rng.randn(*self.g_w2.shape) * noise_scale

            g_losses.append(g_loss)

            if len(fake) > 1:
                diversities.append(float(np.mean(pdist(fake))))
            else:
                diversities.append(0.0)

        return {
            'd_losses': d_losses,
            'g_losses': g_losses,
            'diversities': diversities
        }

    def generate_diverse_set(self, n: int, existing: np.ndarray,
                              n_candidates: int = 100) -> np.ndarray:
        """Generate n diverse items that are different from existing."""
        candidates = self.generate(n_candidates)

        disc_scores = self.discriminator.predict(candidates)
        novelty = 1.0 - disc_scores

        if len(existing) > 0:
            dists = cdist(candidates, existing)
            min_dists = np.min(dists, axis=1)
        else:
            min_dists = np.ones(n_candidates)

        scores = 0.5 * novelty + 0.5 * (min_dists / (np.max(min_dists) + 1e-10))

        selected_indices = np.argsort(scores)[-n:]
        return candidates[selected_indices]


class ContrastiveDiversity:
    """Maximize contrastive loss between selected items."""

    def __init__(self, temperature: float = 0.5, seed: int = 42):
        self.temperature = temperature
        self.rng = np.random.RandomState(seed)

    def contrastive_loss(self, items: np.ndarray) -> float:
        """Contrastive loss: encourage dissimilarity between all pairs."""
        if len(items) < 2:
            return 0.0

        norms = np.linalg.norm(items, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = items / norms

        sim_matrix = normalized @ normalized.T
        np.fill_diagonal(sim_matrix, -np.inf)

        scaled = sim_matrix / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        np.fill_diagonal(exp_scaled, 0)
        log_sum_exp = np.log(np.sum(exp_scaled, axis=1) + 1e-10)

        return float(np.mean(log_sum_exp))

    def maximize_contrastive(self, pool: np.ndarray, k: int) -> List[int]:
        """Greedily select items maximizing contrastive loss (= maximizing diversity)."""
        selected: List[int] = []
        remaining = set(range(len(pool)))

        first = self.rng.randint(len(pool))
        selected.append(first)
        remaining.discard(first)

        for _ in range(k - 1):
            if not remaining:
                break
            best_loss = -np.inf
            best_idx = -1

            for idx in remaining:
                test_items = pool[selected + [idx]]
                loss = self.contrastive_loss(test_items)
                if loss > best_loss:
                    best_loss = loss
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)

        return selected

    def contrastive_rerank(self, items: np.ndarray,
                           relevance_scores: np.ndarray,
                           k: int, lambda_div: float = 0.5) -> List[int]:
        """Re-rank balancing relevance and contrastive diversity."""
        selected: List[int] = []
        remaining = set(range(len(items)))

        first = int(np.argmax(relevance_scores))
        selected.append(first)
        remaining.discard(first)

        for _ in range(k - 1):
            if not remaining:
                break
            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                test_items = items[selected + [idx]]
                div_score = self.contrastive_loss(test_items)
                score = (1 - lambda_div) * relevance_scores[idx] + lambda_div * div_score

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)

        return selected


class AdversarialDeduplication:
    """Use adversarial perturbations to test dedup robustness."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_adversarial_duplicates(self, data: np.ndarray,
                                         epsilon: float = 0.1,
                                         n_per_item: int = 3) -> np.ndarray:
        """Generate adversarial near-duplicates that try to evade dedup."""
        duplicates = []
        for item in data:
            for _ in range(n_per_item):
                perturbation = self.rng.randn(*item.shape)
                perturbation = perturbation / (np.linalg.norm(perturbation) + 1e-10) * epsilon
                duplicates.append(item + perturbation)
        return np.array(duplicates)

    def test_dedup_robustness(self, data: np.ndarray,
                               dedup_func: Callable,
                               epsilon_range: List[float] = None) -> Dict[str, Any]:
        """Test deduplication at various perturbation levels."""
        if epsilon_range is None:
            if len(data) > 1:
                avg_dist = float(np.mean(pdist(data)))
                epsilon_range = [avg_dist * f for f in [0.01, 0.05, 0.1, 0.2, 0.5]]
            else:
                epsilon_range = [0.01, 0.05, 0.1]

        results = {}
        for eps in epsilon_range:
            adversarial = self.generate_adversarial_duplicates(data, epsilon=eps, n_per_item=1)
            combined = np.vstack([data, adversarial])

            kept = dedup_func(combined)
            n_original_kept = sum(1 for i in kept if i < len(data))
            n_adversarial_kept = sum(1 for i in kept if i >= len(data))

            results[f'eps_{eps:.4f}'] = {
                'epsilon': eps,
                'total_input': len(combined),
                'total_kept': len(kept),
                'original_kept': n_original_kept,
                'adversarial_kept': n_adversarial_kept,
                'dedup_rate': 1.0 - len(kept) / len(combined),
                'adversarial_evasion_rate': n_adversarial_kept / len(data)
            }

        return results

    def find_evasion_perturbation(self, item: np.ndarray,
                                  dedup_func: Callable,
                                  existing: np.ndarray,
                                  max_epsilon: float = 0.5,
                                  n_attempts: int = 50) -> Dict[str, Any]:
        """Find smallest perturbation that evades deduplication."""
        best_epsilon = max_epsilon
        best_perturbation = None
        success = False

        for _ in range(n_attempts):
            eps = self.rng.uniform(0, max_epsilon)
            direction = self.rng.randn(*item.shape)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            perturbed = item + eps * direction

            combined = np.vstack([existing, perturbed.reshape(1, -1)])
            kept = dedup_func(combined)

            if len(combined) - 1 in kept:
                if eps < best_epsilon:
                    best_epsilon = eps
                    best_perturbation = perturbed
                    success = True

        return {
            'success': success,
            'best_epsilon': best_epsilon if success else None,
            'perturbation': best_perturbation
        }


class DiversityRobustnessTester:
    """Test robustness of diversity score to adversarial perturbations."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def test_robustness(self, items: np.ndarray,
                        diversity_func: Callable,
                        epsilon: float = 0.1,
                        n_perturbations: int = 100) -> RobustnessResult:
        """How robust is diversity score to small perturbations?"""
        original_div = diversity_func(items)

        min_div = original_div
        max_decrease = 0.0
        worst_perturbation = None

        for _ in range(n_perturbations):
            perturbed = items.copy()
            item_idx = self.rng.randint(len(items))
            noise = self.rng.randn(*items[item_idx].shape)
            noise = noise / (np.linalg.norm(noise) + 1e-10) * epsilon
            perturbed[item_idx] += noise

            perturbed_div = diversity_func(perturbed)
            decrease = original_div - perturbed_div

            if decrease > max_decrease:
                max_decrease = decrease
                min_div = perturbed_div
                worst_perturbation = noise

        robustness = 1.0 - max_decrease / max(original_div, 1e-15)

        return RobustnessResult(
            original_diversity=original_div,
            perturbed_diversity=min_div,
            max_decrease=max_decrease,
            perturbation_norm=epsilon,
            robustness_score=max(0, robustness),
            details={
                'n_perturbations': n_perturbations,
                'worst_perturbation_norm': float(np.linalg.norm(worst_perturbation)) if worst_perturbation is not None else 0.0
            }
        )

    def gradient_attack(self, items: np.ndarray,
                        diversity_func: Callable,
                        epsilon: float = 0.1,
                        n_steps: int = 50,
                        step_size: float = 0.01) -> RobustnessResult:
        """Gradient-based attack to minimally decrease diversity."""
        original_div = diversity_func(items)
        perturbed = items.copy()

        for step in range(n_steps):
            current_div = diversity_func(perturbed)

            grad = np.zeros_like(perturbed)
            for i in range(len(perturbed)):
                for d in range(perturbed.shape[1]):
                    perturbed_plus = perturbed.copy()
                    perturbed_plus[i, d] += 1e-5
                    grad[i, d] = (diversity_func(perturbed_plus) - current_div) / 1e-5

            perturbed -= step_size * grad

            total_pert = perturbed - items
            pert_norm = np.linalg.norm(total_pert)
            if pert_norm > epsilon:
                total_pert = total_pert * epsilon / pert_norm
                perturbed = items + total_pert

        final_div = diversity_func(perturbed)

        return RobustnessResult(
            original_diversity=original_div,
            perturbed_diversity=final_div,
            max_decrease=original_div - final_div,
            perturbation_norm=float(np.linalg.norm(perturbed - items)),
            robustness_score=max(0, 1.0 - (original_div - final_div) / max(original_div, 1e-15))
        )


class DiversityAttack:
    """Find small perturbation that maximally decreases diversity score."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def random_attack(self, items: np.ndarray,
                      diversity_func: Callable,
                      epsilon: float = 0.1,
                      n_attempts: int = 200) -> Dict[str, Any]:
        """Random search for worst-case perturbation."""
        original_div = diversity_func(items)
        best_decrease = 0.0
        best_perturbed = None

        for _ in range(n_attempts):
            perturbed = items.copy()
            n_items_to_perturb = self.rng.randint(1, max(2, len(items) // 2 + 1))
            indices = self.rng.choice(len(items), size=n_items_to_perturb, replace=False)

            for idx in indices:
                noise = self.rng.randn(*items[idx].shape)
                noise = noise / (np.linalg.norm(noise) + 1e-10) * epsilon
                perturbed[idx] += noise

            new_div = diversity_func(perturbed)
            decrease = original_div - new_div

            if decrease > best_decrease:
                best_decrease = decrease
                best_perturbed = perturbed.copy()

        return {
            'original_diversity': original_div,
            'attacked_diversity': original_div - best_decrease,
            'max_decrease': best_decrease,
            'decrease_fraction': best_decrease / max(original_div, 1e-15),
            'epsilon': epsilon
        }

    def targeted_collapse(self, items: np.ndarray,
                           diversity_func: Callable,
                           target_idx: int,
                           epsilon: float = 0.5) -> Dict[str, Any]:
        """Move one item toward centroid to collapse diversity."""
        original_div = diversity_func(items)
        centroid = np.mean(items, axis=0)

        direction = centroid - items[target_idx]
        direction_norm = np.linalg.norm(direction)
        if direction_norm > epsilon:
            direction = direction / direction_norm * epsilon

        perturbed = items.copy()
        perturbed[target_idx] += direction

        new_div = diversity_func(perturbed)

        return {
            'original_diversity': original_div,
            'attacked_diversity': new_div,
            'decrease': original_div - new_div,
            'target_idx': target_idx,
            'perturbation_norm': float(np.linalg.norm(direction))
        }

    def find_most_vulnerable(self, items: np.ndarray,
                              diversity_func: Callable,
                              epsilon: float = 0.1) -> Dict[str, Any]:
        """Find which item is most vulnerable to perturbation."""
        original_div = diversity_func(items)
        vulnerabilities = []

        for i in range(len(items)):
            result = self.targeted_collapse(items, diversity_func, i, epsilon)
            vulnerabilities.append({
                'item_idx': i,
                'decrease': result['decrease'],
                'fraction': result['decrease'] / max(original_div, 1e-15)
            })

        vulnerabilities.sort(key=lambda x: x['decrease'], reverse=True)

        return {
            'original_diversity': original_div,
            'most_vulnerable': vulnerabilities[0] if vulnerabilities else None,
            'all_vulnerabilities': vulnerabilities
        }


class AdversarialDiversifier:
    """Main class: adversarial approaches to diversity."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.adv_filter = AdversarialFilter(seed)
        self.contrastive = ContrastiveDiversity(seed=seed)
        self.robustness_tester = DiversityRobustnessTester(seed)
        self.attacker = DiversityAttack(seed)

    def diversify(self, items: np.ndarray,
                  discriminator: Optional[SimpleDiscriminator] = None,
                  method: str = 'contrastive',
                  k: Optional[int] = None) -> AdversarialDiversityResult:
        """Select diverse subset using adversarial methods."""
        if k is None:
            k = len(items) // 2

        if method == 'filter' and discriminator is not None:
            existing = items[:len(items) // 3]
            candidates = items[len(items) // 3:]
            result = self.adv_filter.filter(existing, candidates)
            return result

        elif method == 'contrastive':
            indices = self.contrastive.maximize_contrastive(items, k)
            diversity = float(np.mean(pdist(items[indices]))) if len(indices) > 1 else 0.0
            return AdversarialDiversityResult(
                selected_indices=indices,
                diversity_score=diversity,
                details={'method': 'contrastive'}
            )

        else:
            dists = squareform(pdist(items))
            selected = [int(np.argmax(np.sum(dists, axis=1)))]
            remaining = set(range(len(items))) - set(selected)

            for _ in range(k - 1):
                if not remaining:
                    break
                best_min_dist = -1
                best_idx = -1
                for idx in remaining:
                    min_d = min(dists[idx, s] for s in selected)
                    if min_d > best_min_dist:
                        best_min_dist = min_d
                        best_idx = idx
                if best_idx >= 0:
                    selected.append(best_idx)
                    remaining.discard(best_idx)

            diversity = float(np.mean(pdist(items[selected]))) if len(selected) > 1 else 0.0
            return AdversarialDiversityResult(
                selected_indices=selected,
                diversity_score=diversity,
                details={'method': 'greedy_maxmin'}
            )
