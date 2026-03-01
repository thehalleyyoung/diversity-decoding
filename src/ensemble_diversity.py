"""
Diversity in model ensembles: disagreement, Q-statistic, correlation coefficient,
double fault, entropy, Kohavi-Wolpert variance, diversity-accuracy tradeoff,
diversity-promoting ensemble construction.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import math
from itertools import combinations


@dataclass
class EnsembleDiversityReport:
    """Report on diversity of model ensemble."""
    n_models: int
    n_samples: int
    disagreement: float = 0.0
    q_statistic: float = 0.0
    correlation_diversity: float = 0.0
    double_fault: float = 0.0
    entropy_diversity: float = 0.0
    kw_variance: float = 0.0
    individual_accuracies: List[float] = field(default_factory=list)
    ensemble_accuracy: float = 0.0
    diversity_accuracy_tradeoff: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


class DisagreementMeasure:
    """Fraction of inputs where models disagree."""

    def pairwise_disagreement(self, pred_i: np.ndarray,
                               pred_j: np.ndarray) -> float:
        """Fraction of samples where two models make different predictions."""
        return float(np.mean(pred_i != pred_j))

    def average_disagreement(self, predictions: np.ndarray) -> float:
        """Average pairwise disagreement across all model pairs.
        predictions: (n_models, n_samples) array of predicted labels.
        """
        n_models = predictions.shape[0]
        total_dis = 0.0
        count = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                total_dis += self.pairwise_disagreement(predictions[i], predictions[j])
                count += 1

        return total_dis / max(count, 1)

    def disagreement_matrix(self, predictions: np.ndarray) -> np.ndarray:
        """Matrix of pairwise disagreements."""
        n_models = predictions.shape[0]
        matrix = np.zeros((n_models, n_models))

        for i in range(n_models):
            for j in range(i + 1, n_models):
                d = self.pairwise_disagreement(predictions[i], predictions[j])
                matrix[i, j] = d
                matrix[j, i] = d

        return matrix


class QStatistic:
    """Q-statistic: pairwise diversity between classifiers."""

    def compute_contingency(self, pred_i: np.ndarray, pred_j: np.ndarray,
                            labels: np.ndarray) -> Tuple[int, int, int, int]:
        """Compute contingency table:
        N11: both correct, N10: i correct j wrong
        N01: i wrong j correct, N00: both wrong
        """
        correct_i = (pred_i == labels)
        correct_j = (pred_j == labels)

        n11 = int(np.sum(correct_i & correct_j))
        n10 = int(np.sum(correct_i & ~correct_j))
        n01 = int(np.sum(~correct_i & correct_j))
        n00 = int(np.sum(~correct_i & ~correct_j))

        return n11, n10, n01, n00

    def pairwise_q(self, pred_i: np.ndarray, pred_j: np.ndarray,
                   labels: np.ndarray) -> float:
        """Q-statistic between two classifiers. Q=-1 is max diversity, Q=1 is no diversity."""
        n11, n10, n01, n00 = self.compute_contingency(pred_i, pred_j, labels)
        numerator = n11 * n00 - n01 * n10
        denominator = n11 * n00 + n01 * n10
        if abs(denominator) < 1e-15:
            return 0.0
        return float(numerator / denominator)

    def average_q(self, predictions: np.ndarray,
                  labels: np.ndarray) -> float:
        """Average Q-statistic across all pairs."""
        n_models = predictions.shape[0]
        total_q = 0.0
        count = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                total_q += self.pairwise_q(predictions[i], predictions[j], labels)
                count += 1

        return total_q / max(count, 1)

    def q_matrix(self, predictions: np.ndarray,
                 labels: np.ndarray) -> np.ndarray:
        """Matrix of pairwise Q-statistics."""
        n_models = predictions.shape[0]
        matrix = np.zeros((n_models, n_models))

        for i in range(n_models):
            for j in range(i + 1, n_models):
                q = self.pairwise_q(predictions[i], predictions[j], labels)
                matrix[i, j] = q
                matrix[j, i] = q

        np.fill_diagonal(matrix, 1.0)
        return matrix


class CorrelationDiversity:
    """Average pairwise correlation of errors."""

    def pairwise_error_correlation(self, pred_i: np.ndarray,
                                    pred_j: np.ndarray,
                                    labels: np.ndarray) -> float:
        """Correlation between error indicators of two models."""
        errors_i = (pred_i != labels).astype(float)
        errors_j = (pred_j != labels).astype(float)

        if np.std(errors_i) < 1e-15 or np.std(errors_j) < 1e-15:
            return 0.0

        corr, _ = pearsonr(errors_i, errors_j)
        return float(corr)

    def average_correlation(self, predictions: np.ndarray,
                            labels: np.ndarray) -> float:
        """Average pairwise error correlation."""
        n_models = predictions.shape[0]
        total_corr = 0.0
        count = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr = self.pairwise_error_correlation(
                    predictions[i], predictions[j], labels
                )
                total_corr += corr
                count += 1

        return total_corr / max(count, 1)

    def correlation_matrix(self, predictions: np.ndarray,
                           labels: np.ndarray) -> np.ndarray:
        """Matrix of pairwise error correlations."""
        n_models = predictions.shape[0]
        matrix = np.zeros((n_models, n_models))

        for i in range(n_models):
            for j in range(i + 1, n_models):
                c = self.pairwise_error_correlation(
                    predictions[i], predictions[j], labels
                )
                matrix[i, j] = c
                matrix[j, i] = c

        np.fill_diagonal(matrix, 1.0)
        return matrix


class DoubleFaultMeasure:
    """Probability both classifiers are wrong."""

    def pairwise_double_fault(self, pred_i: np.ndarray,
                               pred_j: np.ndarray,
                               labels: np.ndarray) -> float:
        """Fraction of samples where both classifiers are wrong."""
        wrong_i = pred_i != labels
        wrong_j = pred_j != labels
        return float(np.mean(wrong_i & wrong_j))

    def average_double_fault(self, predictions: np.ndarray,
                              labels: np.ndarray) -> float:
        """Average pairwise double fault."""
        n_models = predictions.shape[0]
        total = 0.0
        count = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                total += self.pairwise_double_fault(
                    predictions[i], predictions[j], labels
                )
                count += 1

        return total / max(count, 1)

    def double_fault_matrix(self, predictions: np.ndarray,
                             labels: np.ndarray) -> np.ndarray:
        """Matrix of pairwise double fault measures."""
        n_models = predictions.shape[0]
        matrix = np.zeros((n_models, n_models))

        for i in range(n_models):
            for j in range(i + 1, n_models):
                df = self.pairwise_double_fault(
                    predictions[i], predictions[j], labels
                )
                matrix[i, j] = df
                matrix[j, i] = df

        return matrix


class EnsembleEntropy:
    """Entropy of ensemble vote distribution."""

    def vote_entropy(self, predictions: np.ndarray) -> np.ndarray:
        """Entropy of vote distribution per sample.
        predictions: (n_models, n_samples)
        Returns: (n_samples,) entropy values.
        """
        n_models, n_samples = predictions.shape
        entropies = np.zeros(n_samples)

        for s in range(n_samples):
            votes = predictions[:, s]
            unique_labels, counts = np.unique(votes, return_counts=True)
            probs = counts / n_models
            probs = np.clip(probs, 1e-15, 1.0)
            entropies[s] = -np.sum(probs * np.log2(probs))

        return entropies

    def average_entropy(self, predictions: np.ndarray) -> float:
        """Average vote entropy across samples."""
        return float(np.mean(self.vote_entropy(predictions)))

    def max_entropy(self, n_models: int, n_classes: int) -> float:
        """Maximum possible vote entropy."""
        max_labels = min(n_models, n_classes)
        return float(np.log2(max_labels))

    def normalized_entropy(self, predictions: np.ndarray,
                           n_classes: int) -> float:
        """Normalized entropy: actual / max possible."""
        avg_ent = self.average_entropy(predictions)
        max_ent = self.max_entropy(predictions.shape[0], n_classes)
        if max_ent < 1e-15:
            return 0.0
        return avg_ent / max_ent

    def difficulty_distribution(self, predictions: np.ndarray,
                                 labels: np.ndarray) -> Dict[str, float]:
        """Distribution of sample difficulty based on ensemble agreement."""
        n_models, n_samples = predictions.shape
        correct_counts = np.zeros(n_samples)

        for m in range(n_models):
            correct_counts += (predictions[m] == labels).astype(float)

        fractions = correct_counts / n_models

        return {
            'easy_frac': float(np.mean(fractions > 0.8)),
            'medium_frac': float(np.mean((fractions >= 0.4) & (fractions <= 0.8))),
            'hard_frac': float(np.mean(fractions < 0.4)),
            'mean_agreement': float(np.mean(fractions)),
            'std_agreement': float(np.std(fractions)),
        }


class KohaviWolpertVariance:
    """Kohavi-Wolpert variance decomposition."""

    def compute(self, predictions: np.ndarray,
                labels: np.ndarray) -> Dict[str, float]:
        """Decompose ensemble error into bias, variance, and covariance.
        predictions: (n_models, n_samples)
        """
        n_models, n_samples = predictions.shape

        individual_errors = np.zeros((n_models, n_samples))
        for m in range(n_models):
            individual_errors[m] = (predictions[m] != labels).astype(float)

        mean_error = np.mean(individual_errors, axis=0)

        avg_individual_error = float(np.mean(individual_errors))

        from scipy.stats import mode as scipy_mode
        ensemble_pred = np.zeros(n_samples, dtype=int)
        for s in range(n_samples):
            votes = predictions[:, s]
            values, counts = np.unique(votes, return_counts=True)
            ensemble_pred[s] = values[np.argmax(counts)]

        ensemble_error = float(np.mean(ensemble_pred != labels))

        variance_per_sample = np.zeros(n_samples)
        for s in range(n_samples):
            p = mean_error[s]
            variance_per_sample[s] = p * (1 - p)

        kw_variance = float(np.mean(variance_per_sample))

        bias = ensemble_error
        diversity_benefit = avg_individual_error - ensemble_error

        return {
            'ensemble_error': ensemble_error,
            'avg_individual_error': avg_individual_error,
            'kw_variance': kw_variance,
            'bias': bias,
            'diversity_benefit': diversity_benefit,
            'error_reduction_ratio': diversity_benefit / max(avg_individual_error, 1e-15)
        }


class DiversityAccuracyTradeoff:
    """Pareto frontier of individual accuracy vs ensemble diversity."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_diverse_models(self, n_samples: int, n_classes: int,
                                 labels: np.ndarray, n_models: int,
                                 accuracy_range: Tuple[float, float] = (0.5, 0.9),
                                 correlation: float = 0.3) -> np.ndarray:
        """Generate synthetic model predictions with controlled diversity."""
        predictions = np.zeros((n_models, n_samples), dtype=int)

        for m in range(n_models):
            accuracy = self.rng.uniform(accuracy_range[0], accuracy_range[1])
            correct_mask = self.rng.rand(n_samples) < accuracy

            if m > 0 and correlation > 0:
                inherit_mask = self.rng.rand(n_samples) < correlation
                correct_mask = np.where(inherit_mask,
                                       predictions[0] == labels,
                                       correct_mask)

            predictions[m] = np.where(
                correct_mask, labels,
                self.rng.randint(0, n_classes, size=n_samples)
            )

        return predictions

    def compute_pareto_frontier(self, all_predictions: List[np.ndarray],
                                 labels: np.ndarray,
                                 max_ensemble_size: int = 10) -> List[Dict[str, float]]:
        """Find Pareto-optimal subsets trading off accuracy and diversity."""
        n_available = len(all_predictions)
        pred_array = np.array(all_predictions)

        points = []
        dis_measure = DisagreementMeasure()

        for size in range(2, min(max_ensemble_size + 1, n_available + 1)):
            for combo in combinations(range(n_available), size):
                sub_preds = pred_array[list(combo)]

                from scipy.stats import mode as scipy_mode
                n_samples = sub_preds.shape[1]
                ensemble_pred = np.zeros(n_samples, dtype=int)
                for s in range(n_samples):
                    votes = sub_preds[:, s]
                    values, counts = np.unique(votes, return_counts=True)
                    ensemble_pred[s] = values[np.argmax(counts)]

                acc = float(np.mean(ensemble_pred == labels))
                div = dis_measure.average_disagreement(sub_preds)

                individual_accs = [
                    float(np.mean(sub_preds[m] == labels))
                    for m in range(len(combo))
                ]

                points.append({
                    'models': list(combo),
                    'ensemble_accuracy': acc,
                    'diversity': div,
                    'avg_individual_accuracy': float(np.mean(individual_accs)),
                    'size': size
                })

        pareto = []
        for p in points:
            dominated = False
            for q in points:
                if (q['ensemble_accuracy'] >= p['ensemble_accuracy'] and
                    q['diversity'] >= p['diversity'] and
                    (q['ensemble_accuracy'] > p['ensemble_accuracy'] or
                     q['diversity'] > p['diversity'])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(p)

        pareto.sort(key=lambda x: x['diversity'])
        return pareto


class DiverseEnsembleConstruction:
    """Greedily add diverse models to ensemble."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def greedy_construct(self, candidate_predictions: np.ndarray,
                          labels: np.ndarray, k: int,
                          lambda_div: float = 0.5) -> List[int]:
        """Greedily select k models maximizing accuracy + diversity.
        candidate_predictions: (n_candidates, n_samples)
        """
        n_candidates = candidate_predictions.shape[0]
        n_samples = candidate_predictions.shape[1]
        selected: List[int] = []
        remaining = set(range(n_candidates))

        individual_accs = np.array([
            float(np.mean(candidate_predictions[m] == labels))
            for m in range(n_candidates)
        ])

        first = int(np.argmax(individual_accs))
        selected.append(first)
        remaining.discard(first)

        for _ in range(k - 1):
            if not remaining:
                break
            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                test_ensemble = selected + [idx]
                sub_preds = candidate_predictions[test_ensemble]

                ensemble_pred = np.zeros(n_samples, dtype=int)
                for s in range(n_samples):
                    votes = sub_preds[:, s]
                    values, counts = np.unique(votes, return_counts=True)
                    ensemble_pred[s] = values[np.argmax(counts)]

                acc = float(np.mean(ensemble_pred == labels))

                avg_disagreement = 0.0
                count = 0
                for s_idx in selected:
                    avg_disagreement += float(
                        np.mean(candidate_predictions[idx] != candidate_predictions[s_idx])
                    )
                    count += 1
                avg_disagreement /= max(count, 1)

                score = (1 - lambda_div) * acc + lambda_div * avg_disagreement
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)

        return selected

    def negative_correlation_learning(self, features: np.ndarray,
                                       labels: np.ndarray,
                                       n_models: int,
                                       lambda_ncl: float = 0.5,
                                       n_epochs: int = 50,
                                       lr: float = 0.01) -> np.ndarray:
        """Train ensemble with negative correlation learning.
        Simplified: linear models with diversity-promoting penalty.
        """
        n_samples, n_features = features.shape
        n_classes = len(np.unique(labels))

        weights = self.rng.randn(n_models, n_features, n_classes) * 0.1
        biases = np.zeros((n_models, n_classes))

        for epoch in range(n_epochs):
            all_outputs = []
            for m in range(n_models):
                logits = features @ weights[m] + biases[m]
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                all_outputs.append(probs)

            ensemble_avg = np.mean(all_outputs, axis=0)

            for m in range(n_models):
                probs = all_outputs[m]
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(n_samples), labels] = 1.0

                grad = probs - one_hot

                diversity_penalty = lambda_ncl * (probs - ensemble_avg)
                grad = grad - diversity_penalty

                weights[m] -= lr * (features.T @ grad) / n_samples
                biases[m] -= lr * np.mean(grad, axis=0)

        predictions = np.zeros((n_models, n_samples), dtype=int)
        for m in range(n_models):
            logits = features @ weights[m] + biases[m]
            predictions[m] = np.argmax(logits, axis=1)

        return predictions


class EnsembleDiversity:
    """Main class: analyze diversity of model ensembles."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.disagreement = DisagreementMeasure()
        self.q_stat = QStatistic()
        self.correlation = CorrelationDiversity()
        self.double_fault = DoubleFaultMeasure()
        self.entropy_div = EnsembleEntropy()
        self.kw_variance = KohaviWolpertVariance()
        self.tradeoff = DiversityAccuracyTradeoff(seed)
        self.construction = DiverseEnsembleConstruction(seed)

    def analyze(self, model_predictions: np.ndarray,
                labels: Optional[np.ndarray] = None) -> EnsembleDiversityReport:
        """Comprehensive ensemble diversity analysis.
        model_predictions: (n_models, n_samples) predicted labels
        labels: (n_samples,) true labels (optional)
        """
        n_models, n_samples = model_predictions.shape

        if labels is None:
            from scipy.stats import mode as scipy_mode
            ensemble_pred = np.zeros(n_samples, dtype=int)
            for s in range(n_samples):
                votes = model_predictions[:, s]
                values, counts = np.unique(votes, return_counts=True)
                ensemble_pred[s] = values[np.argmax(counts)]
            labels = ensemble_pred

        dis = self.disagreement.average_disagreement(model_predictions)
        q = self.q_stat.average_q(model_predictions, labels)
        corr = self.correlation.average_correlation(model_predictions, labels)
        df = self.double_fault.average_double_fault(model_predictions, labels)
        ent = self.entropy_div.average_entropy(model_predictions)
        kw = self.kw_variance.compute(model_predictions, labels)

        individual_accs = [
            float(np.mean(model_predictions[m] == labels))
            for m in range(n_models)
        ]

        ensemble_pred = np.zeros(n_samples, dtype=int)
        for s in range(n_samples):
            votes = model_predictions[:, s]
            values, counts = np.unique(votes, return_counts=True)
            ensemble_pred[s] = values[np.argmax(counts)]
        ensemble_acc = float(np.mean(ensemble_pred == labels))

        return EnsembleDiversityReport(
            n_models=n_models,
            n_samples=n_samples,
            disagreement=dis,
            q_statistic=q,
            correlation_diversity=corr,
            double_fault=df,
            entropy_diversity=ent,
            kw_variance=kw['kw_variance'],
            individual_accuracies=individual_accs,
            ensemble_accuracy=ensemble_acc,
            diversity_accuracy_tradeoff=kw,
            details={
                'disagreement_matrix': self.disagreement.disagreement_matrix(
                    model_predictions
                ).tolist(),
                'difficulty_distribution': self.entropy_div.difficulty_distribution(
                    model_predictions, labels
                ),
            }
        )

    def generate_test_ensemble(self, n_models: int = 5,
                                n_samples: int = 200,
                                n_classes: int = 3,
                                diversity_level: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic ensemble for testing."""
        labels = self.rng.randint(0, n_classes, size=n_samples)
        correlation = 1.0 - diversity_level

        predictions = self.tradeoff.generate_diverse_models(
            n_samples, n_classes, labels, n_models,
            accuracy_range=(0.6, 0.85),
            correlation=correlation
        )

        return predictions, labels
