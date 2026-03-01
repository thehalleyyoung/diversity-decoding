"""
Temperature Sampling for the Diversity Decoding Arena.

Implements temperature-scaled categorical sampling with multiple schedule
strategies (constant, linear decay, cosine, adaptive, entropy-based),
Gumbel-max sampling, and comprehensive analysis utilities for studying
the effect of temperature on generation diversity.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import softmax as _scipy_softmax, log_softmax as _scipy_log_softmax
from scipy.optimize import brentq

from src.algorithms.base import (
    AlgorithmRegistry,
    DecodingAlgorithm,
    DecodingConfig,
    DecodingState,
    LogitSource,
    TokenSequence,
    categorical_sample,
    register,
)

logger = logging.getLogger(__name__)

# =========================================================================
# Helper functions
# =========================================================================


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Compute softmax with temperature scaling.

    Parameters
    ----------
    logits : np.ndarray
        Raw logits, any shape.  Softmax is applied along the last axis.
    temperature : float
        Sampling temperature.  Must be > 0.

    Returns
    -------
    np.ndarray
        Probability distribution with the same shape as *logits*.

    Raises
    ------
    ValueError
        If *temperature* is not positive.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    logits = np.asarray(logits, dtype=np.float64)
    scaled = logits / temperature
    # Numerical stability: subtract max
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def log_softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Compute log-softmax with temperature scaling.

    Parameters
    ----------
    logits : np.ndarray
        Raw logits, any shape.
    temperature : float
        Sampling temperature.  Must be > 0.

    Returns
    -------
    np.ndarray
        Log-probabilities with the same shape as *logits*.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    logits = np.asarray(logits, dtype=np.float64)
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def entropy_from_logits(logits: np.ndarray) -> float:
    """Compute the Shannon entropy (in nats) of the softmax distribution.

    Parameters
    ----------
    logits : np.ndarray
        1-D array of raw logits.

    Returns
    -------
    float
        Entropy in nats.
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()
    probs = softmax_with_temperature(logits, 1.0)
    # Avoid log(0)
    probs_clipped = np.clip(probs, 1e-30, None)
    return float(-np.sum(probs * np.log(probs_clipped)))


def sample_categorical(probs: np.ndarray, n: int = 1) -> List[int]:
    """Draw *n* samples from a categorical distribution.

    Parameters
    ----------
    probs : np.ndarray
        1-D probability vector.
    n : int
        Number of independent samples.

    Returns
    -------
    List[int]
        Sampled indices.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        return [0] * n
    probs /= total

    cumulative = np.cumsum(probs)
    samples: List[int] = []
    for _ in range(n):
        r = np.random.uniform()
        idx = int(np.searchsorted(cumulative, r))
        idx = max(0, min(idx, len(probs) - 1))
        samples.append(idx)
    return samples


def find_temperature_for_entropy(
    logits: np.ndarray,
    target_entropy: float,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """Find the temperature that yields the target entropy via root-finding.

    Uses Brent's method (``scipy.optimize.brentq``) on the function
    ``H(T) - target_entropy`` over the bracket ``[1e-4, 50.0]``.

    Parameters
    ----------
    logits : np.ndarray
        1-D raw logits.
    target_entropy : float
        Desired entropy in nats.
    tol : float
        Absolute tolerance for convergence.
    max_iter : int
        Maximum iterations for the root finder.

    Returns
    -------
    float
        Temperature that achieves approximately the target entropy.
        Clamped to ``[1e-4, 50.0]`` if no exact root is found.
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()

    def _entropy_at_temp(t: float) -> float:
        p = softmax_with_temperature(logits, t)
        p_clip = np.clip(p, 1e-30, None)
        return float(-np.sum(p * np.log(p_clip)))

    lo, hi = 1e-4, 50.0
    h_lo = _entropy_at_temp(lo)
    h_hi = _entropy_at_temp(hi)

    # Target out of reachable range — clamp
    if target_entropy <= h_lo:
        return lo
    if target_entropy >= h_hi:
        return hi

    try:
        result = brentq(
            lambda t: _entropy_at_temp(t) - target_entropy,
            lo,
            hi,
            xtol=tol,
            maxiter=max_iter,
        )
        return float(result)
    except ValueError:
        # Fallback: binary search
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            h_mid = _entropy_at_temp(mid)
            if abs(h_mid - target_entropy) < tol:
                return mid
            if h_mid < target_entropy:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0


# =========================================================================
# TemperatureConfig
# =========================================================================


@dataclass
class TemperatureConfig(DecodingConfig):
    """Configuration specific to temperature-based sampling.

    Extends :class:`DecodingConfig` with temperature schedule parameters,
    dynamic temperature support, Gumbel noise, and repetition penalty.

    Attributes
    ----------
    temperature : float
        Base sampling temperature.  Values < 1 sharpen the distribution
        (more greedy); values > 1 flatten it (more random).
    dynamic_temperature : bool
        Whether to adjust temperature on the fly during generation.
    temp_schedule : str
        Schedule type for dynamic temperature.  One of ``"constant"``,
        ``"linear_decay"``, ``"cosine"``, ``"adaptive"``,
        ``"entropy_based"``.
    temp_start : float
        Starting temperature for non-constant schedules.
    temp_end : float
        Ending temperature for decay schedules.
    entropy_target : float
        Target entropy (in nats) for the adaptive / entropy-based
        temperature controller.
    gumbel_noise : bool
        If ``True``, use the Gumbel-max trick for sampling instead of
        standard categorical sampling.
    repetition_penalty : float
        Multiplicative repetition penalty (>= 1.0).
    """

    algorithm_name: str = "temperature"
    temperature: float = 1.0
    dynamic_temperature: bool = False
    temp_schedule: str = "constant"
    temp_start: float = 1.5
    temp_end: float = 0.5
    entropy_target: float = 3.0
    gumbel_noise: bool = False
    repetition_penalty: float = 1.0

    # Supported schedule names
    _VALID_SCHEDULES = frozenset({
        "constant", "linear_decay", "cosine", "adaptive", "entropy_based",
        "exponential_decay", "cyclical",
    })

    def validate(self) -> List[str]:
        """Return validation errors (empty list == valid)."""
        errors = super().validate()
        if self.temperature <= 0:
            errors.append("temperature must be > 0")
        if self.temp_start <= 0:
            errors.append("temp_start must be > 0")
        if self.temp_end <= 0:
            errors.append("temp_end must be > 0")
        if self.entropy_target < 0:
            errors.append("entropy_target must be >= 0")
        if self.temp_schedule not in self._VALID_SCHEDULES:
            errors.append(
                f"temp_schedule '{self.temp_schedule}' not in "
                f"{sorted(self._VALID_SCHEDULES)}"
            )
        if self.repetition_penalty < 1.0:
            errors.append("repetition_penalty must be >= 1.0")
        return errors


# =========================================================================
# TemperatureSchedule
# =========================================================================


class TemperatureSchedule:
    """Compute temperature values according to a predefined schedule.

    Supports constant, linear decay, cosine annealing, exponential decay,
    cyclical, and adaptive (entropy-matching) schedules.

    Parameters
    ----------
    schedule_type : str
        One of ``"constant"``, ``"linear_decay"``, ``"cosine"``,
        ``"exponential_decay"``, ``"cyclical"``, ``"adaptive"``.
    start : float
        Starting temperature.
    end : float
        Ending temperature (for decay / annealing schedules).
    total_steps : int
        Total number of generation steps.
    target_entropy : float
        Target entropy for the adaptive schedule.
    """

    def __init__(
        self,
        schedule_type: str = "constant",
        start: float = 1.5,
        end: float = 0.5,
        total_steps: int = 100,
        target_entropy: float = 3.0,
    ) -> None:
        self.schedule_type = schedule_type
        self.start = start
        self.end = end
        self.total_steps = max(total_steps, 1)
        self.target_entropy = target_entropy

        self._dispatch: Dict[str, Callable] = {
            "constant": self.constant,
            "linear_decay": self.linear_decay,
            "cosine": self.cosine,
            "exponential_decay": self.exponential_decay,
            "cyclical": self.cyclical,
            "adaptive": lambda step: self.start,  # needs logits; handled in get_temperature
        }
        if schedule_type not in self._dispatch and schedule_type != "entropy_based":
            logger.warning(
                "Unknown schedule '%s'; falling back to constant.", schedule_type
            )
            self.schedule_type = "constant"

        # State for adaptive schedule
        self._prev_temp: float = start
        self._adaptive_lr: float = 0.1

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def get_temperature(self, step: int, logits: Optional[np.ndarray] = None) -> float:
        """Return the temperature for the given step.

        Parameters
        ----------
        step : int
            Current generation step (0-indexed).
        logits : np.ndarray, optional
            Current logits, required for ``"adaptive"`` and
            ``"entropy_based"`` schedules.

        Returns
        -------
        float
            Temperature value (clamped to ``[1e-4, 100.0]``).
        """
        if self.schedule_type in ("adaptive", "entropy_based") and logits is not None:
            temp = self.adaptive(logits)
        elif self.schedule_type in self._dispatch:
            temp = self._dispatch[self.schedule_type](step)
        else:
            temp = self.start

        return float(np.clip(temp, 1e-4, 100.0))

    # ------------------------------------------------------------------
    # Schedule implementations
    # ------------------------------------------------------------------

    def constant(self, step: int) -> float:
        """Return a constant temperature (``self.start``)."""
        return self.start

    def linear_decay(self, step: int) -> float:
        """Linearly interpolate from ``start`` to ``end``.

        .. math::
            T(t) = T_{\\text{start}} + (T_{\\text{end}} - T_{\\text{start}})
                   \\cdot \\frac{t}{T_{\\text{total}} - 1}
        """
        if self.total_steps <= 1:
            return self.start
        frac = min(step / (self.total_steps - 1), 1.0)
        return self.start + (self.end - self.start) * frac

    def cosine(self, step: int) -> float:
        """Cosine annealing from ``start`` to ``end``.

        .. math::
            T(t) = T_{\\text{end}} + \\frac{T_{\\text{start}} - T_{\\text{end}}}{2}
                   \\left(1 + \\cos\\left(\\frac{\\pi t}{T_{\\text{total}}}\\right)\\right)
        """
        frac = min(step / max(self.total_steps, 1), 1.0)
        return self.end + 0.5 * (self.start - self.end) * (1.0 + math.cos(math.pi * frac))

    def exponential_decay(self, step: int) -> float:
        """Exponential decay from ``start`` toward ``end``.

        .. math::
            T(t) = T_{\\text{end}} + (T_{\\text{start}} - T_{\\text{end}})
                   \\cdot \\exp\\left(-5 \\cdot \\frac{t}{T_{\\text{total}}}\\right)

        The factor ``-5`` ensures the curve reaches roughly ``end`` by the
        last step.
        """
        frac = min(step / max(self.total_steps, 1), 1.0)
        return self.end + (self.start - self.end) * math.exp(-5.0 * frac)

    def cyclical(self, step: int) -> float:
        """Cyclical schedule oscillating between ``start`` and ``end``.

        Uses a cosine wave with a period of ``total_steps / 4`` (i.e. four
        full cycles over the generation run).
        """
        period = max(self.total_steps / 4.0, 1.0)
        phase = (step % period) / period
        mid = 0.5 * (self.start + self.end)
        amp = 0.5 * abs(self.start - self.end)
        return mid + amp * math.cos(2.0 * math.pi * phase)

    def adaptive(self, logits: np.ndarray) -> float:
        """Adjust temperature via Newton's method to match target entropy.

        Performs a single Newton update starting from the previous
        temperature value.  Clamped to ``[1e-4, 100.0]``.

        Parameters
        ----------
        logits : np.ndarray
            Current step's logits (1-D).

        Returns
        -------
        float
            Updated temperature.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        t = self._prev_temp

        for _ in range(5):  # up to 5 Newton iterations per step
            h = self._entropy(logits, t)
            dh = self._entropy_gradient(logits, t)
            residual = h - self.target_entropy

            if abs(residual) < 1e-4:
                break

            if abs(dh) < 1e-12:
                # Gradient too small; nudge
                t *= (1.0 + 0.01 * np.sign(residual))
            else:
                t = t - self._adaptive_lr * residual / dh

            t = float(np.clip(t, 1e-4, 100.0))

        self._prev_temp = t
        return t

    # ------------------------------------------------------------------
    # Entropy helpers
    # ------------------------------------------------------------------

    def _entropy(self, logits: np.ndarray, temperature: float) -> float:
        """Shannon entropy of softmax(logits / T).

        Parameters
        ----------
        logits : np.ndarray
            1-D raw logits.
        temperature : float
            Temperature value.

        Returns
        -------
        float
            Entropy in nats.
        """
        p = softmax_with_temperature(logits, temperature)
        p_clip = np.clip(p, 1e-30, None)
        return float(-np.sum(p * np.log(p_clip)))

    def _entropy_gradient(self, logits: np.ndarray, temperature: float) -> float:
        """Gradient of entropy w.r.t. temperature: dH/dT.

        Uses the identity:

        .. math::
            \\frac{dH}{dT} = \\frac{1}{T^2} \\left(
                \\text{Var}_{p}[\\ell] - \\text{Cov}_{p}[\\ell, \\log p]
            \\right)

        where :math:`\\ell_i = \\text{logits}_i` and :math:`p = \\text{softmax}(\\ell / T)`.

        In practice we compute this via finite differences for robustness.

        Parameters
        ----------
        logits : np.ndarray
            1-D raw logits.
        temperature : float
            Temperature value.

        Returns
        -------
        float
            Approximate dH/dT.
        """
        eps = max(temperature * 1e-4, 1e-8)
        h_plus = self._entropy(logits, temperature + eps)
        h_minus = self._entropy(logits, max(temperature - eps, 1e-8))
        denom = (temperature + eps) - max(temperature - eps, 1e-8)
        if abs(denom) < 1e-15:
            return 0.0
        return (h_plus - h_minus) / denom

    def __repr__(self) -> str:
        return (
            f"TemperatureSchedule(type={self.schedule_type!r}, "
            f"start={self.start}, end={self.end}, steps={self.total_steps})"
        )


# =========================================================================
# GumbelSampler
# =========================================================================


class GumbelSampler:
    """Sampling via the Gumbel-max trick and the Gumbel-Softmax relaxation.

    Provides both hard (argmax) and differentiable (softmax) Gumbel
    sampling methods plus a straight-through estimator.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self._rng = rng or np.random.default_rng()

    def sample(self, logits: np.ndarray, temperature: float = 1.0, n: int = 1) -> List[int]:
        """Draw *n* samples via the Gumbel-max trick.

        Parameters
        ----------
        logits : np.ndarray
            1-D raw logits.
        temperature : float
            Temperature applied before adding Gumbel noise.
        n : int
            Number of independent samples.

        Returns
        -------
        List[int]
            Sampled token indices.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()

        if temperature <= 0:
            return [int(np.argmax(logits))] * n

        scaled = logits / temperature
        samples: List[int] = []
        for _ in range(n):
            noise = self._gumbel_noise(scaled.shape)
            perturbed = scaled + noise
            samples.append(int(np.argmax(perturbed)))
        return samples

    def _gumbel_noise(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate Gumbel(0, 1) noise via the inverse CDF method.

        .. math::
            G = -\\log(-\\log(U)), \\quad U \\sim \\text{Uniform}(0, 1)

        Parameters
        ----------
        shape : tuple of int
            Output shape.

        Returns
        -------
        np.ndarray
            Gumbel noise array.
        """
        u = self._rng.uniform(low=1e-10, high=1.0 - 1e-10, size=shape)
        return -np.log(-np.log(u))

    def _gumbel_softmax(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Differentiable Gumbel-Softmax relaxation.

        Returns a soft (continuous) approximation to a one-hot sample
        using the concrete distribution (Jang et al., 2017; Maddison et
        al., 2017).

        Parameters
        ----------
        logits : np.ndarray
            1-D raw logits.
        temperature : float
            Temperature for the relaxation.  As ``T -> 0`` the output
            approaches a one-hot vector.

        Returns
        -------
        np.ndarray
            Soft sample vector summing to ~1.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        if temperature <= 0:
            temperature = 1e-8

        noise = self._gumbel_noise(logits.shape)
        perturbed = (logits + noise) / temperature

        # Stable softmax
        shifted = perturbed - np.max(perturbed)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals)

    def _straight_through(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Straight-through Gumbel-Softmax estimator.

        In the forward pass, returns a hard one-hot vector.  The soft
        Gumbel-Softmax output is kept as a "surrogate" for the backward
        pass (in frameworks that support autograd; here we simply return
        the hard sample plus the soft sample for reference).

        Parameters
        ----------
        logits : np.ndarray
            1-D raw logits.
        temperature : float
            Relaxation temperature.

        Returns
        -------
        np.ndarray
            Hard one-hot vector (with soft values stored for gradient
            estimation in autograd contexts).
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        soft = self._gumbel_softmax(logits, temperature)
        # Hard: argmax → one-hot
        hard = np.zeros_like(soft)
        hard[np.argmax(soft)] = 1.0
        # Straight-through: hard in forward, soft gradient in backward
        # In NumPy we simply return hard + (soft - soft).  The detach
        # semantics only matter in autograd frameworks.
        return hard + (soft - soft)  # == hard, but retains soft's "gradient"

    def __repr__(self) -> str:
        return f"GumbelSampler(rng={self._rng!r})"


# =========================================================================
# TemperatureSampling — main algorithm
# =========================================================================


@register("temperature")
class TemperatureSampling(DecodingAlgorithm):
    """Temperature-scaled categorical sampling with schedule support.

    This algorithm applies a (possibly dynamic) temperature to the raw
    logit distribution before sampling each token.  Multiple schedule
    strategies are supported, ranging from simple constant temperature to
    entropy-adaptive controllers.

    The class also integrates optional Gumbel-max sampling and
    multiplicative repetition penalty.

    Parameters
    ----------
    config : TemperatureConfig
        Full configuration for the sampling run.
    """

    def __init__(self, config: Optional[TemperatureConfig] = None) -> None:
        if config is None:
            config = TemperatureConfig()
        super().__init__(config)
        self.config: TemperatureConfig = config  # type narrow

        # Build schedule
        self._schedule = TemperatureSchedule(
            schedule_type=config.temp_schedule,
            start=config.temp_start if config.dynamic_temperature else config.temperature,
            end=config.temp_end,
            total_steps=config.max_new_tokens,
            target_entropy=config.entropy_target,
        )

        # Gumbel sampler
        rng = self._rng if self._rng is not None else np.random.default_rng()
        self._gumbel = GumbelSampler(rng=rng)

        # Per-run statistics
        self._temp_history: List[float] = []
        self._entropy_history: List[float] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def description(self) -> str:
        sched = self.config.temp_schedule if self.config.dynamic_temperature else "constant"
        return (
            f"Temperature sampling (T={self.config.temperature}, "
            f"schedule={sched})"
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> List[TokenSequence]:
        """Generate ``num_sequences`` temperature-scaled samples.

        Overrides the base ``generate`` to reset per-run bookkeeping and
        to rebuild the schedule with the correct total-step count.

        Parameters
        ----------
        logit_source : LogitSource
            Callable returning logits of shape ``(batch, vocab_size)``.
        prompt_ids : List[int]
            Token ids of the prompt.

        Returns
        -------
        List[TokenSequence]
            Generated continuations sorted by score (best first).
        """
        # Reset per-run stats
        self._temp_history = []
        self._entropy_history = []

        # Rebuild schedule
        self._schedule = TemperatureSchedule(
            schedule_type=self.config.temp_schedule,
            start=self.config.temp_start if self.config.dynamic_temperature else self.config.temperature,
            end=self.config.temp_end,
            total_steps=self.config.max_new_tokens,
            target_entropy=self.config.entropy_target,
        )

        # Delegate to base
        result = super().generate(logit_source, prompt_ids)

        logger.info(
            "Temperature sampling finished: %d sequences, %d steps, "
            "avg temp=%.3f",
            len(result),
            len(self._temp_history),
            float(np.mean(self._temp_history)) if self._temp_history else self.config.temperature,
        )

        return result

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        """Execute one decoding step: get logits, apply temperature, sample.

        For every active sequence the method:

        1. Queries the logit source for next-token logits.
        2. Applies constraints (repetition penalty, n-gram blocking).
        3. Computes the current temperature (possibly dynamic).
        4. Temperature-scales the logits and samples a token.
        5. Updates scores with the log-probability of the sampled token.

        Parameters
        ----------
        state : DecodingState
            Current generation state.
        logit_source : LogitSource
            Model providing next-token logits.

        Returns
        -------
        DecodingState
            Updated state.
        """
        active = state.active_indices()
        if not active:
            return state

        # Batch query
        batch_ids = [state.sequences[i] for i in active]
        all_logits = logit_source(batch_ids)  # (len(active), vocab)

        for batch_idx, seq_idx in enumerate(active):
            logits = np.asarray(all_logits[batch_idx], dtype=np.float64).ravel()

            # Apply constraints
            logits = self._apply_constraints(logits, state)

            # Repetition penalty (algorithm-specific; may differ from base)
            if self.config.repetition_penalty > 1.0:
                logits = self._apply_repetition_penalty(
                    logits,
                    state.sequences[seq_idx],
                    self.config.repetition_penalty,
                )

            # Determine temperature
            temperature = self._resolve_temperature(logits, state)

            # Record stats
            self._temp_history.append(temperature)
            ent = entropy_from_logits(logits)
            self._entropy_history.append(ent)

            # Sample token
            if self.config.gumbel_noise:
                token = self._gumbel_sample(logits, temperature)
            else:
                probs = self._apply_temperature(logits, temperature)
                token = categorical_sample(probs)

            # Score: log-probability under the temperature-scaled distribution
            log_probs = log_softmax_with_temperature(logits, temperature)
            token_logprob = float(log_probs[token])
            state.scores[seq_idx] += token_logprob

            # Append token
            state.update_sequence(seq_idx, token)

            # Check EOS
            if self.config.eos_token_id is not None and token == self.config.eos_token_id:
                prompt_len = state.metadata.get("prompt_length", 0)
                gen_len = len(state.sequences[seq_idx]) - prompt_len
                if gen_len >= self.config.min_new_tokens:
                    state.mark_finished(seq_idx)

            # Store logit history if list exists
            if state.logit_history is not None:
                state.logit_history.append(logits)

        return state

    # ------------------------------------------------------------------
    # Temperature application
    # ------------------------------------------------------------------

    def _apply_temperature(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Scale logits by temperature and return a probability distribution.

        Parameters
        ----------
        logits : np.ndarray
            1-D raw logits.
        temperature : float
            Sampling temperature (> 0).

        Returns
        -------
        np.ndarray
            Probability vector.
        """
        return softmax_with_temperature(logits, temperature)

    def _resolve_temperature(self, logits: np.ndarray, state: DecodingState) -> float:
        """Determine the temperature for this step.

        If ``dynamic_temperature`` is enabled, consults the schedule.
        Otherwise returns the static ``config.temperature``.

        Parameters
        ----------
        logits : np.ndarray
            Current logits (needed for adaptive schedules).
        state : DecodingState
            Current state (provides step count).

        Returns
        -------
        float
            Temperature value clamped to ``[1e-4, 100.0]``.
        """
        if self.config.dynamic_temperature:
            return self._dynamic_temperature(logits, state, self.config)
        return float(np.clip(self.config.temperature, 1e-4, 100.0))

    def _dynamic_temperature(
        self,
        logits: np.ndarray,
        state: DecodingState,
        config: TemperatureConfig,
    ) -> float:
        """Compute temperature from the configured schedule.

        Parameters
        ----------
        logits : np.ndarray
            Current logits.
        state : DecodingState
            Current state.
        config : TemperatureConfig
            Algorithm configuration.

        Returns
        -------
        float
            Dynamic temperature.
        """
        schedule = config.temp_schedule

        if schedule == "constant":
            return config.temperature
        elif schedule == "linear_decay":
            return self._linear_decay(
                state.step, config.max_new_tokens, config.temp_start, config.temp_end
            )
        elif schedule == "cosine":
            return self._cosine_schedule(
                state.step, config.max_new_tokens, config.temp_start, config.temp_end
            )
        elif schedule == "adaptive" or schedule == "entropy_based":
            return self._entropy_based_temperature(logits, config.entropy_target)
        else:
            # Fallback: use the TemperatureSchedule object
            return self._schedule.get_temperature(state.step, logits)

    @staticmethod
    def _linear_decay(step: int, total_steps: int, start: float, end: float) -> float:
        """Linearly decay temperature from *start* to *end*.

        Parameters
        ----------
        step : int
            Current step.
        total_steps : int
            Total generation steps.
        start : float
            Starting temperature.
        end : float
            Ending temperature.

        Returns
        -------
        float
            Interpolated temperature.
        """
        if total_steps <= 1:
            return start
        frac = min(step / (total_steps - 1), 1.0)
        return start + (end - start) * frac

    @staticmethod
    def _cosine_schedule(step: int, total_steps: int, start: float, end: float) -> float:
        """Cosine annealing from *start* to *end*.

        Parameters
        ----------
        step : int
            Current step.
        total_steps : int
            Total generation steps.
        start : float
            Starting temperature.
        end : float
            Ending temperature.

        Returns
        -------
        float
            Temperature.
        """
        frac = min(step / max(total_steps, 1), 1.0)
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * frac))

    @staticmethod
    def _entropy_based_temperature(logits: np.ndarray, target_entropy: float) -> float:
        """Find temperature that yields *target_entropy* for the given logits.

        Parameters
        ----------
        logits : np.ndarray
            1-D raw logits.
        target_entropy : float
            Desired entropy in nats.

        Returns
        -------
        float
            Temperature achieving approximately the target entropy.
        """
        return find_temperature_for_entropy(logits, target_entropy)

    def _gumbel_sample(self, logits: np.ndarray, temperature: float) -> int:
        """Sample a single token using the Gumbel-max trick.

        Parameters
        ----------
        logits : np.ndarray
            1-D raw logits.
        temperature : float
            Sampling temperature.

        Returns
        -------
        int
            Sampled token index.
        """
        tokens = self._gumbel.sample(logits, temperature, n=1)
        return tokens[0]

    # ------------------------------------------------------------------
    # Hyperparameter space
    # ------------------------------------------------------------------

    @classmethod
    def hyperparameter_space(cls) -> Dict[str, Any]:
        """Hyper-parameter search space for temperature sampling."""
        space = super().hyperparameter_space()
        space.update({
            "temperature": {"type": "float", "low": 0.1, "high": 3.0, "log": True},
            "dynamic_temperature": {"type": "categorical", "choices": [True, False]},
            "temp_schedule": {
                "type": "categorical",
                "choices": ["constant", "linear_decay", "cosine", "adaptive", "entropy_based"],
            },
            "temp_start": {"type": "float", "low": 0.5, "high": 3.0},
            "temp_end": {"type": "float", "low": 0.1, "high": 1.5},
            "entropy_target": {"type": "float", "low": 1.0, "high": 6.0},
            "gumbel_noise": {"type": "categorical", "choices": [True, False]},
            "repetition_penalty": {"type": "float", "low": 1.0, "high": 2.0},
        })
        return space

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_run_statistics(self) -> Dict[str, Any]:
        """Return statistics from the most recent generation run.

        Returns
        -------
        dict
            Keys: ``temp_history``, ``entropy_history``, ``avg_temp``,
            ``avg_entropy``, ``temp_std``, ``entropy_std``.
        """
        stats: Dict[str, Any] = {
            "temp_history": list(self._temp_history),
            "entropy_history": list(self._entropy_history),
        }
        if self._temp_history:
            arr = np.array(self._temp_history)
            stats["avg_temp"] = float(np.mean(arr))
            stats["temp_std"] = float(np.std(arr))
            stats["min_temp"] = float(np.min(arr))
            stats["max_temp"] = float(np.max(arr))
        if self._entropy_history:
            arr = np.array(self._entropy_history)
            stats["avg_entropy"] = float(np.mean(arr))
            stats["entropy_std"] = float(np.std(arr))
            stats["min_entropy"] = float(np.min(arr))
            stats["max_entropy"] = float(np.max(arr))
        return stats


# =========================================================================
# TemperatureAnalyzer
# =========================================================================


class TemperatureAnalyzer:
    """Utilities for analysing how temperature affects generation quality.

    All methods are stateless; the analyser simply requires a logit source
    and prompt ids.

    Parameters
    ----------
    vocab_size : int, optional
        Vocabulary size for effective-vocab-size calculations.
    """

    def __init__(self, vocab_size: int = 50257) -> None:
        self.vocab_size = vocab_size

    # ------------------------------------------------------------------
    # High-level analysis
    # ------------------------------------------------------------------

    def analyze_temperature_effect(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        temperatures: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Run a comprehensive temperature sweep.

        Produces entropy curves, effective vocabulary sizes, and sample
        diversity metrics for a range of temperatures.

        Parameters
        ----------
        logit_source : LogitSource
            Model providing logits.
        prompt_ids : List[int]
            Prompt token ids.
        temperatures : list of float, optional
            Temperatures to evaluate.  Defaults to a log-spaced range
            from 0.1 to 3.0.

        Returns
        -------
        dict
            Comprehensive analysis results with keys ``temperatures``,
            ``entropy``, ``effective_vocab_size``, ``top1_prob``,
            ``top5_prob``, ``perplexity``.
        """
        if temperatures is None:
            temperatures = list(np.round(np.geomspace(0.1, 3.0, 20), 4))

        # Get logits for the first next token
        logits = logit_source([prompt_ids])  # (1, vocab)
        logits = np.asarray(logits[0], dtype=np.float64).ravel()

        entropy_data = self.entropy_vs_temperature(logits, temperatures)
        evs_data = self.effective_vocab_size_vs_temperature(logits, temperatures)

        top1_probs: List[float] = []
        top5_probs: List[float] = []
        perplexities: List[float] = []

        for t in temperatures:
            probs = softmax_with_temperature(logits, t)
            sorted_probs = np.sort(probs)[::-1]
            top1_probs.append(float(sorted_probs[0]))
            top5_probs.append(float(np.sum(sorted_probs[:5])))

            # Perplexity = exp(entropy)
            h = entropy_data["entropies"][temperatures.index(t)]
            perplexities.append(float(np.exp(h)))

        return {
            "temperatures": temperatures,
            "entropy": entropy_data,
            "effective_vocab_size": evs_data,
            "top1_prob": top1_probs,
            "top5_prob": top5_probs,
            "perplexity": perplexities,
        }

    # ------------------------------------------------------------------
    # Entropy analysis
    # ------------------------------------------------------------------

    def entropy_vs_temperature(
        self,
        logits: np.ndarray,
        temperatures: List[float],
    ) -> Dict[str, Any]:
        """Compute entropy as a function of temperature.

        Parameters
        ----------
        logits : np.ndarray
            1-D raw logits.
        temperatures : list of float
            Temperatures to evaluate.

        Returns
        -------
        dict
            ``temperatures``, ``entropies``, ``entropies_bits`` (base-2).
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        entropies: List[float] = []

        for t in temperatures:
            p = softmax_with_temperature(logits, t)
            p_clip = np.clip(p, 1e-30, None)
            h = float(-np.sum(p * np.log(p_clip)))
            entropies.append(h)

        return {
            "temperatures": list(temperatures),
            "entropies": entropies,
            "entropies_bits": [h / math.log(2) for h in entropies],
        }

    # ------------------------------------------------------------------
    # Effective vocabulary size
    # ------------------------------------------------------------------

    def effective_vocab_size_vs_temperature(
        self,
        logits: np.ndarray,
        temperatures: List[float],
    ) -> Dict[str, Any]:
        """Compute effective vocabulary size (``exp(entropy)``) vs temperature.

        Parameters
        ----------
        logits : np.ndarray
            1-D raw logits.
        temperatures : list of float
            Temperatures to evaluate.

        Returns
        -------
        dict
            ``temperatures``, ``effective_sizes``, ``fraction_of_vocab``.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        sizes: List[float] = []

        for t in temperatures:
            p = softmax_with_temperature(logits, t)
            p_clip = np.clip(p, 1e-30, None)
            h = float(-np.sum(p * np.log(p_clip)))
            sizes.append(float(np.exp(h)))

        fractions = [s / self.vocab_size for s in sizes]

        return {
            "temperatures": list(temperatures),
            "effective_sizes": sizes,
            "fraction_of_vocab": fractions,
        }

    # ------------------------------------------------------------------
    # Diversity analysis
    # ------------------------------------------------------------------

    def diversity_vs_temperature(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        temps: Optional[List[float]] = None,
        n_samples: int = 50,
    ) -> Dict[str, Any]:
        """Measure sample diversity across temperatures.

        For each temperature, generates *n_samples* greedy-free samples
        and computes:

        - **unique_ratio**: fraction of unique first tokens.
        - **pairwise_jaccard**: average Jaccard similarity between pairs of
          full-sequence samples (using a short generation of 20 tokens).
        - **self_bleu**: average BLEU-like n-gram overlap (unigram) among
          generated samples.

        Parameters
        ----------
        logit_source : LogitSource
            Model providing logits.
        prompt_ids : List[int]
            Prompt token ids.
        temps : list of float, optional
            Temperatures to test.
        n_samples : int
            Samples per temperature.

        Returns
        -------
        dict
            Per-temperature diversity metrics.
        """
        if temps is None:
            temps = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]

        logits = logit_source([prompt_ids])  # (1, vocab)
        logits_1d = np.asarray(logits[0], dtype=np.float64).ravel()

        unique_ratios: List[float] = []
        mean_entropies: List[float] = []
        sample_sets: List[List[int]] = []

        for t in temps:
            probs = softmax_with_temperature(logits_1d, t)
            samples = sample_categorical(probs, n_samples)
            unique = len(set(samples))
            unique_ratios.append(unique / n_samples)

            p_clip = np.clip(probs, 1e-30, None)
            h = float(-np.sum(probs * np.log(p_clip)))
            mean_entropies.append(h)
            sample_sets.append(samples)

        # Pairwise Jaccard on first-token sets (as a proxy for diversity)
        pairwise_jaccards: List[float] = []
        for samples in sample_sets:
            if len(samples) < 2:
                pairwise_jaccards.append(0.0)
                continue
            jaccards: List[float] = []
            for i in range(min(len(samples), 30)):
                for j in range(i + 1, min(len(samples), 30)):
                    s1 = {samples[i]}
                    s2 = {samples[j]}
                    inter = len(s1 & s2)
                    union = len(s1 | s2)
                    jaccards.append(inter / union if union > 0 else 0.0)
            pairwise_jaccards.append(float(np.mean(jaccards)) if jaccards else 0.0)

        # Self-BLEU proxy: unigram overlap
        self_bleu_proxy: List[float] = []
        for samples in sample_sets:
            if len(samples) < 2:
                self_bleu_proxy.append(0.0)
                continue
            from collections import Counter
            counts = Counter(samples)
            total_pairs = n_samples * (n_samples - 1) / 2
            collision_pairs = sum(c * (c - 1) / 2 for c in counts.values())
            self_bleu_proxy.append(collision_pairs / total_pairs if total_pairs > 0 else 0.0)

        return {
            "temperatures": temps,
            "unique_ratio": unique_ratios,
            "mean_entropy": mean_entropies,
            "pairwise_jaccard": pairwise_jaccards,
            "self_bleu_proxy": self_bleu_proxy,
        }

    # ------------------------------------------------------------------
    # Optimal temperature
    # ------------------------------------------------------------------

    def optimal_temperature(
        self,
        logits: np.ndarray,
        target_metric: str = "entropy",
        target_value: float = 3.0,
    ) -> float:
        """Find the temperature that achieves a target metric value.

        Supported metrics:

        - ``"entropy"``: match target Shannon entropy (nats).
        - ``"effective_vocab_size"``: match target ``exp(H)``.
        - ``"top1_prob"``: match target probability of the most-likely token.

        Parameters
        ----------
        logits : np.ndarray
            1-D raw logits.
        target_metric : str
            Metric name.
        target_value : float
            Desired value.

        Returns
        -------
        float
            Temperature achieving the target, or ``NaN`` if not achievable.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()

        if target_metric == "entropy":
            return find_temperature_for_entropy(logits, target_value)

        elif target_metric == "effective_vocab_size":
            target_entropy = math.log(max(target_value, 1.0))
            return find_temperature_for_entropy(logits, target_entropy)

        elif target_metric == "top1_prob":
            # Binary search: higher T → lower top-1 prob
            lo, hi = 1e-4, 50.0
            for _ in range(200):
                mid = (lo + hi) / 2.0
                probs = softmax_with_temperature(logits, mid)
                top1 = float(np.max(probs))
                if abs(top1 - target_value) < 1e-5:
                    return mid
                if top1 > target_value:
                    lo = mid  # need higher T to reduce top-1
                else:
                    hi = mid
            return (lo + hi) / 2.0

        else:
            logger.warning("Unknown target metric '%s'", target_metric)
            return float("nan")

    # ------------------------------------------------------------------
    # Temperature landscape
    # ------------------------------------------------------------------

    def temperature_landscape(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
    ) -> Dict[str, Any]:
        """Compute a 2-D temperature landscape over steps and temperature values.

        For the first 20 generation steps (using greedy decoding to fix
        the context), records how entropy and top-1 probability vary
        across a grid of temperatures.

        Parameters
        ----------
        logit_source : LogitSource
            Model providing logits.
        prompt_ids : List[int]
            Prompt token ids.

        Returns
        -------
        dict
            ``temperatures`` (1-D), ``steps`` (1-D),
            ``entropy_grid`` (2-D: steps × temps),
            ``top1_grid`` (2-D: steps × temps),
            ``effective_vocab_grid`` (2-D: steps × temps).
        """
        temperatures = list(np.round(np.geomspace(0.1, 3.0, 15), 4))
        n_steps = 20

        entropy_grid: List[List[float]] = []
        top1_grid: List[List[float]] = []
        evs_grid: List[List[float]] = []

        current_ids = list(prompt_ids)

        for step in range(n_steps):
            logits = logit_source([current_ids])  # (1, vocab)
            logits_1d = np.asarray(logits[0], dtype=np.float64).ravel()

            step_entropies: List[float] = []
            step_top1: List[float] = []
            step_evs: List[float] = []

            for t in temperatures:
                probs = softmax_with_temperature(logits_1d, t)
                p_clip = np.clip(probs, 1e-30, None)
                h = float(-np.sum(probs * np.log(p_clip)))
                step_entropies.append(h)
                step_top1.append(float(np.max(probs)))
                step_evs.append(float(np.exp(h)))

            entropy_grid.append(step_entropies)
            top1_grid.append(step_top1)
            evs_grid.append(step_evs)

            # Advance context greedily
            greedy_token = int(np.argmax(logits_1d))
            current_ids.append(greedy_token)

        return {
            "temperatures": temperatures,
            "steps": list(range(n_steps)),
            "entropy_grid": entropy_grid,
            "top1_grid": top1_grid,
            "effective_vocab_grid": evs_grid,
        }


# =========================================================================
# Self-test
# =========================================================================


def _self_test() -> None:
    """Smoke test for the temperature module."""

    # -- Helper functions --
    logits = np.array([2.0, 1.0, 0.5, -1.0, 0.0])

    p1 = softmax_with_temperature(logits, 1.0)
    assert abs(p1.sum() - 1.0) < 1e-6
    p2 = softmax_with_temperature(logits, 0.5)
    assert p2[0] > p1[0], "Lower T should sharpen"
    p3 = softmax_with_temperature(logits, 2.0)
    assert p3[0] < p1[0], "Higher T should flatten"

    lp = log_softmax_with_temperature(logits, 1.0)
    assert np.allclose(np.exp(lp), p1, atol=1e-6)

    h = entropy_from_logits(logits)
    assert 0.0 < h < math.log(len(logits))

    samples = sample_categorical(p1, 100)
    assert len(samples) == 100
    assert all(0 <= s < len(logits) for s in samples)

    t_found = find_temperature_for_entropy(logits, h)
    assert abs(t_found - 1.0) < 0.1, f"Expected ~1.0, got {t_found}"

    # -- TemperatureConfig --
    cfg = TemperatureConfig(temperature=0.8)
    assert cfg.validate() == []
    bad_cfg = TemperatureConfig(temperature=-1.0)
    assert len(bad_cfg.validate()) > 0

    # -- TemperatureSchedule --
    sched = TemperatureSchedule("linear_decay", start=1.5, end=0.5, total_steps=100)
    assert abs(sched.get_temperature(0) - 1.5) < 1e-6
    assert abs(sched.get_temperature(99) - 0.5) < 0.02
    mid = sched.get_temperature(50)
    assert 0.5 < mid < 1.5

    sched_cos = TemperatureSchedule("cosine", start=2.0, end=0.5, total_steps=100)
    assert abs(sched_cos.get_temperature(0) - 2.0) < 1e-6

    sched_exp = TemperatureSchedule("exponential_decay", start=2.0, end=0.3, total_steps=100)
    t_exp_end = sched_exp.get_temperature(100)
    assert t_exp_end < 0.5

    sched_cyc = TemperatureSchedule("cyclical", start=2.0, end=0.5, total_steps=100)
    t_cyc = sched_cyc.get_temperature(0)
    assert 0.5 <= t_cyc <= 2.0

    sched_adapt = TemperatureSchedule("adaptive", start=1.0, end=0.5, total_steps=100, target_entropy=1.0)
    t_adapt = sched_adapt.get_temperature(0, logits)
    assert 0.0 < t_adapt < 100.0

    # -- GumbelSampler --
    gs = GumbelSampler(rng=np.random.default_rng(42))
    gsamples = gs.sample(logits, temperature=1.0, n=50)
    assert len(gsamples) == 50
    assert all(0 <= s < len(logits) for s in gsamples)

    soft = gs._gumbel_softmax(logits, 0.5)
    assert abs(soft.sum() - 1.0) < 1e-6

    hard = gs._straight_through(logits, 0.5)
    assert abs(hard.sum() - 1.0) < 1e-6
    assert np.max(hard) == 1.0

    # -- TemperatureSampling with a dummy logit source --
    vocab_size = 10
    dummy_logits = np.random.randn(vocab_size)

    def dummy_source(ids: List[List[int]]) -> np.ndarray:
        batch_size = len(ids)
        return np.tile(dummy_logits, (batch_size, 1))

    # Constant temperature
    config = TemperatureConfig(
        temperature=0.7,
        num_sequences=3,
        max_new_tokens=10,
        eos_token_id=None,
    )
    algo = TemperatureSampling(config)
    results = algo.generate(dummy_source, [0, 1, 2])
    assert len(results) == 3
    assert all(len(seq) > 0 for seq in results)

    stats = algo.get_run_statistics()
    assert "avg_temp" in stats
    assert abs(stats["avg_temp"] - 0.7) < 1e-6

    # Dynamic: linear decay
    config_dyn = TemperatureConfig(
        dynamic_temperature=True,
        temp_schedule="linear_decay",
        temp_start=1.5,
        temp_end=0.3,
        num_sequences=2,
        max_new_tokens=10,
    )
    algo_dyn = TemperatureSampling(config_dyn)
    results_dyn = algo_dyn.generate(dummy_source, [0])
    assert len(results_dyn) == 2
    dyn_stats = algo_dyn.get_run_statistics()
    assert dyn_stats["temp_history"][0] >= dyn_stats["temp_history"][-1]

    # Gumbel mode
    config_gumbel = TemperatureConfig(
        temperature=1.0,
        gumbel_noise=True,
        num_sequences=2,
        max_new_tokens=5,
    )
    algo_gumbel = TemperatureSampling(config_gumbel)
    results_gumbel = algo_gumbel.generate(dummy_source, [0])
    assert len(results_gumbel) == 2

    # -- TemperatureAnalyzer --
    analyzer = TemperatureAnalyzer(vocab_size=vocab_size)

    analysis = analyzer.analyze_temperature_effect(dummy_source, [0, 1])
    assert "entropy" in analysis
    assert "effective_vocab_size" in analysis
    assert len(analysis["top1_prob"]) == len(analysis["temperatures"])

    ent_data = analyzer.entropy_vs_temperature(dummy_logits, [0.5, 1.0, 2.0])
    assert len(ent_data["entropies"]) == 3
    assert ent_data["entropies"][0] < ent_data["entropies"][2]

    evs_data = analyzer.effective_vocab_size_vs_temperature(dummy_logits, [0.5, 1.0, 2.0])
    assert evs_data["effective_sizes"][0] < evs_data["effective_sizes"][2]

    div_data = analyzer.diversity_vs_temperature(dummy_source, [0], n_samples=20)
    assert len(div_data["unique_ratio"]) == len(div_data["temperatures"])

    opt_t = analyzer.optimal_temperature(dummy_logits, "entropy", 1.5)
    assert 0.0 < opt_t < 50.0

    opt_t2 = analyzer.optimal_temperature(dummy_logits, "effective_vocab_size", 5.0)
    assert 0.0 < opt_t2 < 50.0

    opt_t3 = analyzer.optimal_temperature(dummy_logits, "top1_prob", 0.3)
    assert 0.0 < opt_t3 < 50.0

    landscape = analyzer.temperature_landscape(dummy_source, [0])
    assert len(landscape["entropy_grid"]) == 20
    assert len(landscape["entropy_grid"][0]) == len(landscape["temperatures"])

    # -- Registry --
    assert AlgorithmRegistry.is_registered("temperature")
    algo2 = AlgorithmRegistry.create("temperature", TemperatureConfig(temperature=1.2, num_sequences=1, max_new_tokens=5))
    r2 = algo2.generate(dummy_source, [0])
    assert len(r2) == 1

    print("temperature.py self-test passed ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _self_test()
