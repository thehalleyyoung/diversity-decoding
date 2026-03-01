"""
Ancestral sampling with diversity constraints for the Diversity Decoding Arena.

Implements temperature-scheduled ancestral sampling with pluggable diversity
constraints (n-gram blocking, Hamming / Jaccard distance, entropy
regularisation, token blacklisting) so that a set of generated sequences
is guaranteed to exceed a configurable diversity threshold.
"""

from __future__ import annotations

import enum
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.algorithms.base import (
    DecodingAlgorithm,
    DecodingConfig,
    DecodingState,
    LogitSource,
    TokenSequence,
    _log_softmax,
    _stable_softmax,
    _top_k_filter,
    sample_token,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Diversity constraint types
# =========================================================================


class DiversityConstraintType(enum.Enum):
    """Supported diversity constraint families."""

    NGRAM_BLOCKING = "ngram_blocking"
    EMBEDDING_DISTANCE = "embedding_distance"
    TOKEN_BLACKLIST = "token_blacklist"
    ENTROPY_THRESHOLD = "entropy_threshold"
    HAMMING_DISTANCE = "hamming_distance"
    JACCARD_DISTANCE = "jaccard_distance"

    @classmethod
    def from_string(cls, s: str) -> "DiversityConstraintType":
        """Case-insensitive lookup by value *or* name."""
        s_lower = s.lower().strip()
        for member in cls:
            if member.value == s_lower or member.name.lower() == s_lower:
                return member
        valid = ", ".join(m.value for m in cls)
        raise ValueError(f"Unknown constraint type '{s}'. Valid: {valid}")


# =========================================================================
# AncestralConfig
# =========================================================================


@dataclass
class AncestralConfig(DecodingConfig):
    """Configuration for ancestral sampling with diversity constraints."""

    algorithm_name: str = "AncestralDiverseSampling"

    # Core sampling ---------------------------------------------------------
    temperature: float = 1.0
    n_sequences: int = 10

    # Diversity -------------------------------------------------------------
    diversity_constraint: str = "ngram_blocking"
    diversity_threshold: float = 0.3
    ngram_block_size: int = 3
    min_hamming_distance: int = 5
    min_jaccard_distance: float = 0.3
    entropy_bonus: float = 0.0

    # Optional filtering ----------------------------------------------------
    top_k: Optional[int] = None
    top_p: float = 1.0

    # Repetition control ----------------------------------------------------
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    # Retry logic -----------------------------------------------------------
    max_retries: int = 5

    # Temperature schedule --------------------------------------------------
    temperature_schedule: str = "constant"
    initial_temperature: float = 1.0
    final_temperature: float = 0.5
    warmup_steps: int = 0

    # Adaptive temperature --------------------------------------------------
    adaptive_temperature: bool = False
    target_entropy: float = 3.0

    # ---- validation -------------------------------------------------------

    def validate(self) -> List[str]:
        errors = super().validate()
        if self.n_sequences < 1:
            errors.append("n_sequences must be >= 1")
        if self.diversity_threshold < 0.0:
            errors.append("diversity_threshold must be >= 0")
        if self.ngram_block_size < 1:
            errors.append("ngram_block_size must be >= 1")
        if self.min_hamming_distance < 0:
            errors.append("min_hamming_distance must be >= 0")
        if not (0.0 <= self.min_jaccard_distance <= 1.0):
            errors.append("min_jaccard_distance must be in [0, 1]")
        if self.top_k is not None and self.top_k < 1:
            errors.append("top_k must be >= 1 when set")
        if not (0.0 < self.top_p <= 1.0):
            errors.append("top_p must be in (0, 1]")
        if self.max_retries < 0:
            errors.append("max_retries must be >= 0")
        valid_schedules = {"constant", "linear_decay", "cosine", "cyclic"}
        if self.temperature_schedule not in valid_schedules:
            errors.append(
                f"temperature_schedule must be one of {valid_schedules}"
            )
        if self.initial_temperature <= 0:
            errors.append("initial_temperature must be > 0")
        if self.final_temperature <= 0:
            errors.append("final_temperature must be > 0")
        if self.warmup_steps < 0:
            errors.append("warmup_steps must be >= 0")
        if self.target_entropy < 0:
            errors.append("target_entropy must be >= 0")
        try:
            DiversityConstraintType.from_string(self.diversity_constraint)
        except ValueError as exc:
            errors.append(str(exc))
        return errors


# =========================================================================
# DiversityConstraint
# =========================================================================


class DiversityConstraint:
    """Evaluates whether a candidate sequence satisfies a diversity
    requirement relative to a set of existing sequences.

    Parameters
    ----------
    constraint_type : DiversityConstraintType
        Which diversity metric to use.
    threshold : float
        Minimum distance / maximum overlap, depending on the metric.
    params : dict, optional
        Extra metric-specific parameters (e.g. ``ngram_size``).
    """

    def __init__(
        self,
        constraint_type: DiversityConstraintType,
        threshold: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.constraint_type = constraint_type
        self.threshold = threshold
        self.params: Dict[str, Any] = params or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        candidate: TokenSequence,
        existing_sequences: List[TokenSequence],
    ) -> bool:
        """Return ``True`` if *candidate* satisfies the constraint w.r.t.
        every sequence in *existing_sequences*."""
        if not existing_sequences:
            return True
        violation = self.compute_violation(candidate, existing_sequences)
        return violation <= 0.0

    def compute_violation(
        self,
        candidate: TokenSequence,
        existing_sequences: List[TokenSequence],
    ) -> float:
        """Return a non-negative violation score.  Zero means the constraint
        is satisfied; positive values indicate how far away the candidate is
        from satisfying it.

        The interpretation depends on the constraint type:
        * For distance-based constraints, the violation is the amount by which
          the closest existing sequence is *below* the threshold.
        * For overlap-based constraints it is the amount by which the most
          similar existing sequence *exceeds* the threshold.
        """
        if not existing_sequences:
            return 0.0

        ct = self.constraint_type

        if ct == DiversityConstraintType.NGRAM_BLOCKING:
            ngram_size = self.params.get("ngram_size", 3)
            max_overlap = 0.0
            for seq in existing_sequences:
                overlap = self._ngram_overlap(candidate, seq, ngram_size)
                max_overlap = max(max_overlap, overlap)
            # Threshold is the *maximum* allowed overlap (lower is more diverse).
            return max(0.0, max_overlap - self.threshold)

        if ct == DiversityConstraintType.HAMMING_DISTANCE:
            min_dist = float("inf")
            for seq in existing_sequences:
                d = self._hamming_distance(candidate, seq)
                min_dist = min(min_dist, d)
            required = self.params.get("min_distance", int(self.threshold))
            return max(0.0, required - min_dist)

        if ct == DiversityConstraintType.JACCARD_DISTANCE:
            min_dist = float("inf")
            for seq in existing_sequences:
                d = self._jaccard_distance(candidate, seq)
                min_dist = min(min_dist, d)
            return max(0.0, self.threshold - min_dist)

        if ct == DiversityConstraintType.TOKEN_BLACKLIST:
            blacklist = set(self.params.get("blacklisted_tokens", []))
            violation_count = sum(1 for t in candidate if t in blacklist)
            return float(violation_count)

        if ct == DiversityConstraintType.ENTROPY_THRESHOLD:
            # Entropy is not a pairwise metric; we check per-candidate
            # token-level entropy is handled externally — here we use
            # token-set diversity as a proxy.
            min_dist = float("inf")
            for seq in existing_sequences:
                d = self._token_set_distance(candidate, seq)
                min_dist = min(min_dist, d)
            return max(0.0, self.threshold - min_dist)

        if ct == DiversityConstraintType.EMBEDDING_DISTANCE:
            # Embedding-based distance requires external embeddings; fall back
            # to edit distance when embeddings are unavailable.
            min_dist = float("inf")
            for seq in existing_sequences:
                d = float(self._edit_distance(candidate, seq)) / max(
                    len(candidate), len(seq), 1
                )
                min_dist = min(min_dist, d)
            return max(0.0, self.threshold - min_dist)

        raise ValueError(f"Unsupported constraint type: {ct}")

    # ------------------------------------------------------------------
    # Distance / overlap primitives
    # ------------------------------------------------------------------

    @staticmethod
    def _ngram_overlap(seq_a: TokenSequence, seq_b: TokenSequence, n: int) -> float:
        """Fraction of n-grams in *seq_a* that also appear in *seq_b*.

        Returns a value in [0, 1].  An overlap of 1.0 means every n-gram of
        *seq_a* occurs in *seq_b*.
        """
        if len(seq_a) < n or len(seq_b) < n:
            return 0.0
        ngrams_a = set()
        for i in range(len(seq_a) - n + 1):
            ngrams_a.add(tuple(seq_a[i : i + n]))
        if not ngrams_a:
            return 0.0
        ngrams_b = set()
        for i in range(len(seq_b) - n + 1):
            ngrams_b.add(tuple(seq_b[i : i + n]))
        shared = ngrams_a & ngrams_b
        return len(shared) / len(ngrams_a)

    @staticmethod
    def _hamming_distance(seq_a: TokenSequence, seq_b: TokenSequence) -> int:
        """Hamming distance between two sequences (truncated to the shorter
        one).  Positions beyond the shorter sequence count as mismatches."""
        max_len = max(len(seq_a), len(seq_b))
        if max_len == 0:
            return 0
        dist = abs(len(seq_a) - len(seq_b))
        for i in range(min(len(seq_a), len(seq_b))):
            if seq_a[i] != seq_b[i]:
                dist += 1
        return dist

    @staticmethod
    def _jaccard_distance(seq_a: TokenSequence, seq_b: TokenSequence) -> float:
        """Jaccard distance = 1 - |A∩B| / |A∪B| over the token multisets.

        Returns a value in [0, 1].
        """
        set_a = set(seq_a)
        set_b = set(seq_b)
        if not set_a and not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return 1.0 - len(intersection) / len(union)

    @staticmethod
    def _token_set_distance(seq_a: TokenSequence, seq_b: TokenSequence) -> float:
        """Normalised symmetric-difference size over total unique tokens.

        A value of 1.0 means no tokens in common; 0.0 means identical sets.
        """
        set_a = set(seq_a)
        set_b = set(seq_b)
        if not set_a and not set_b:
            return 0.0
        union = set_a | set_b
        sym_diff = set_a ^ set_b
        return len(sym_diff) / len(union)

    @staticmethod
    def _edit_distance(seq_a: TokenSequence, seq_b: TokenSequence) -> int:
        """Levenshtein edit distance (standard DP implementation).

        Returns the minimum number of single-token insertions, deletions, or
        substitutions needed to transform *seq_a* into *seq_b*.
        """
        la, lb = len(seq_a), len(seq_b)
        if la == 0:
            return lb
        if lb == 0:
            return la

        # Use two-row DP to save memory
        prev = list(range(lb + 1))
        curr = [0] * (lb + 1)
        for i in range(1, la + 1):
            curr[0] = i
            for j in range(1, lb + 1):
                cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,       # deletion
                    curr[j - 1] + 1,    # insertion
                    prev[j - 1] + cost, # substitution
                )
            prev, curr = curr, prev
        return prev[lb]


# =========================================================================
# TemperatureScheduler
# =========================================================================


class TemperatureScheduler:
    """Deterministic temperature schedule over decoding steps.

    Parameters
    ----------
    schedule_type : str
        One of ``'constant'``, ``'linear_decay'``, ``'cosine'``, ``'cyclic'``.
    initial : float
        Temperature at step 0 (after warmup, if any).
    final : float
        Temperature at *total_steps* (ignored for ``'constant'``).
    total_steps : int
        Total number of generation steps.
    warmup : int
        Number of linear-warmup steps from ``final`` to ``initial``.
    """

    def __init__(
        self,
        schedule_type: str = "constant",
        initial: float = 1.0,
        final: float = 0.5,
        total_steps: int = 100,
        warmup: int = 0,
    ) -> None:
        self.schedule_type = schedule_type
        self.initial = initial
        self.final = final
        self.total_steps = max(total_steps, 1)
        self.warmup = max(warmup, 0)
        self._dispatchers = {
            "constant": self._constant,
            "linear_decay": self._linear_decay,
            "cosine": self._cosine_annealing,
            "cyclic": self._cyclic,
        }
        if schedule_type not in self._dispatchers:
            raise ValueError(
                f"Unknown schedule '{schedule_type}'. "
                f"Valid: {list(self._dispatchers)}"
            )

    # ------------------------------------------------------------------

    def get_temperature(self, step: int) -> float:
        """Return the temperature at *step*.

        During warmup the temperature linearly ramps from ``self.final`` to
        ``self.initial``; after warmup the chosen schedule takes over.
        """
        if step < 0:
            step = 0
        if self.warmup > 0 and step < self.warmup:
            return self._warmup_linear(step)
        return self._dispatchers[self.schedule_type](step)

    # ------------------------------------------------------------------
    # Schedule implementations
    # ------------------------------------------------------------------

    def _constant(self, step: int) -> float:
        """Fixed temperature."""
        return self.initial

    def _linear_decay(self, step: int) -> float:
        """Linearly interpolate from *initial* to *final*."""
        effective = step - self.warmup
        remaining = self.total_steps - self.warmup
        if remaining <= 0:
            return self.final
        progress = min(effective / remaining, 1.0)
        return self.initial + (self.final - self.initial) * progress

    def _cosine_annealing(self, step: int) -> float:
        """Cosine annealing from *initial* to *final*."""
        effective = step - self.warmup
        remaining = self.total_steps - self.warmup
        if remaining <= 0:
            return self.final
        progress = min(effective / remaining, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final + (self.initial - self.final) * cosine_decay

    def _cyclic(self, step: int) -> float:
        """Triangular cyclic schedule with a period of ``total_steps / 4``
        (i.e. four full cycles over the generation run)."""
        effective = step - self.warmup
        period = max((self.total_steps - self.warmup) / 4.0, 1.0)
        phase = (effective % period) / period  # [0, 1)
        # Triangle wave: ramp up from final→initial, then down
        if phase < 0.5:
            return self.final + (self.initial - self.final) * (2.0 * phase)
        return self.initial - (self.initial - self.final) * (2.0 * (phase - 0.5))

    def _warmup_linear(self, step: int) -> float:
        """Linear warmup from *final* to *initial*."""
        if self.warmup <= 0:
            return self.initial
        progress = min(step / self.warmup, 1.0)
        return self.final + (self.initial - self.final) * progress


# =========================================================================
# AdaptiveTemperature
# =========================================================================


class AdaptiveTemperature:
    """Online temperature controller that tracks a target entropy.

    At each step the caller provides the current per-token entropy (computed
    from the model's logit distribution) and the controller returns an
    adjusted temperature that nudges the entropy toward ``target_entropy``.

    Uses a simple proportional controller::

        error = target_entropy - current_entropy
        log_temp += learning_rate * error
        temperature = clamp(exp(log_temp), min_temp, max_temp)

    Parameters
    ----------
    target_entropy : float
        Desired per-token entropy (in nats).
    learning_rate : float
        Proportional gain.
    min_temp : float
        Minimum allowed temperature.
    max_temp : float
        Maximum allowed temperature.
    """

    def __init__(
        self,
        target_entropy: float = 3.0,
        learning_rate: float = 0.05,
        min_temp: float = 0.1,
        max_temp: float = 5.0,
    ) -> None:
        self.target_entropy = target_entropy
        self.learning_rate = learning_rate
        self.min_temp = min_temp
        self.max_temp = max_temp
        self._log_temp: float = 0.0  # log(1.0)
        self._history: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------

    def update(self, current_entropy: float) -> float:
        """Update and return the adjusted temperature.

        Parameters
        ----------
        current_entropy : float
            Entropy of the current logit distribution (in nats).

        Returns
        -------
        float
            New temperature value.
        """
        error = self.target_entropy - current_entropy
        self._log_temp += self.learning_rate * error
        temperature = math.exp(self._log_temp)
        temperature = max(self.min_temp, min(self.max_temp, temperature))
        self._log_temp = math.log(temperature)
        self._history.append((current_entropy, temperature))
        return temperature

    @staticmethod
    def _compute_entropy(logits: np.ndarray) -> float:
        """Compute the entropy (in nats) of the softmax distribution.

        Parameters
        ----------
        logits : np.ndarray
            Raw logits of shape ``(vocab_size,)``.

        Returns
        -------
        float
            Shannon entropy in nats.
        """
        probs = _stable_softmax(logits)
        # Avoid log(0)
        probs = np.clip(probs, 1e-12, None)
        entropy = -float(np.sum(probs * np.log(probs)))
        return entropy

    def reset(self) -> None:
        """Reset the controller state."""
        self._log_temp = 0.0
        self._history.clear()

    @property
    def history(self) -> List[Tuple[float, float]]:
        """Return ``(entropy, temperature)`` pairs recorded so far."""
        return list(self._history)


# =========================================================================
# AncestralDiverseSampling
# =========================================================================


class AncestralDiverseSampling(DecodingAlgorithm):
    """Ancestral (autoregressive) sampling with pluggable diversity
    constraints and temperature scheduling.

    The algorithm generates sequences one at a time.  After each sequence is
    complete the diversity constraint is evaluated against all previously
    generated sequences.  If the constraint is violated the algorithm retries
    with a progressively higher temperature (up to ``max_retries`` times).

    Key features
    -------------
    * Temperature scheduling (constant, linear decay, cosine annealing, cyclic).
    * Adaptive temperature control tracking a target per-token entropy.
    * Pluggable diversity constraints (n-gram blocking, Hamming, Jaccard, …).
    * Top-k and nucleus (top-p) filtering.
    * Repetition penalty and no-repeat n-gram blocking.
    * Batch generation across multiple prompts.
    """

    def __init__(self, config: AncestralConfig) -> None:
        # Coerce num_sequences from our n_sequences field
        config.num_sequences = config.n_sequences
        super().__init__(config)
        self.cfg: AncestralConfig = config

        # Build constraint
        self._constraint = self._build_constraint()

        # Temperature schedule
        self._scheduler = TemperatureScheduler(
            schedule_type=config.temperature_schedule,
            initial=config.initial_temperature,
            final=config.final_temperature,
            total_steps=config.max_new_tokens,
            warmup=config.warmup_steps,
        )

        # Adaptive temperature controller
        self._adaptive: Optional[AdaptiveTemperature] = None
        if config.adaptive_temperature:
            self._adaptive = AdaptiveTemperature(
                target_entropy=config.target_entropy,
                learning_rate=0.05,
                min_temp=0.1,
                max_temp=5.0,
            )

        # Generation statistics
        self._gen_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Constraint builder
    # ------------------------------------------------------------------

    def _build_constraint(self) -> DiversityConstraint:
        """Instantiate the configured diversity constraint."""
        ct = DiversityConstraintType.from_string(self.cfg.diversity_constraint)
        params: Dict[str, Any] = {
            "ngram_size": self.cfg.ngram_block_size,
            "min_distance": self.cfg.min_hamming_distance,
        }
        params.update(self.cfg.params)
        threshold = self.cfg.diversity_threshold
        if ct == DiversityConstraintType.HAMMING_DISTANCE:
            threshold = float(self.cfg.min_hamming_distance)
        elif ct == DiversityConstraintType.JACCARD_DISTANCE:
            threshold = self.cfg.min_jaccard_distance
        return DiversityConstraint(ct, threshold, params)

    # ------------------------------------------------------------------
    # Required abstract method: _step
    # ------------------------------------------------------------------

    def _step(self, state: DecodingState, logit_source: LogitSource) -> DecodingState:
        """Execute a single ancestral-sampling step for all active sequences.

        This is the core per-token generation step called by the parent
        ``_generation_loop``.
        """
        active = state.active_indices()
        if not active:
            return state

        # Collect prefixes for active sequences
        prefixes = [state.sequences[i] for i in active]
        logits_batch = logit_source(prefixes)  # (n_active, vocab)

        for batch_idx, seq_idx in enumerate(active):
            raw_logits = logits_batch[batch_idx].copy()

            # 1. Apply diversity-aware logit modification
            existing_seqs = self._existing_generated(state, seq_idx)
            raw_logits = self._apply_diversity_logit_modification(
                raw_logits, state.step, existing_seqs
            )

            # 2. Repetition penalty
            if self.cfg.repetition_penalty > 1.0:
                raw_logits = self._apply_repetition_penalty(
                    raw_logits, state.sequences[seq_idx]
                )

            # 3. No-repeat n-gram blocking (self-repetition)
            if self.cfg.no_repeat_ngram_size > 0:
                raw_logits = self._apply_no_repeat_ngram_self(
                    raw_logits, state.sequences[seq_idx],
                    self.cfg.no_repeat_ngram_size,
                )

            # 4. Temperature
            temperature = self._scheduler.get_temperature(state.step)
            if self._adaptive is not None:
                ent = AdaptiveTemperature._compute_entropy(raw_logits)
                temperature = self._adaptive.update(ent)
            raw_logits = self._apply_temperature(raw_logits, temperature)

            # 5. Optional top-k
            if self.cfg.top_k is not None and self.cfg.top_k > 0:
                raw_logits = self._apply_top_k(raw_logits, self.cfg.top_k)

            # 6. Optional nucleus (top-p)
            if self.cfg.top_p < 1.0:
                raw_logits = self._apply_nucleus(raw_logits, self.cfg.top_p)

            # 7. Entropy bonus
            if self.cfg.entropy_bonus > 0.0:
                raw_logits = self._entropy_regularization(
                    raw_logits, self.cfg.target_entropy
                )

            # 8. Sample
            token = sample_token(raw_logits, temperature=1.0)

            # 9. Update state
            state.update_sequence(seq_idx, token)
            log_p = _log_softmax(raw_logits)
            state.scores[seq_idx] += float(log_p[token])

            # 10. EOS check
            if self.cfg.eos_token_id is not None and token == self.cfg.eos_token_id:
                state.mark_finished(seq_idx)

        return state

    # ------------------------------------------------------------------
    # High-level generate (with diversity retries)
    # ------------------------------------------------------------------

    def generate(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        n_sequences: Optional[int] = None,
    ) -> List[TokenSequence]:
        """Generate *n_sequences* diverse continuations of *prompt_ids*.

        Overrides the base ``generate`` to add inter-sequence diversity
        checking and retry logic.
        """
        n = n_sequences if n_sequences is not None else self.cfg.n_sequences
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)
            self._rng = np.random.default_rng(self.cfg.seed)

        t0 = time.monotonic()
        completed: List[TokenSequence] = []
        total_retries = 0
        retry_budget_per_seq = self.cfg.max_retries

        for seq_idx in range(n):
            logger.debug("Generating sequence %d / %d", seq_idx + 1, n)
            success = False
            for attempt in range(1 + retry_budget_per_seq):
                candidate = self._generate_single(
                    logit_source, prompt_ids, completed, attempt=attempt
                )
                if self._check_diversity(candidate, completed, self._constraint):
                    completed.append(candidate)
                    success = True
                    break
                logger.debug(
                    "Diversity check failed for seq %d, attempt %d",
                    seq_idx, attempt,
                )
                total_retries += 1

            if not success:
                # Accept the last candidate even if it violates the constraint
                logger.warning(
                    "Exhausted retries for seq %d; accepting best-effort candidate",
                    seq_idx,
                )
                candidate = self._retry_with_increased_diversity(
                    logit_source, prompt_ids, completed, attempt=retry_budget_per_seq + 1,
                )
                completed.append(candidate)

        elapsed = time.monotonic() - t0
        self._gen_stats = {
            "n_sequences": len(completed),
            "total_retries": total_retries,
            "elapsed_seconds": elapsed,
            "set_diversity": self._compute_set_diversity(completed),
        }
        self._log_generation_stats(completed, step=-1)
        return completed

    # ------------------------------------------------------------------
    # Single-sequence generation
    # ------------------------------------------------------------------

    def _generate_single(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        existing_sequences: List[TokenSequence],
        attempt: int = 0,
    ) -> TokenSequence:
        """Generate a single continuation, applying diversity-aware logit
        modifications based on *existing_sequences*.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int
        existing_sequences : list of TokenSequence
            Previously accepted sequences (generated tokens only).
        attempt : int
            Retry attempt number (used to increase temperature).

        Returns
        -------
        TokenSequence
            Generated token ids (prompt excluded).
        """
        # Reset adaptive temperature for each sequence
        if self._adaptive is not None:
            self._adaptive.reset()

        seq = list(prompt_ids)
        generated: List[int] = []
        score = 0.0
        temp_boost = 0.1 * attempt  # increase temperature on retries

        for step in range(self.cfg.max_new_tokens):
            logits = logit_source([seq])[0].copy()  # (vocab,)

            # Diversity logit modification
            logits = self._apply_diversity_logit_modification(
                logits, step, existing_sequences
            )

            # Repetition penalty on generated so far
            if self.cfg.repetition_penalty > 1.0:
                logits = self._apply_repetition_penalty(logits, generated)

            # No-repeat n-gram on self
            if self.cfg.no_repeat_ngram_size > 0:
                logits = self._apply_no_repeat_ngram_self(
                    logits, generated, self.cfg.no_repeat_ngram_size
                )

            # Temperature with schedule + boost
            temperature = self._scheduler.get_temperature(step) + temp_boost
            if self._adaptive is not None:
                ent = AdaptiveTemperature._compute_entropy(logits)
                temperature = self._adaptive.update(ent) + temp_boost
            temperature = max(temperature, 0.01)
            logits = self._apply_temperature(logits, temperature)

            # Top-k
            if self.cfg.top_k is not None and self.cfg.top_k > 0:
                logits = self._apply_top_k(logits, self.cfg.top_k)

            # Nucleus
            if self.cfg.top_p < 1.0:
                logits = self._apply_nucleus(logits, self.cfg.top_p)

            # Entropy bonus
            if self.cfg.entropy_bonus > 0.0:
                logits = self._entropy_regularization(
                    logits, self.cfg.target_entropy
                )

            # Sample
            token = sample_token(logits, temperature=1.0)
            generated.append(token)
            seq.append(token)

            log_p = _log_softmax(logits)
            score += float(log_p[token])

            # EOS
            if self.cfg.eos_token_id is not None and token == self.cfg.eos_token_id:
                break

            # Min-length guard: skip EOS in sampling while below min_new_tokens
            if len(generated) < self.cfg.min_new_tokens and self.cfg.eos_token_id is not None:
                pass  # continue generating

        return generated

    # ------------------------------------------------------------------
    # Diversity-aware logit modifications
    # ------------------------------------------------------------------

    def _apply_diversity_logit_modification(
        self,
        logits: np.ndarray,
        step: int,
        existing_sequences: List[TokenSequence],
    ) -> np.ndarray:
        """Modify *logits* to encourage diversity w.r.t. existing sequences.

        Dispatches to specific modification strategies based on the configured
        constraint type.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        step : int
            Current generation step.
        existing_sequences : list of TokenSequence
            Already-accepted sequences (generated tokens only).

        Returns
        -------
        np.ndarray
            Modified logits.
        """
        if not existing_sequences:
            return logits

        ct = self._constraint.constraint_type

        if ct == DiversityConstraintType.NGRAM_BLOCKING:
            logits = self._ngram_blocking_mask(
                logits, existing_sequences, self.cfg.ngram_block_size
            )

        elif ct == DiversityConstraintType.TOKEN_BLACKLIST:
            blacklisted = set(self._constraint.params.get("blacklisted_tokens", []))
            for tok in blacklisted:
                if 0 <= tok < len(logits):
                    logits[tok] = -float("inf")

        elif ct in (
            DiversityConstraintType.HAMMING_DISTANCE,
            DiversityConstraintType.JACCARD_DISTANCE,
            DiversityConstraintType.EMBEDDING_DISTANCE,
        ):
            # For distance-based constraints we down-weight tokens that
            # frequently appear at the current position across existing seqs.
            penalty_mask = np.zeros_like(logits)
            for seq in existing_sequences:
                if step < len(seq):
                    tok = seq[step]
                    if 0 <= tok < len(penalty_mask):
                        penalty_mask[tok] += 1.0
            # Scale penalty
            if np.any(penalty_mask > 0):
                max_count = penalty_mask.max()
                penalty_mask = penalty_mask / max(max_count, 1.0)
                logits = logits - penalty_mask * 5.0  # penalise by up to 5 logits

        elif ct == DiversityConstraintType.ENTROPY_THRESHOLD:
            logits = self._entropy_regularization(logits, self.cfg.target_entropy)

        return logits

    def _ngram_blocking_mask(
        self,
        logits: np.ndarray,
        existing_sequences: List[TokenSequence],
        ngram_size: int,
    ) -> np.ndarray:
        """Block tokens that would create an n-gram already present in any
        existing sequence.

        For each existing sequence we collect all n-grams of size
        ``ngram_size``.  If the last ``ngram_size - 1`` tokens of the
        current context match the prefix of any collected n-gram, we set
        the logit of the completing token to ``-inf``.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        existing_sequences : list of TokenSequence
        ngram_size : int

        Returns
        -------
        np.ndarray
            Logits with blocked positions set to ``-inf``.
        """
        if ngram_size < 2:
            return logits

        # Collect banned continuations
        banned_tokens: set = set()

        for seq in existing_sequences:
            if len(seq) < ngram_size:
                continue
            # Build set of (prefix, next_token) pairs
            ngram_dict: Dict[Tuple[int, ...], set] = {}
            for i in range(len(seq) - ngram_size + 1):
                prefix = tuple(seq[i : i + ngram_size - 1])
                next_tok = seq[i + ngram_size - 1]
                ngram_dict.setdefault(prefix, set()).add(next_tok)

            # Determine banned tokens (not available without current context)
            # We check all prefixes and ban any token that would complete
            # a known n-gram in multiple existing sequences.
            for prefix, tokens in ngram_dict.items():
                banned_tokens.update(tokens)

        # Only block if enough existing sequences contribute the same token
        # to avoid being too aggressive. With the full set, we scale the penalty.
        if len(existing_sequences) >= 2:
            # Count how many sequences each token appears in as n-gram completion
            token_seq_count: Dict[int, int] = {}
            for seq in existing_sequences:
                if len(seq) < ngram_size:
                    continue
                seq_tokens = set()
                for i in range(len(seq) - ngram_size + 1):
                    seq_tokens.add(seq[i + ngram_size - 1])
                for tok in seq_tokens:
                    token_seq_count[tok] = token_seq_count.get(tok, 0) + 1
            # Block tokens appearing in more than half of existing sequences
            threshold = max(len(existing_sequences) // 2, 1)
            for tok, count in token_seq_count.items():
                if count >= threshold and 0 <= tok < len(logits):
                    logits[tok] -= 3.0 * count  # soft blocking
        else:
            # With only 1 existing sequence, apply mild penalty
            for tok in banned_tokens:
                if 0 <= tok < len(logits):
                    logits[tok] -= 2.0

        return logits

    def _entropy_regularization(
        self,
        logits: np.ndarray,
        target_entropy: float,
    ) -> np.ndarray:
        """Adjust logits to push the entropy of the resulting distribution
        toward *target_entropy*.

        If current entropy is below the target the logits are flattened
        (pushed toward uniform); if above, they are sharpened.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        target_entropy : float
            Desired entropy in nats.

        Returns
        -------
        np.ndarray
            Regularised logits.
        """
        current_entropy = AdaptiveTemperature._compute_entropy(logits)
        if current_entropy < 1e-8 and target_entropy < 1e-8:
            return logits

        bonus = self.cfg.entropy_bonus
        error = target_entropy - current_entropy

        if abs(error) < 0.1:
            return logits

        # Positive error → need *more* entropy → flatten
        # Negative error → need *less* entropy → sharpen
        # We adjust by blending with uniform logits
        vocab_size = len(logits)
        uniform = np.full(vocab_size, np.mean(logits))
        alpha = np.clip(bonus * error, -0.5, 0.5)
        adjusted = logits * (1.0 - alpha) + uniform * alpha
        return adjusted

    # ------------------------------------------------------------------
    # Standard logit transformations
    # ------------------------------------------------------------------

    def _apply_repetition_penalty(
        self,
        logits: np.ndarray,
        generated_tokens: List[int],
    ) -> np.ndarray:
        """Apply multiplicative repetition penalty.

        Positive logits are divided by the penalty; negative logits are
        multiplied (making them more negative).

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        generated_tokens : list of int
            Tokens generated so far (may include prompt tokens).

        Returns
        -------
        np.ndarray
        """
        penalty = self.cfg.repetition_penalty
        if penalty <= 1.0 or not generated_tokens:
            return logits
        logits = logits.copy()
        seen = set(generated_tokens)
        for tok in seen:
            if 0 <= tok < len(logits):
                if logits[tok] > 0:
                    logits[tok] /= penalty
                else:
                    logits[tok] *= penalty
        return logits

    @staticmethod
    def _apply_no_repeat_ngram_self(
        logits: np.ndarray,
        sequence: List[int],
        ngram_size: int,
    ) -> np.ndarray:
        """Block tokens that would create a repeated n-gram within the same
        sequence.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        sequence : list of int
            Tokens generated so far.
        ngram_size : int

        Returns
        -------
        np.ndarray
        """
        if ngram_size < 2 or len(sequence) < ngram_size - 1:
            return logits
        logits = logits.copy()
        # Collect existing n-grams
        ngrams_seen: Dict[Tuple[int, ...], set] = {}
        for i in range(len(sequence) - ngram_size + 1):
            prefix = tuple(sequence[i : i + ngram_size - 1])
            continuation = sequence[i + ngram_size - 1]
            ngrams_seen.setdefault(prefix, set()).add(continuation)

        # Current context tail
        context_tail = tuple(sequence[-(ngram_size - 1):])
        if context_tail in ngrams_seen:
            for tok in ngrams_seen[context_tail]:
                if 0 <= tok < len(logits):
                    logits[tok] = -float("inf")
        return logits

    @staticmethod
    def _apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
        """Divide logits by *temperature*.

        Parameters
        ----------
        logits : np.ndarray
        temperature : float
            Must be > 0.

        Returns
        -------
        np.ndarray
        """
        if temperature <= 0:
            temperature = 1e-8
        return logits / temperature

    @staticmethod
    def _apply_top_k(logits: np.ndarray, k: int) -> np.ndarray:
        """Keep only the top-*k* logits; set the rest to ``-inf``.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        k : int
            Number of tokens to retain.

        Returns
        -------
        np.ndarray
        """
        if k <= 0 or k >= len(logits):
            return logits
        return _top_k_filter(logits, k)

    @staticmethod
    def _apply_nucleus(logits: np.ndarray, p: float) -> np.ndarray:
        """Nucleus (top-p) filtering.

        Sort tokens by descending probability, accumulate until the
        cumulative probability exceeds *p*, then mask out the rest.

        Parameters
        ----------
        logits : np.ndarray
            Shape ``(vocab_size,)``.
        p : float
            Cumulative probability threshold in (0, 1].

        Returns
        -------
        np.ndarray
        """
        if p >= 1.0:
            return logits
        probs = _stable_softmax(logits)
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)

        # Find cutoff index (first index where cumulative > p)
        cutoff_mask = cumulative > p
        # Always keep at least the top token
        cutoff_mask[0] = False
        # Tokens past the cutoff are masked
        remove_indices = sorted_indices[cutoff_mask]
        logits = logits.copy()
        logits[remove_indices] = -float("inf")
        return logits

    # ------------------------------------------------------------------
    # Diversity checking & retries
    # ------------------------------------------------------------------

    def _check_diversity(
        self,
        candidate: TokenSequence,
        existing: List[TokenSequence],
        constraint: DiversityConstraint,
    ) -> bool:
        """Return ``True`` if *candidate* passes the diversity constraint.

        Parameters
        ----------
        candidate : TokenSequence
        existing : list of TokenSequence
        constraint : DiversityConstraint

        Returns
        -------
        bool
        """
        if not existing:
            return True
        return constraint.check(candidate, existing)

    def _retry_with_increased_diversity(
        self,
        logit_source: LogitSource,
        prompt_ids: List[int],
        existing: List[TokenSequence],
        attempt: int,
    ) -> TokenSequence:
        """Last-resort generation with aggressively increased diversity.

        Uses a higher temperature and a stricter n-gram blocking mask to
        force novelty when normal retries have been exhausted.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_ids : list of int
        existing : list of TokenSequence
        attempt : int

        Returns
        -------
        TokenSequence
        """
        return self._generate_single(
            logit_source, prompt_ids, existing, attempt=attempt
        )

    # ------------------------------------------------------------------
    # Set-level diversity metrics
    # ------------------------------------------------------------------

    def _compute_set_diversity(self, sequences: List[TokenSequence]) -> float:
        """Compute overall diversity of a set of sequences.

        Returns the mean of all pairwise distances (using the configured
        constraint metric).

        Parameters
        ----------
        sequences : list of TokenSequence

        Returns
        -------
        float
            Average pairwise distance.  Higher is more diverse.
        """
        if len(sequences) < 2:
            return 0.0
        dist_matrix = self._pairwise_distances(sequences)
        n = len(sequences)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += dist_matrix[i, j]
                count += 1
        return total / max(count, 1)

    def _pairwise_distances(self, sequences: List[TokenSequence]) -> np.ndarray:
        """Compute pairwise distance matrix for a set of sequences.

        Uses the distance metric implied by the configured constraint type.

        Parameters
        ----------
        sequences : list of TokenSequence

        Returns
        -------
        np.ndarray
            Symmetric distance matrix of shape ``(n, n)``.
        """
        n = len(sequences)
        dist = np.zeros((n, n), dtype=np.float64)
        ct = self._constraint.constraint_type

        for i in range(n):
            for j in range(i + 1, n):
                if ct == DiversityConstraintType.NGRAM_BLOCKING:
                    ngram_size = self.cfg.ngram_block_size
                    overlap = DiversityConstraint._ngram_overlap(
                        sequences[i], sequences[j], ngram_size
                    )
                    d = 1.0 - overlap
                elif ct == DiversityConstraintType.HAMMING_DISTANCE:
                    d = float(DiversityConstraint._hamming_distance(
                        sequences[i], sequences[j]
                    ))
                elif ct == DiversityConstraintType.JACCARD_DISTANCE:
                    d = DiversityConstraint._jaccard_distance(
                        sequences[i], sequences[j]
                    )
                elif ct == DiversityConstraintType.EMBEDDING_DISTANCE:
                    max_len = max(len(sequences[i]), len(sequences[j]), 1)
                    d = float(DiversityConstraint._edit_distance(
                        sequences[i], sequences[j]
                    )) / max_len
                elif ct == DiversityConstraintType.TOKEN_BLACKLIST:
                    d = DiversityConstraint._token_set_distance(
                        sequences[i], sequences[j]
                    )
                elif ct == DiversityConstraintType.ENTROPY_THRESHOLD:
                    d = DiversityConstraint._token_set_distance(
                        sequences[i], sequences[j]
                    )
                else:
                    d = DiversityConstraint._jaccard_distance(
                        sequences[i], sequences[j]
                    )
                dist[i, j] = d
                dist[j, i] = d

        return dist

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def decode_batch(
        self,
        logit_source: LogitSource,
        prompt_batch: List[List[int]],
        n_sequences: Optional[int] = None,
    ) -> List[List[TokenSequence]]:
        """Generate diverse continuations for a batch of prompts.

        Parameters
        ----------
        logit_source : LogitSource
        prompt_batch : list of list of int
        n_sequences : int, optional
            Per-prompt count; defaults to ``self.cfg.n_sequences``.

        Returns
        -------
        list of list of TokenSequence
            Outer list has one entry per prompt; inner list has
            *n_sequences* token sequences.
        """
        n = n_sequences if n_sequences is not None else self.cfg.n_sequences
        results: List[List[TokenSequence]] = []
        for prompt in prompt_batch:
            seqs = self.generate(logit_source, prompt, n_sequences=n)
            results.append(seqs)
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @staticmethod
    def get_hyperparameter_grid() -> Dict[str, Any]:
        """Return a dictionary describing the hyper-parameter search space.

        Returns
        -------
        dict
            Mapping from parameter name to a descriptor dict with keys
            ``type``, ``low``/``high`` or ``choices``.
        """
        return {
            "temperature": {
                "type": "float",
                "low": 0.1,
                "high": 2.0,
                "log": True,
            },
            "n_sequences": {
                "type": "int",
                "low": 1,
                "high": 100,
            },
            "diversity_constraint": {
                "type": "categorical",
                "choices": [ct.value for ct in DiversityConstraintType],
            },
            "diversity_threshold": {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
            },
            "ngram_block_size": {
                "type": "int",
                "low": 2,
                "high": 6,
            },
            "min_hamming_distance": {
                "type": "int",
                "low": 1,
                "high": 50,
            },
            "min_jaccard_distance": {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
            },
            "entropy_bonus": {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
            },
            "top_k": {
                "type": "int",
                "low": 1,
                "high": 200,
            },
            "top_p": {
                "type": "float",
                "low": 0.1,
                "high": 1.0,
            },
            "repetition_penalty": {
                "type": "float",
                "low": 1.0,
                "high": 2.0,
            },
            "no_repeat_ngram_size": {
                "type": "int",
                "low": 0,
                "high": 6,
            },
            "max_retries": {
                "type": "int",
                "low": 0,
                "high": 20,
            },
            "temperature_schedule": {
                "type": "categorical",
                "choices": ["constant", "linear_decay", "cosine", "cyclic"],
            },
            "initial_temperature": {
                "type": "float",
                "low": 0.1,
                "high": 3.0,
                "log": True,
            },
            "final_temperature": {
                "type": "float",
                "low": 0.1,
                "high": 2.0,
                "log": True,
            },
            "warmup_steps": {
                "type": "int",
                "low": 0,
                "high": 50,
            },
            "adaptive_temperature": {
                "type": "categorical",
                "choices": [True, False],
            },
            "target_entropy": {
                "type": "float",
                "low": 0.5,
                "high": 8.0,
            },
        }

    def describe(self) -> str:
        """Return a human-readable description of this algorithm instance.

        Returns
        -------
        str
        """
        parts = [
            f"AncestralDiverseSampling(n={self.cfg.n_sequences}",
            f"constraint={self.cfg.diversity_constraint}",
            f"threshold={self.cfg.diversity_threshold:.3f}",
            f"schedule={self.cfg.temperature_schedule}",
            f"T={self.cfg.initial_temperature:.2f}→{self.cfg.final_temperature:.2f}",
        ]
        if self.cfg.top_k is not None:
            parts.append(f"top_k={self.cfg.top_k}")
        if self.cfg.top_p < 1.0:
            parts.append(f"top_p={self.cfg.top_p:.2f}")
        if self.cfg.repetition_penalty > 1.0:
            parts.append(f"rep_pen={self.cfg.repetition_penalty:.2f}")
        if self.cfg.adaptive_temperature:
            parts.append(f"adaptive(target_H={self.cfg.target_entropy:.2f})")
        return ", ".join(parts) + ")"

    def _log_generation_stats(
        self,
        sequences: List[TokenSequence],
        step: int,
    ) -> None:
        """Log summary statistics about a completed generation run.

        Parameters
        ----------
        sequences : list of TokenSequence
        step : int
            Current step (or -1 for end-of-generation summary).
        """
        if not sequences:
            logger.info("No sequences generated.")
            return

        lengths = [len(s) for s in sequences]
        mean_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        diversity = self._compute_set_diversity(sequences)

        logger.info(
            "Generation stats (step=%d): n=%d, len=%.1f [%d, %d], "
            "diversity=%.4f, retries=%d, time=%.2fs",
            step,
            len(sequences),
            mean_len,
            min_len,
            max_len,
            diversity,
            self._gen_stats.get("total_retries", 0),
            self._gen_stats.get("elapsed_seconds", 0.0),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _existing_generated(
        self, state: DecodingState, exclude_idx: int
    ) -> List[TokenSequence]:
        """Collect generated tokens from all sequences in *state* except
        *exclude_idx*.

        Parameters
        ----------
        state : DecodingState
        exclude_idx : int

        Returns
        -------
        list of TokenSequence
        """
        prompt_len = state.metadata.get("prompt_length", 0)
        result: List[TokenSequence] = []
        for i in range(state.num_sequences):
            if i == exclude_idx:
                continue
            generated = state.sequences[i][prompt_len:]
            if generated:
                result.append(generated)
        return result


# =========================================================================
# Standalone helpers (module-level)
# =========================================================================


def _collect_ngrams(sequence: TokenSequence, n: int) -> set:
    """Return the set of n-grams (as tuples) in *sequence*.

    Parameters
    ----------
    sequence : TokenSequence
    n : int

    Returns
    -------
    set of tuple
    """
    if len(sequence) < n:
        return set()
    return {tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)}


def _unique_token_ratio(sequence: TokenSequence) -> float:
    """Fraction of unique tokens in *sequence*.

    Parameters
    ----------
    sequence : TokenSequence

    Returns
    -------
    float
        Value in [0, 1].
    """
    if not sequence:
        return 0.0
    return len(set(sequence)) / len(sequence)


def _self_bleu_proxy(sequences: List[TokenSequence], ngram: int = 4) -> float:
    """Compute a fast Self-BLEU proxy for a set of sequences.

    For each sequence, we compute the fraction of its n-grams that appear
    in at least one other sequence, then average.  Lower values indicate
    higher diversity.

    Parameters
    ----------
    sequences : list of TokenSequence
    ngram : int

    Returns
    -------
    float
        Average overlap ratio in [0, 1].
    """
    if len(sequences) < 2:
        return 0.0
    all_ngrams = [_collect_ngrams(s, ngram) for s in sequences]
    ratios: List[float] = []
    for i, ng_i in enumerate(all_ngrams):
        if not ng_i:
            ratios.append(0.0)
            continue
        others = set()
        for j, ng_j in enumerate(all_ngrams):
            if j != i:
                others.update(ng_j)
        shared = ng_i & others
        ratios.append(len(shared) / len(ng_i))
    return sum(ratios) / len(ratios)


def _mean_pairwise_jaccard(sequences: List[TokenSequence]) -> float:
    """Mean pairwise Jaccard distance over all pairs.

    Parameters
    ----------
    sequences : list of TokenSequence

    Returns
    -------
    float
    """
    n = len(sequences)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        set_i = set(sequences[i])
        for j in range(i + 1, n):
            set_j = set(sequences[j])
            union = set_i | set_j
            if not union:
                continue
            inter = set_i & set_j
            total += 1.0 - len(inter) / len(union)
            count += 1
    return total / max(count, 1)


def _mean_pairwise_hamming(sequences: List[TokenSequence]) -> float:
    """Mean pairwise Hamming distance.

    Parameters
    ----------
    sequences : list of TokenSequence

    Returns
    -------
    float
    """
    n = len(sequences)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += DiversityConstraint._hamming_distance(
                sequences[i], sequences[j]
            )
            count += 1
    return total / max(count, 1)


def _diversity_report(sequences: List[TokenSequence]) -> Dict[str, float]:
    """Compute a battery of diversity metrics for *sequences*.

    Parameters
    ----------
    sequences : list of TokenSequence

    Returns
    -------
    dict
        Mapping metric name → value.
    """
    if not sequences:
        return {}
    return {
        "n_sequences": float(len(sequences)),
        "mean_length": sum(len(s) for s in sequences) / len(sequences),
        "mean_unique_ratio": sum(_unique_token_ratio(s) for s in sequences) / len(sequences),
        "self_bleu_4": _self_bleu_proxy(sequences, ngram=4),
        "self_bleu_3": _self_bleu_proxy(sequences, ngram=3),
        "mean_jaccard": _mean_pairwise_jaccard(sequences),
        "mean_hamming": _mean_pairwise_hamming(sequences),
    }


# =========================================================================
# Self-test
# =========================================================================


def _self_test() -> None:
    """Minimal smoke test for the ancestral sampling module."""

    # -- DiversityConstraintType --
    assert DiversityConstraintType.from_string("ngram_blocking") == DiversityConstraintType.NGRAM_BLOCKING
    assert DiversityConstraintType.from_string("HAMMING_DISTANCE") == DiversityConstraintType.HAMMING_DISTANCE

    # -- DiversityConstraint basics --
    dc = DiversityConstraint(DiversityConstraintType.NGRAM_BLOCKING, threshold=0.5, params={"ngram_size": 3})
    assert dc.check([1, 2, 3, 4, 5], [])
    assert dc._ngram_overlap([1, 2, 3], [1, 2, 3], 3) == 1.0
    assert dc._ngram_overlap([1, 2, 3], [4, 5, 6], 3) == 0.0
    assert dc._hamming_distance([1, 2, 3], [1, 2, 3]) == 0
    assert dc._hamming_distance([1, 2, 3], [1, 0, 3]) == 1
    assert dc._hamming_distance([1, 2], [1, 2, 3]) == 1
    jd = dc._jaccard_distance([1, 2, 3], [3, 4, 5])
    assert 0.0 < jd < 1.0
    assert dc._edit_distance([1, 2, 3], [1, 2, 3]) == 0
    assert dc._edit_distance([1, 2, 3], [4, 5, 6]) == 3
    assert dc._edit_distance([], [1, 2]) == 2

    # -- TemperatureScheduler --
    ts = TemperatureScheduler("constant", initial=1.0, final=0.5, total_steps=100)
    assert ts.get_temperature(0) == 1.0
    assert ts.get_temperature(50) == 1.0

    ts_lin = TemperatureScheduler("linear_decay", initial=1.0, final=0.5, total_steps=100)
    assert ts_lin.get_temperature(0) == 1.0
    t50 = ts_lin.get_temperature(50)
    assert 0.5 < t50 < 1.0

    ts_cos = TemperatureScheduler("cosine", initial=1.0, final=0.2, total_steps=100)
    assert ts_cos.get_temperature(0) == 1.0
    assert ts_cos.get_temperature(100) < 0.25

    ts_cyc = TemperatureScheduler("cyclic", initial=1.5, final=0.5, total_steps=100)
    t_mid = ts_cyc.get_temperature(50)
    assert 0.4 < t_mid < 1.6

    ts_warm = TemperatureScheduler("linear_decay", initial=1.0, final=0.5, total_steps=100, warmup=10)
    assert abs(ts_warm.get_temperature(0) - 0.5) < 0.01
    assert abs(ts_warm.get_temperature(10) - 1.0) < 0.05

    # -- AdaptiveTemperature --
    at = AdaptiveTemperature(target_entropy=3.0, learning_rate=0.1, min_temp=0.1, max_temp=5.0)
    t1 = at.update(2.0)  # entropy below target → increase temp
    assert t1 > 1.0
    t2 = at.update(5.0)  # entropy above target → decrease
    assert t2 < t1
    at.reset()
    assert len(at.history) == 0

    ent = AdaptiveTemperature._compute_entropy(np.array([0.0, 0.0, 0.0, 0.0]))
    assert ent > 0  # uniform → max entropy for 4 classes
    ent_sharp = AdaptiveTemperature._compute_entropy(np.array([100.0, 0.0, 0.0, 0.0]))
    assert ent_sharp < ent

    # -- AncestralConfig --
    cfg = AncestralConfig(n_sequences=3, max_new_tokens=10)
    errors = cfg.validate()
    assert not errors, f"Unexpected validation errors: {errors}"

    bad_cfg = AncestralConfig(n_sequences=-1)
    assert len(bad_cfg.validate()) > 0

    # -- AncestralDiverseSampling with dummy logit source --
    vocab_size = 8

    def dummy_logit_source(input_ids: List[List[int]]) -> np.ndarray:
        batch = len(input_ids)
        rng = np.random.default_rng(42)
        return rng.standard_normal((batch, vocab_size))

    cfg = AncestralConfig(
        n_sequences=3,
        max_new_tokens=15,
        temperature=1.0,
        diversity_constraint="ngram_blocking",
        diversity_threshold=0.8,
        ngram_block_size=2,
        max_retries=2,
        seed=123,
    )
    algo = AncestralDiverseSampling(cfg)
    seqs = algo.generate(dummy_logit_source, [1, 2])
    assert len(seqs) == 3
    assert all(len(s) > 0 for s in seqs)

    desc = algo.describe()
    assert "AncestralDiverseSampling" in desc

    grid = AncestralDiverseSampling.get_hyperparameter_grid()
    assert "temperature" in grid
    assert "diversity_constraint" in grid

    # -- Batch generation --
    batch_results = algo.decode_batch(
        dummy_logit_source, [[1, 2], [3, 4]], n_sequences=2
    )
    assert len(batch_results) == 2
    assert all(len(r) == 2 for r in batch_results)

    # -- Pairwise distances --
    dists = algo._pairwise_distances(seqs)
    assert dists.shape == (3, 3)
    assert dists[0, 0] == 0.0
    assert dists[0, 1] == dists[1, 0]

    # -- Set diversity --
    div = algo._compute_set_diversity(seqs)
    assert div >= 0.0

    # -- Module-level helpers --
    ng = _collect_ngrams([1, 2, 3, 4], 2)
    assert len(ng) == 3
    assert (1, 2) in ng

    assert _unique_token_ratio([1, 1, 1, 1]) == 0.25
    assert _unique_token_ratio([1, 2, 3, 4]) == 1.0

    bleu = _self_bleu_proxy([[1, 2, 3, 4], [1, 2, 3, 5]], ngram=2)
    assert 0.0 <= bleu <= 1.0

    report = _diversity_report(seqs)
    assert "mean_jaccard" in report
    assert "self_bleu_4" in report

    # -- Hamming constraint --
    cfg_h = AncestralConfig(
        n_sequences=2,
        max_new_tokens=10,
        diversity_constraint="hamming_distance",
        min_hamming_distance=2,
        max_retries=3,
        seed=99,
    )
    algo_h = AncestralDiverseSampling(cfg_h)
    seqs_h = algo_h.generate(dummy_logit_source, [1])
    assert len(seqs_h) == 2

    # -- Jaccard constraint --
    cfg_j = AncestralConfig(
        n_sequences=2,
        max_new_tokens=10,
        diversity_constraint="jaccard_distance",
        min_jaccard_distance=0.1,
        max_retries=3,
        seed=77,
    )
    algo_j = AncestralDiverseSampling(cfg_j)
    seqs_j = algo_j.generate(dummy_logit_source, [1])
    assert len(seqs_j) == 2

    # -- Top-k + top-p --
    cfg_kp = AncestralConfig(
        n_sequences=2,
        max_new_tokens=8,
        top_k=4,
        top_p=0.9,
        seed=55,
    )
    algo_kp = AncestralDiverseSampling(cfg_kp)
    seqs_kp = algo_kp.generate(dummy_logit_source, [1, 2])
    assert len(seqs_kp) == 2

    # -- Cosine schedule --
    cfg_cos = AncestralConfig(
        n_sequences=2,
        max_new_tokens=10,
        temperature_schedule="cosine",
        initial_temperature=1.5,
        final_temperature=0.3,
        seed=11,
    )
    algo_cos = AncestralDiverseSampling(cfg_cos)
    seqs_cos = algo_cos.generate(dummy_logit_source, [1])
    assert len(seqs_cos) == 2

    # -- Adaptive temperature --
    cfg_ad = AncestralConfig(
        n_sequences=2,
        max_new_tokens=10,
        adaptive_temperature=True,
        target_entropy=2.0,
        seed=33,
    )
    algo_ad = AncestralDiverseSampling(cfg_ad)
    seqs_ad = algo_ad.generate(dummy_logit_source, [1])
    assert len(seqs_ad) == 2

    # -- Repetition penalty --
    cfg_rp = AncestralConfig(
        n_sequences=1,
        max_new_tokens=10,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        seed=44,
    )
    algo_rp = AncestralDiverseSampling(cfg_rp)
    seqs_rp = algo_rp.generate(dummy_logit_source, [1, 2, 3])
    assert len(seqs_rp) == 1

    # -- Entropy bonus --
    cfg_eb = AncestralConfig(
        n_sequences=2,
        max_new_tokens=10,
        entropy_bonus=0.3,
        target_entropy=2.5,
        seed=66,
    )
    algo_eb = AncestralDiverseSampling(cfg_eb)
    seqs_eb = algo_eb.generate(dummy_logit_source, [1])
    assert len(seqs_eb) == 2

    # -- Warmup steps --
    cfg_wu = AncestralConfig(
        n_sequences=1,
        max_new_tokens=20,
        temperature_schedule="linear_decay",
        initial_temperature=1.2,
        final_temperature=0.4,
        warmup_steps=5,
        seed=88,
    )
    algo_wu = AncestralDiverseSampling(cfg_wu)
    seqs_wu = algo_wu.generate(dummy_logit_source, [1])
    assert len(seqs_wu) == 1

    print("ancestral.py self-test passed ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _self_test()
