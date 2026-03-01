"""
Hyperparameter sweep orchestration for the Diversity Decoding Arena.

Implements grid, random, Bayesian, successive halving, Hyperband,
and Latin hypercube sweep strategies with a from-scratch GP surrogate,
checkpointing, resumption, parameter importance analysis, and result export.

Dependencies: stdlib + numpy only.
"""

from __future__ import annotations

import enum
import hashlib
import itertools
import json
import math
import os
import pickle
import time
import copy
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SweepStrategy(enum.Enum):
    """Supported sweep strategies."""

    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"
    SUCCESSIVE_HALVING = "successive_halving"
    HYPERBAND = "hyperband"
    LATIN_HYPERCUBE = "latin_hypercube"


class ParameterType(enum.Enum):
    """Supported parameter types."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    LOG_UNIFORM = "log_uniform"
    INT_UNIFORM = "int_uniform"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class ParameterSpec:
    """Specification for a single hyperparameter."""

    name: str
    param_type: ParameterType
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None
    log_scale: bool = False
    step: Optional[float] = None

    def __post_init__(self) -> None:
        if self.param_type == ParameterType.CATEGORICAL:
            if not self.choices:
                raise ValueError(
                    f"Parameter '{self.name}': CATEGORICAL requires 'choices'."
                )
        elif self.param_type in (
            ParameterType.CONTINUOUS,
            ParameterType.LOG_UNIFORM,
            ParameterType.INT_UNIFORM,
            ParameterType.DISCRETE,
        ):
            if self.low is None or self.high is None:
                raise ValueError(
                    f"Parameter '{self.name}': {self.param_type.value} requires 'low' and 'high'."
                )
            if self.low > self.high:
                raise ValueError(
                    f"Parameter '{self.name}': low ({self.low}) > high ({self.high})."
                )
        if self.param_type == ParameterType.LOG_UNIFORM:
            if self.low is not None and self.low <= 0:
                raise ValueError(
                    f"Parameter '{self.name}': LOG_UNIFORM requires positive low bound."
                )
            self.log_scale = True

    def grid_values(self, n_points: int = 10) -> List[Any]:
        """Return a list of grid points for this parameter."""
        if self.param_type == ParameterType.CATEGORICAL:
            return list(self.choices)  # type: ignore[arg-type]
        if self.param_type == ParameterType.DISCRETE:
            if self.step is not None:
                vals = []
                v = self.low
                while v <= self.high + 1e-12:
                    vals.append(v)
                    v += self.step
                return vals
            return list(
                np.linspace(self.low, self.high, min(n_points, int(self.high - self.low + 1)))
            )
        if self.param_type == ParameterType.INT_UNIFORM:
            lo, hi = int(self.low), int(self.high)
            total = hi - lo + 1
            if total <= n_points:
                return list(range(lo, hi + 1))
            step = max(1, total // n_points)
            vals = list(range(lo, hi + 1, step))
            if vals[-1] != hi:
                vals.append(hi)
            return vals
        if self.param_type == ParameterType.LOG_UNIFORM:
            return list(np.exp(np.linspace(np.log(self.low), np.log(self.high), n_points)))
        # CONTINUOUS
        if self.step is not None:
            vals = []
            v = self.low
            while v <= self.high + 1e-12:
                vals.append(round(v, 12))
                v += self.step
            return vals
        return list(np.linspace(self.low, self.high, n_points))

    def sample(self, rng: np.random.RandomState) -> Any:
        """Draw a single random sample."""
        if self.param_type == ParameterType.CATEGORICAL:
            return self.choices[rng.randint(len(self.choices))]  # type: ignore[index]
        if self.param_type == ParameterType.INT_UNIFORM:
            return int(rng.randint(int(self.low), int(self.high) + 1))
        if self.param_type == ParameterType.LOG_UNIFORM:
            return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        if self.param_type == ParameterType.DISCRETE:
            if self.step is not None:
                n_steps = int(round((self.high - self.low) / self.step))
                idx = rng.randint(0, n_steps + 1)
                return self.low + idx * self.step
            return float(rng.uniform(self.low, self.high))
        # CONTINUOUS
        if self.step is not None:
            n_steps = int(round((self.high - self.low) / self.step))
            idx = rng.randint(0, n_steps + 1)
            return self.low + idx * self.step
        return float(rng.uniform(self.low, self.high))

    def encode(self, value: Any) -> float:
        """Encode a parameter value to [0, 1] for the GP."""
        if self.param_type == ParameterType.CATEGORICAL:
            idx = self.choices.index(value)  # type: ignore[union-attr]
            n = len(self.choices)  # type: ignore[arg-type]
            return idx / max(n - 1, 1)
        if self.param_type == ParameterType.LOG_UNIFORM:
            log_val = np.log(float(value))
            log_lo = np.log(self.low)
            log_hi = np.log(self.high)
            denom = log_hi - log_lo
            if abs(denom) < 1e-15:
                return 0.5
            return float((log_val - log_lo) / denom)
        lo, hi = float(self.low), float(self.high)
        denom = hi - lo
        if abs(denom) < 1e-15:
            return 0.5
        return float((float(value) - lo) / denom)

    def decode(self, unit_value: float) -> Any:
        """Decode a [0, 1] value back to the original parameter space."""
        unit_value = float(np.clip(unit_value, 0.0, 1.0))
        if self.param_type == ParameterType.CATEGORICAL:
            n = len(self.choices)  # type: ignore[arg-type]
            idx = int(round(unit_value * (n - 1)))
            idx = max(0, min(idx, n - 1))
            return self.choices[idx]  # type: ignore[index]
        if self.param_type == ParameterType.LOG_UNIFORM:
            log_lo = np.log(self.low)
            log_hi = np.log(self.high)
            return float(np.exp(log_lo + unit_value * (log_hi - log_lo)))
        lo, hi = float(self.low), float(self.high)
        raw = lo + unit_value * (hi - lo)
        if self.param_type == ParameterType.INT_UNIFORM:
            return int(round(raw))
        if self.step is not None:
            n_steps = round((raw - lo) / self.step)
            raw = lo + n_steps * self.step
            raw = float(np.clip(raw, lo, hi))
        if self.param_type == ParameterType.DISCRETE:
            if self.step is not None:
                return raw
            return float(raw)
        return float(raw)

    def contains(self, value: Any) -> bool:
        """Check if *value* is within this parameter's domain."""
        if self.param_type == ParameterType.CATEGORICAL:
            return value in self.choices  # type: ignore[operator]
        v = float(value)
        return self.low - 1e-9 <= v <= self.high + 1e-9


@dataclass
class SweepConfig:
    """Full configuration for a hyperparameter sweep."""

    strategy: SweepStrategy = SweepStrategy.RANDOM
    parameters: List[ParameterSpec] = field(default_factory=list)
    n_trials: int = 50
    n_parallel: int = 1
    budget: float = float("inf")
    objective: str = "loss"
    maximize: bool = False
    seed: int = 42
    early_stopping_rounds: int = 0
    patience: int = 10
    checkpoint_dir: Optional[str] = None

    def param_by_name(self, name: str) -> ParameterSpec:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"No parameter named '{name}'")


@dataclass
class TrialResult:
    """Result of a single trial evaluation."""

    trial_id: str
    parameters: Dict[str, Any]
    objective_value: float
    all_metrics: Dict[str, float] = field(default_factory=dict)
    duration: float = 0.0
    status: str = "completed"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "parameters": self.parameters,
            "objective_value": self.objective_value,
            "all_metrics": self.all_metrics,
            "duration": self.duration,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class SweepState:
    """Mutable state of an in-progress sweep."""

    completed_trials: List[TrialResult] = field(default_factory=list)
    best_trial: Optional[TrialResult] = None
    iteration: int = 0
    start_time: float = field(default_factory=time.time)
    total_budget_used: float = 0.0

    def update_best(self, trial: TrialResult, maximize: bool) -> None:
        if self.best_trial is None:
            self.best_trial = trial
            return
        if maximize:
            if trial.objective_value > self.best_trial.objective_value:
                self.best_trial = trial
        else:
            if trial.objective_value < self.best_trial.objective_value:
                self.best_trial = trial

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completed_trials": [t.to_dict() for t in self.completed_trials],
            "best_trial": self.best_trial.to_dict() if self.best_trial else None,
            "iteration": self.iteration,
            "start_time": self.start_time,
            "total_budget_used": self.total_budget_used,
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _trial_id(params: Dict[str, Any], idx: int) -> str:
    """Deterministic trial id from parameters and index."""
    raw = json.dumps(params, sort_keys=True, default=str) + f":{idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _format_value(v: Any, width: int = 12) -> str:
    """Format a value for display in a table."""
    if isinstance(v, float):
        if abs(v) < 1e-3 or abs(v) > 1e5:
            s = f"{v:.4e}"
        else:
            s = f"{v:.6f}"
    else:
        s = str(v)
    return s[:width].ljust(width)


def _evaluate_trial(
    evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    params: Dict[str, Any],
    trial_idx: int,
    objective: str,
) -> TrialResult:
    """Run a single trial and wrap the result."""
    tid = _trial_id(params, trial_idx)
    t0 = time.time()
    try:
        metrics = evaluate_fn(params)
        obj = metrics.get(objective, float("nan"))
        return TrialResult(
            trial_id=tid,
            parameters=dict(params),
            objective_value=float(obj),
            all_metrics=dict(metrics),
            duration=time.time() - t0,
            status="completed",
        )
    except Exception as exc:
        return TrialResult(
            trial_id=tid,
            parameters=dict(params),
            objective_value=float("inf"),
            all_metrics={},
            duration=time.time() - t0,
            status="failed",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# ParameterGrid
# ---------------------------------------------------------------------------


class ParameterGrid:
    """Generates parameter configurations via grid, random, LHS, or Sobol."""

    def __init__(self, specs: List[ParameterSpec]) -> None:
        self.specs = list(specs)
        self._names = [s.name for s in self.specs]

    # -- Grid search ----------------------------------------------------------

    def grid_search(self, n_points_per_param: int = 10) -> Iterator[Dict[str, Any]]:
        """Yield every point on the Cartesian-product grid."""
        axes: List[List[Any]] = []
        for spec in self.specs:
            axes.append(spec.grid_values(n_points_per_param))
        for combo in itertools.product(*axes):
            yield dict(zip(self._names, combo))

    def total_combinations(self, n_points_per_param: int = 10) -> int:
        """Number of grid points."""
        total = 1
        for spec in self.specs:
            total *= len(spec.grid_values(n_points_per_param))
        return total

    # -- Random search --------------------------------------------------------

    def random_search(
        self, n_trials: int, seed: int = 42
    ) -> Iterator[Dict[str, Any]]:
        """Yield *n_trials* uniformly random configurations."""
        rng = np.random.RandomState(seed)
        for _ in range(n_trials):
            yield self._sample_random(rng)

    def _sample_random(self, rng: np.random.RandomState) -> Dict[str, Any]:
        return {spec.name: spec.sample(rng) for spec in self.specs}

    # -- Latin Hypercube Sampling ---------------------------------------------

    def latin_hypercube(
        self, n_trials: int, seed: int = 42
    ) -> Iterator[Dict[str, Any]]:
        """Yield *n_trials* Latin-hypercube samples."""
        rng = np.random.RandomState(seed)
        d = len(self.specs)
        if d == 0 or n_trials == 0:
            return

        # Build an n_trials x d matrix of unit-cube samples via LHS
        intervals = np.linspace(0, 1, n_trials + 1)
        unit_samples = np.empty((n_trials, d))
        for j in range(d):
            perm = rng.permutation(n_trials)
            for i in range(n_trials):
                lo_bin = intervals[perm[i]]
                hi_bin = intervals[perm[i] + 1]
                unit_samples[i, j] = rng.uniform(lo_bin, hi_bin)

        for i in range(n_trials):
            params: Dict[str, Any] = {}
            for j, spec in enumerate(self.specs):
                params[spec.name] = spec.decode(unit_samples[i, j])
            yield params

    # -- Sobol quasi-random sequence ------------------------------------------

    def sobol_sequence(self, n_trials: int) -> Iterator[Dict[str, Any]]:
        """Yield *n_trials* points from a Sobol sequence (simple implementation)."""
        d = len(self.specs)
        if d == 0 or n_trials == 0:
            return

        # Direction numbers for up to 21 dimensions (standard Joe-Kuo tables simplified)
        direction_numbers = self._sobol_direction_numbers(d)

        points = np.zeros((n_trials, d))
        x = np.zeros(d, dtype=np.int64)
        for i in range(n_trials):
            if i == 0:
                points[i] = 0.0
            else:
                c = self._rightmost_zero_bit(i - 1)
                for j in range(d):
                    if c < len(direction_numbers[j]):
                        x[j] ^= direction_numbers[j][c]
                    else:
                        x[j] ^= direction_numbers[j][-1]
                points[i] = x / (2**32)

            params: Dict[str, Any] = {}
            for j, spec in enumerate(self.specs):
                params[spec.name] = spec.decode(float(points[i, j]))
            yield params

    @staticmethod
    def _rightmost_zero_bit(n: int) -> int:
        """Index of the rightmost zero bit of *n*."""
        i = 0
        while n & 1:
            n >>= 1
            i += 1
        return i

    @staticmethod
    def _sobol_direction_numbers(d: int) -> List[List[int]]:
        """Generate simple direction-number vectors for *d* dimensions."""
        bits = 32
        result: List[List[int]] = []
        for dim in range(d):
            v: List[int] = []
            for i in range(bits):
                if dim == 0:
                    v.append(1 << (bits - 1 - i))
                else:
                    seed_val = (dim * 37 + i * 13 + 7) % (1 << (i + 1))
                    seed_val = seed_val | 1  # ensure odd
                    v.append(seed_val << (bits - 1 - i))
            result.append(v)
        return result

    # -- Helpers --------------------------------------------------------------

    def _sample_parameter(self, spec: ParameterSpec, rng: np.random.RandomState) -> Any:
        """Sample a single parameter value."""
        return spec.sample(rng)

    def validate_point(self, params: Dict[str, Any]) -> bool:
        """Check that *params* is within the defined domain."""
        for spec in self.specs:
            if spec.name not in params:
                return False
            if not spec.contains(params[spec.name]):
                return False
        return True


# ---------------------------------------------------------------------------
# BayesianOptimizer  (GP-based, pure numpy)
# ---------------------------------------------------------------------------


class BayesianOptimizer:
    """Bayesian optimization with a Gaussian-process surrogate built from scratch."""

    def __init__(
        self,
        specs: List[ParameterSpec],
        n_initial: int = 5,
        acquisition_fn: str = "ei",
        seed: int = 42,
    ) -> None:
        self.specs = list(specs)
        self.n_initial = max(n_initial, 2)
        self.acquisition_fn = acquisition_fn.lower()
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.d = len(specs)

        # Observed data
        self._X: List[np.ndarray] = []
        self._y: List[float] = []

        # Kernel hyper-parameters (log length_scale, log variance, log noise)
        self._kernel_params: Dict[str, float] = {
            "length_scale": 1.0,
            "variance": 1.0,
            "noise": 1e-4,
        }

        # Grid helper for initial points
        self._grid = ParameterGrid(specs)

        # Cache for convergence tracking
        self._best_so_far: List[float] = []

    # -- Public API -----------------------------------------------------------

    def suggest(self, n_suggestions: int = 1) -> List[Dict[str, Any]]:
        """Suggest the next *n_suggestions* parameter configurations."""
        n_obs = len(self._X)
        if n_obs < self.n_initial:
            # Not enough data yet – return LHS or random points
            needed = min(n_suggestions, self.n_initial - n_obs)
            suggestions: List[Dict[str, Any]] = []
            for pt in self._grid.latin_hypercube(needed, seed=self.seed + n_obs):
                suggestions.append(pt)
            remaining = n_suggestions - len(suggestions)
            for _ in range(remaining):
                suggestions.append(self._grid._sample_random(self.rng))
            return suggestions

        X_train = np.array(self._X)
        y_train = np.array(self._y)

        # Normalise targets for numerical stability
        y_mean = np.mean(y_train)
        y_std = max(np.std(y_train), 1e-8)
        y_norm = (y_train - y_mean) / y_std

        # Fit kernel hyper-parameters
        self._kernel_params = self._optimize_kernel_params(X_train, y_norm)

        suggestions = []
        for _ in range(n_suggestions):
            x_best = self._optimize_acquisition(X_train, y_norm)
            suggestions.append(self._decode_params(x_best))
            # Add a small jitter copy to X_train so subsequent suggestions differ
            X_train = np.vstack([X_train, x_best.reshape(1, -1)])
            pred_mean, _ = self._gaussian_process_predict(
                np.array(self._X), y_norm, x_best.reshape(1, -1), self._kernel_params
            )
            y_norm = np.append(y_norm, pred_mean[0])

        return suggestions

    def observe(self, params: Dict[str, Any], objective: float) -> None:
        """Record an observation."""
        x = self._encode_params(params)
        self._X.append(x)
        self._y.append(objective)
        best = min(self._y)
        self._best_so_far.append(best)

    # -- Surrogate (GP) -------------------------------------------------------

    def _fit_surrogate(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Fit the GP surrogate and return kernel parameters."""
        return self._optimize_kernel_params(X, y)

    def _gaussian_process_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        kernel_params: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation at *X_test*."""
        ls = kernel_params["length_scale"]
        var = kernel_params["variance"]
        noise = kernel_params.get("noise", 1e-4)

        K = self._rbf_kernel(X_train, X_train, ls, var)
        K += noise * np.eye(len(K))

        # Cholesky solve for numerical stability
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-3 * np.eye(len(K))
            L = np.linalg.cholesky(K)

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

        K_s = self._rbf_kernel(X_train, X_test, ls, var)
        K_ss = self._rbf_kernel(X_test, X_test, ls, var)

        mu = K_s.T @ alpha
        v = np.linalg.solve(L, K_s)
        cov = K_ss - v.T @ v
        std = np.sqrt(np.maximum(np.diag(cov), 1e-12))

        return mu, std

    def _rbf_kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        length_scale: float,
        variance: float,
    ) -> np.ndarray:
        """Squared-exponential (RBF) kernel."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        sq_dist = (
            np.sum(X1**2, axis=1, keepdims=True)
            + np.sum(X2**2, axis=1)
            - 2.0 * X1 @ X2.T
        )
        sq_dist = np.maximum(sq_dist, 0.0)
        return variance * np.exp(-0.5 * sq_dist / max(length_scale**2, 1e-12))

    def _log_marginal_likelihood(
        self,
        X: np.ndarray,
        y: np.ndarray,
        kernel_params: Dict[str, float],
    ) -> float:
        """Negative log marginal likelihood of the GP (lower is better)."""
        ls = kernel_params["length_scale"]
        var = kernel_params["variance"]
        noise = kernel_params.get("noise", 1e-4)

        K = self._rbf_kernel(X, X, ls, var)
        K += noise * np.eye(len(K))

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e12

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        n = len(y)
        nll = (
            0.5 * y @ alpha
            + np.sum(np.log(np.diag(L)))
            + 0.5 * n * np.log(2 * np.pi)
        )
        return float(nll)

    def _optimize_kernel_params(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Grid search over a small set of kernel hyper-parameter combos."""
        best_nll = float("inf")
        best_params = dict(self._kernel_params)

        ls_candidates = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
        var_candidates = [0.1, 0.5, 1.0, 2.0, 5.0]
        noise_candidates = [1e-5, 1e-4, 1e-3, 1e-2]

        for ls in ls_candidates:
            for var in var_candidates:
                for noise in noise_candidates:
                    kp = {"length_scale": ls, "variance": var, "noise": noise}
                    nll = self._log_marginal_likelihood(X, y, kp)
                    if nll < best_nll:
                        best_nll = nll
                        best_params = kp

        return best_params

    # -- Acquisition functions ------------------------------------------------

    def _acquisition_ei(
        self, x: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, best_y: float
    ) -> float:
        """Expected Improvement."""
        mu, std = self._gaussian_process_predict(
            X_train, y_train, x.reshape(1, -1), self._kernel_params
        )
        mu_val = float(mu[0])
        std_val = float(std[0])
        if std_val < 1e-12:
            return 0.0
        z = (best_y - mu_val) / std_val
        ei = std_val * (z * self._standard_normal_cdf(z) + self._standard_normal_pdf(z))
        return float(ei)

    def _acquisition_ucb(
        self, x: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, kappa: float = 2.0
    ) -> float:
        """Upper Confidence Bound (negated for minimisation)."""
        mu, std = self._gaussian_process_predict(
            X_train, y_train, x.reshape(1, -1), self._kernel_params
        )
        return float(-(mu[0] - kappa * std[0]))

    def _acquisition_pi(
        self, x: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, best_y: float
    ) -> float:
        """Probability of Improvement."""
        mu, std = self._gaussian_process_predict(
            X_train, y_train, x.reshape(1, -1), self._kernel_params
        )
        mu_val = float(mu[0])
        std_val = float(std[0])
        if std_val < 1e-12:
            return 0.0
        z = (best_y - mu_val) / std_val
        return float(self._standard_normal_cdf(z))

    def _optimize_acquisition(
        self, X_train: np.ndarray, y_train: np.ndarray, n_restarts: int = 200
    ) -> np.ndarray:
        """Find the point that maximises the acquisition function via random shooting + local refinement."""
        best_y = float(np.min(y_train))
        best_acq = -float("inf")
        best_x = self.rng.rand(self.d)

        # Random candidates
        candidates = self.rng.rand(n_restarts, self.d)
        # Also add LHS candidates for better coverage
        lhs_n = min(n_restarts, 50)
        lhs_pts = self._lhs_unit_cube(lhs_n, self.d)
        candidates = np.vstack([candidates, lhs_pts])

        for i in range(len(candidates)):
            x_c = candidates[i]
            if self.acquisition_fn == "ei":
                acq = self._acquisition_ei(x_c, X_train, y_train, best_y)
            elif self.acquisition_fn == "ucb":
                acq = self._acquisition_ucb(x_c, X_train, y_train)
            elif self.acquisition_fn == "pi":
                acq = self._acquisition_pi(x_c, X_train, y_train, best_y)
            else:
                acq = self._acquisition_ei(x_c, X_train, y_train, best_y)

            if acq > best_acq:
                best_acq = acq
                best_x = x_c.copy()

        # Local refinement around the best candidate
        best_x = self._local_refine(best_x, X_train, y_train, best_y)

        return best_x

    def _local_refine(
        self,
        x0: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        best_y: float,
        n_iter: int = 30,
        step: float = 0.05,
    ) -> np.ndarray:
        """Simple coordinate-wise local search to refine acquisition optimum."""
        x = x0.copy()

        def _eval_acq(xv: np.ndarray) -> float:
            if self.acquisition_fn == "ei":
                return self._acquisition_ei(xv, X_train, y_train, best_y)
            elif self.acquisition_fn == "ucb":
                return self._acquisition_ucb(xv, X_train, y_train)
            elif self.acquisition_fn == "pi":
                return self._acquisition_pi(xv, X_train, y_train, best_y)
            return self._acquisition_ei(xv, X_train, y_train, best_y)

        current_acq = _eval_acq(x)
        for _ in range(n_iter):
            improved = False
            for j in range(self.d):
                for delta in [step, -step, step / 2, -step / 2]:
                    x_new = x.copy()
                    x_new[j] = np.clip(x_new[j] + delta, 0.0, 1.0)
                    acq_new = _eval_acq(x_new)
                    if acq_new > current_acq:
                        x = x_new
                        current_acq = acq_new
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
        return x

    def _lhs_unit_cube(self, n: int, d: int) -> np.ndarray:
        """Generate LHS samples in [0,1]^d."""
        result = np.empty((n, d))
        for j in range(d):
            perm = self.rng.permutation(n)
            for i in range(n):
                lo = perm[i] / n
                hi = (perm[i] + 1) / n
                result[i, j] = self.rng.uniform(lo, hi)
        return result

    # -- Encoding / decoding --------------------------------------------------

    def _encode_params(self, params_dict: Dict[str, Any]) -> np.ndarray:
        """Encode a parameter dictionary into a unit-cube vector."""
        x = np.zeros(self.d)
        for j, spec in enumerate(self.specs):
            x[j] = spec.encode(params_dict[spec.name])
        return x

    def _decode_params(self, x: np.ndarray) -> Dict[str, Any]:
        """Decode a unit-cube vector into a parameter dictionary."""
        params: Dict[str, Any] = {}
        for j, spec in enumerate(self.specs):
            params[spec.name] = spec.decode(float(x[j]))
        return params

    # -- Convergence ----------------------------------------------------------

    def convergence_curve(self) -> List[float]:
        """Return best-so-far objective after each observation."""
        return list(self._best_so_far)

    # -- Math helpers ---------------------------------------------------------

    @staticmethod
    def _standard_normal_cdf(z: float) -> float:
        return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))

    @staticmethod
    def _standard_normal_pdf(z: float) -> float:
        return float(math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi))


# ---------------------------------------------------------------------------
# SuccessiveHalving
# ---------------------------------------------------------------------------


class SuccessiveHalving:
    """Successive Halving for hyperparameter optimisation."""

    def __init__(
        self,
        specs: List[ParameterSpec],
        max_budget: float = 81.0,
        reduction_factor: int = 3,
        min_budget: float = 1.0,
        seed: int = 42,
        maximize: bool = False,
        objective: str = "loss",
    ) -> None:
        self.specs = list(specs)
        self.max_budget = max_budget
        self.eta = reduction_factor
        self.min_budget = min_budget
        self.seed = seed
        self.maximize = maximize
        self.objective = objective
        self.rng = np.random.RandomState(seed)
        self._grid = ParameterGrid(specs)

    def run(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    ) -> List[TrialResult]:
        """Execute the successive halving procedure."""
        allocations = self._allocate_budgets()
        n_initial = allocations[0][0]
        # Sample initial configs
        configs: List[Dict[str, Any]] = []
        for pt in self._grid.random_search(n_initial, seed=self.seed):
            configs.append(pt)

        all_results: List[TrialResult] = []
        trial_counter = 0

        for rung_idx, (n_configs, budget_per) in enumerate(allocations):
            rung_results: List[TrialResult] = []
            for cfg in configs[:n_configs]:
                cfg_with_budget = dict(cfg)
                cfg_with_budget["_budget"] = budget_per
                result = _evaluate_trial(evaluate_fn, cfg_with_budget, trial_counter, self.objective)
                rung_results.append(result)
                all_results.append(result)
                trial_counter += 1

            if rung_idx < len(allocations) - 1:
                next_n = allocations[rung_idx + 1][0]
                configs = self._promote(rung_results, next_n)

        return all_results

    def _allocate_budgets(self) -> List[Tuple[int, float]]:
        """Compute (n_configs, budget_per_config) for each rung."""
        s_max = max(
            0,
            int(np.floor(np.log(self.max_budget / self.min_budget) / np.log(self.eta))),
        )
        n = int(np.ceil((s_max + 1) * self.eta**s_max / (s_max + 1)))
        allocations: List[Tuple[int, float]] = []
        for i in range(s_max + 1):
            n_i = max(1, int(np.floor(n * self.eta ** (-i))))
            r_i = self.min_budget * self.eta**i
            r_i = min(r_i, self.max_budget)
            allocations.append((n_i, r_i))
        return allocations

    def _promote(
        self, results: List[TrialResult], keep_n: int
    ) -> List[Dict[str, Any]]:
        """Keep the top *keep_n* configurations."""
        valid = [r for r in results if r.status == "completed"]
        valid.sort(key=lambda r: r.objective_value, reverse=self.maximize)
        keep_n = min(keep_n, len(valid))
        promoted: List[Dict[str, Any]] = []
        for r in valid[:keep_n]:
            cfg = dict(r.parameters)
            cfg.pop("_budget", None)
            promoted.append(cfg)
        return promoted


# ---------------------------------------------------------------------------
# Hyperband (wraps SuccessiveHalving)
# ---------------------------------------------------------------------------


class _HyperbandBracket:
    """A single bracket in Hyperband."""

    def __init__(
        self,
        specs: List[ParameterSpec],
        s: int,
        s_max: int,
        max_budget: float,
        eta: int,
        seed: int,
        maximize: bool,
        objective: str,
    ) -> None:
        self.s = s
        self.s_max = s_max
        self.max_budget = max_budget
        self.eta = eta
        self.specs = specs
        self.maximize = maximize
        self.objective = objective
        self.seed = seed

        n = int(np.ceil((s_max + 1) / (s + 1) * eta**s))
        r = max_budget * eta ** (-s)
        self.n_initial = max(n, 1)
        self.min_budget = max(r, 1.0)

    def run(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    ) -> List[TrialResult]:
        sh = SuccessiveHalving(
            specs=self.specs,
            max_budget=self.max_budget,
            reduction_factor=self.eta,
            min_budget=self.min_budget,
            seed=self.seed + self.s,
            maximize=self.maximize,
            objective=self.objective,
        )
        # Override initial configs count
        allocs = sh._allocate_budgets()
        if allocs and allocs[0][0] != self.n_initial:
            # Rebuild allocations to start from our n_initial
            new_allocs: List[Tuple[int, float]] = []
            n_i = self.n_initial
            r_i = self.min_budget
            for _ in range(self.s + 1):
                new_allocs.append((max(1, n_i), min(r_i, self.max_budget)))
                n_i = max(1, int(np.floor(n_i / self.eta)))
                r_i *= self.eta
            sh._allocate_budgets = lambda: new_allocs  # type: ignore[assignment]
        return sh.run(evaluate_fn)


# ---------------------------------------------------------------------------
# HyperparameterSweep  (main orchestrator)
# ---------------------------------------------------------------------------


class HyperparameterSweep:
    """Top-level orchestrator for hyperparameter sweeps."""

    def __init__(self, config: SweepConfig) -> None:
        self.config = config
        self.state = SweepState()
        self._grid = ParameterGrid(config.parameters)

    # -- Main entry point -----------------------------------------------------

    def run(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    ) -> SweepState:
        """Execute the sweep according to *config.strategy*."""
        self.state = SweepState(start_time=time.time())

        # Try loading checkpoint
        if self.config.checkpoint_dir:
            ckpt_path = os.path.join(self.config.checkpoint_dir, "sweep_state.pkl")
            if os.path.exists(ckpt_path):
                self.state = self.load_checkpoint(ckpt_path)

        dispatch = {
            SweepStrategy.GRID: self._run_grid,
            SweepStrategy.RANDOM: self._run_random,
            SweepStrategy.BAYESIAN: self._run_bayesian,
            SweepStrategy.SUCCESSIVE_HALVING: self._run_successive_halving,
            SweepStrategy.HYPERBAND: self._run_hyperband,
            SweepStrategy.LATIN_HYPERCUBE: self._run_latin_hypercube,
        }

        runner = dispatch.get(self.config.strategy)
        if runner is None:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

        results = runner(evaluate_fn)

        for r in results:
            if r.trial_id not in {t.trial_id for t in self.state.completed_trials}:
                self.state.completed_trials.append(r)
                self.state.update_best(r, self.config.maximize)
                self.state.iteration += 1
                self.state.total_budget_used += r.duration

        # Final checkpoint
        if self.config.checkpoint_dir:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            self.save_checkpoint(
                self.state,
                os.path.join(self.config.checkpoint_dir, "sweep_state.pkl"),
            )

        return self.state

    # -- Strategy runners -----------------------------------------------------

    def _run_grid(
        self, evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]]
    ) -> List[TrialResult]:
        results: List[TrialResult] = []
        done_ids = {t.trial_id for t in self.state.completed_trials}
        idx = self.state.iteration
        for params in self._grid.grid_search():
            tid = _trial_id(params, idx)
            if tid in done_ids:
                idx += 1
                continue
            result = _evaluate_trial(evaluate_fn, params, idx, self.config.objective)
            results.append(result)
            idx += 1
            if self._over_budget(results):
                break
            if self._check_early_stopping([r.objective_value for r in results]):
                break
            if self.config.checkpoint_dir and idx % 10 == 0:
                self._interim_checkpoint(results)
        return results

    def _run_random(
        self, evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]]
    ) -> List[TrialResult]:
        results: List[TrialResult] = []
        idx = self.state.iteration
        for params in self._grid.random_search(
            self.config.n_trials, seed=self.config.seed
        ):
            if idx < self.state.iteration:
                idx += 1
                continue
            result = _evaluate_trial(evaluate_fn, params, idx, self.config.objective)
            results.append(result)
            idx += 1
            if self._over_budget(results):
                break
            if self._check_early_stopping([r.objective_value for r in results]):
                break
            if self.config.checkpoint_dir and len(results) % 10 == 0:
                self._interim_checkpoint(results)
        return results

    def _run_latin_hypercube(
        self, evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]]
    ) -> List[TrialResult]:
        results: List[TrialResult] = []
        idx = self.state.iteration
        for params in self._grid.latin_hypercube(
            self.config.n_trials, seed=self.config.seed
        ):
            if idx < self.state.iteration:
                idx += 1
                continue
            result = _evaluate_trial(evaluate_fn, params, idx, self.config.objective)
            results.append(result)
            idx += 1
            if self._over_budget(results):
                break
            if self._check_early_stopping([r.objective_value for r in results]):
                break
        return results

    def _run_bayesian(
        self, evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]]
    ) -> List[TrialResult]:
        bo = BayesianOptimizer(
            specs=self.config.parameters,
            n_initial=max(5, self.config.n_trials // 5),
            acquisition_fn="ei",
            seed=self.config.seed,
        )

        # Replay past observations
        for t in self.state.completed_trials:
            sign = -1.0 if self.config.maximize else 1.0
            bo.observe(t.parameters, sign * t.objective_value)

        results: List[TrialResult] = []
        idx = self.state.iteration
        remaining = self.config.n_trials - len(self.state.completed_trials)

        for _ in range(remaining):
            suggestions = bo.suggest(1)
            params = suggestions[0]
            result = _evaluate_trial(evaluate_fn, params, idx, self.config.objective)
            results.append(result)

            sign = -1.0 if self.config.maximize else 1.0
            bo.observe(params, sign * result.objective_value)

            idx += 1
            if self._over_budget(results):
                break
            if self._check_early_stopping([r.objective_value for r in results]):
                break
            if self.config.checkpoint_dir and len(results) % 5 == 0:
                self._interim_checkpoint(results)

        return results

    def _run_successive_halving(
        self, evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]]
    ) -> List[TrialResult]:
        sh = SuccessiveHalving(
            specs=self.config.parameters,
            max_budget=self.config.budget if self.config.budget < float("inf") else 81.0,
            reduction_factor=3,
            min_budget=1.0,
            seed=self.config.seed,
            maximize=self.config.maximize,
            objective=self.config.objective,
        )
        return sh.run(evaluate_fn)

    def _run_hyperband(
        self, evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]]
    ) -> List[TrialResult]:
        eta = 3
        R = self.config.budget if self.config.budget < float("inf") else 81.0
        s_max = max(0, int(np.floor(np.log(R) / np.log(eta))))

        all_results: List[TrialResult] = []
        for s in range(s_max, -1, -1):
            bracket = _HyperbandBracket(
                specs=self.config.parameters,
                s=s,
                s_max=s_max,
                max_budget=R,
                eta=eta,
                seed=self.config.seed,
                maximize=self.config.maximize,
                objective=self.config.objective,
            )
            bracket_results = bracket.run(evaluate_fn)
            all_results.extend(bracket_results)
            if self._over_budget(all_results):
                break

        return all_results

    # -- Early stopping -------------------------------------------------------

    def _check_early_stopping(self, history: List[float]) -> bool:
        """Return True if the sweep should stop early."""
        if self.config.early_stopping_rounds <= 0:
            return False
        if len(history) < self.config.early_stopping_rounds:
            return False

        window = history[-self.config.early_stopping_rounds :]
        if self.config.maximize:
            best_recent = max(window)
            best_before = max(history[: -self.config.early_stopping_rounds]) if len(
                history
            ) > self.config.early_stopping_rounds else -float("inf")
            return best_recent <= best_before
        else:
            best_recent = min(window)
            best_before = min(history[: -self.config.early_stopping_rounds]) if len(
                history
            ) > self.config.early_stopping_rounds else float("inf")
            return best_recent >= best_before

    def _over_budget(self, results: List[TrialResult]) -> bool:
        total = self.state.total_budget_used + sum(r.duration for r in results)
        return total >= self.config.budget

    # -- Checkpointing --------------------------------------------------------

    def save_checkpoint(self, state: SweepState, path: str) -> None:
        """Persist sweep state to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def load_checkpoint(self, path: str) -> SweepState:
        """Restore sweep state from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        if not isinstance(state, SweepState):
            raise TypeError(f"Expected SweepState, got {type(state)}")
        return state

    def _interim_checkpoint(self, new_results: List[TrialResult]) -> None:
        """Save an intermediate checkpoint if a directory is configured."""
        if not self.config.checkpoint_dir:
            return
        merged = SweepState(
            completed_trials=list(self.state.completed_trials) + list(new_results),
            best_trial=self.state.best_trial,
            iteration=self.state.iteration + len(new_results),
            start_time=self.state.start_time,
            total_budget_used=self.state.total_budget_used
            + sum(r.duration for r in new_results),
        )
        for r in new_results:
            merged.update_best(r, self.config.maximize)
        self.save_checkpoint(
            merged,
            os.path.join(self.config.checkpoint_dir, "sweep_state.pkl"),
        )

    # -- Accessors ------------------------------------------------------------

    def best_params(self) -> Dict[str, Any]:
        """Return the parameters of the best trial so far."""
        if self.state.best_trial is None:
            return {}
        return dict(self.state.best_trial.parameters)

    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary of the sweep."""
        elapsed = time.time() - self.state.start_time
        objectives = [
            t.objective_value
            for t in self.state.completed_trials
            if t.status == "completed" and np.isfinite(t.objective_value)
        ]
        return {
            "strategy": self.config.strategy.value,
            "n_completed": len(self.state.completed_trials),
            "n_failed": sum(
                1 for t in self.state.completed_trials if t.status == "failed"
            ),
            "best_objective": self.state.best_trial.objective_value
            if self.state.best_trial
            else None,
            "best_params": self.best_params(),
            "objective_mean": float(np.mean(objectives)) if objectives else None,
            "objective_std": float(np.std(objectives)) if objectives else None,
            "objective_min": float(np.min(objectives)) if objectives else None,
            "objective_max": float(np.max(objectives)) if objectives else None,
            "elapsed_seconds": elapsed,
            "total_budget_used": self.state.total_budget_used,
        }

    # -- Parameter importance (fANOVA-style) ----------------------------------

    def parameter_importance(
        self, results: Optional[List[TrialResult]] = None
    ) -> Dict[str, float]:
        """Estimate marginal importance of each parameter via variance decomposition.

        Uses a simple functional-ANOVA-style approach:
        for each parameter, compute the fraction of total objective variance
        explained by that parameter's marginal effect.
        """
        if results is None:
            results = self.state.completed_trials

        valid = [r for r in results if r.status == "completed" and np.isfinite(r.objective_value)]
        if len(valid) < 3:
            return {s.name: 0.0 for s in self.config.parameters}

        objectives = np.array([r.objective_value for r in valid])
        total_var = float(np.var(objectives))
        if total_var < 1e-15:
            return {s.name: 1.0 / max(len(self.config.parameters), 1) for s in self.config.parameters}

        importance: Dict[str, float] = {}
        for spec in self.config.parameters:
            importance[spec.name] = self._marginal_variance(spec, valid, objectives, total_var)

        # Normalise to sum to 1
        total_imp = sum(importance.values())
        if total_imp > 1e-15:
            for k in importance:
                importance[k] /= total_imp

        return importance

    def _marginal_variance(
        self,
        spec: ParameterSpec,
        results: List[TrialResult],
        objectives: np.ndarray,
        total_var: float,
    ) -> float:
        """Compute variance explained by marginal effect of a single parameter."""
        values = np.array([spec.encode(r.parameters[spec.name]) for r in results])

        # Bin continuous values
        n_bins = min(10, len(set(values)))
        if n_bins < 2:
            return 0.0

        if spec.param_type == ParameterType.CATEGORICAL:
            bins_map: Dict[Any, List[float]] = {}
            for r in results:
                v = r.parameters[spec.name]
                bins_map.setdefault(v, []).append(r.objective_value)
            bin_means = [np.mean(vs) for vs in bins_map.values() if len(vs) > 0]
        else:
            bin_edges = np.linspace(values.min() - 1e-9, values.max() + 1e-9, n_bins + 1)
            bin_indices = np.digitize(values, bin_edges) - 1
            bin_means = []
            for b in range(n_bins):
                mask = bin_indices == b
                if np.any(mask):
                    bin_means.append(float(np.mean(objectives[mask])))

        if len(bin_means) < 2:
            return 0.0

        marginal_var = float(np.var(bin_means))
        return marginal_var / max(total_var, 1e-15)

    def interaction_effects(
        self, results: Optional[List[TrialResult]] = None
    ) -> Dict[Tuple[str, str], float]:
        """Estimate pairwise interaction effects between parameters.

        Returns a dictionary mapping (param_i, param_j) -> interaction strength.
        Uses residual variance analysis: for each pair, compute the joint
        marginal effect minus individual marginal effects.
        """
        if results is None:
            results = self.state.completed_trials

        valid = [r for r in results if r.status == "completed" and np.isfinite(r.objective_value)]
        if len(valid) < 5:
            return {}

        objectives = np.array([r.objective_value for r in valid])
        total_var = float(np.var(objectives))
        if total_var < 1e-15:
            return {}

        params = self.config.parameters
        marginals = self.parameter_importance(valid)
        interactions: Dict[Tuple[str, str], float] = {}

        for i in range(len(params)):
            for j in range(i + 1, len(params)):
                pi, pj = params[i], params[j]
                joint_var = self._joint_marginal_variance(pi, pj, valid, objectives, total_var)
                ind_sum = marginals.get(pi.name, 0.0) + marginals.get(pj.name, 0.0)
                interaction = max(0.0, joint_var - ind_sum)
                interactions[(pi.name, pj.name)] = interaction

        # Normalise
        total_int = sum(interactions.values())
        if total_int > 1e-15:
            for k in interactions:
                interactions[k] /= total_int

        return interactions

    def _joint_marginal_variance(
        self,
        spec_i: ParameterSpec,
        spec_j: ParameterSpec,
        results: List[TrialResult],
        objectives: np.ndarray,
        total_var: float,
    ) -> float:
        """Compute variance explained by joint marginal of two parameters."""
        vals_i = np.array([spec_i.encode(r.parameters[spec_i.name]) for r in results])
        vals_j = np.array([spec_j.encode(r.parameters[spec_j.name]) for r in results])

        n_bins = min(5, max(2, int(np.sqrt(len(results)))))
        edges_i = np.linspace(vals_i.min() - 1e-9, vals_i.max() + 1e-9, n_bins + 1)
        edges_j = np.linspace(vals_j.min() - 1e-9, vals_j.max() + 1e-9, n_bins + 1)

        bi = np.digitize(vals_i, edges_i) - 1
        bj = np.digitize(vals_j, edges_j) - 1

        cell_means: List[float] = []
        for ci in range(n_bins):
            for cj in range(n_bins):
                mask = (bi == ci) & (bj == cj)
                if np.any(mask):
                    cell_means.append(float(np.mean(objectives[mask])))

        if len(cell_means) < 2:
            return 0.0
        return float(np.var(cell_means)) / max(total_var, 1e-15)

    # -- Visualisation --------------------------------------------------------

    def plot_sweep_results(
        self, results: Optional[List[TrialResult]] = None, width: int = 800, height: int = 500
    ) -> str:
        """Generate an SVG visualisation of the sweep results.

        Returns:
            An SVG string showing:
            - Objective value vs trial index (scatter)
            - Running best curve
            - Parameter importance bar chart
        """
        if results is None:
            results = self.state.completed_trials

        valid = [r for r in results if r.status == "completed" and np.isfinite(r.objective_value)]
        if not valid:
            return "<svg></svg>"

        objs = [r.objective_value for r in valid]
        n = len(objs)

        # Running best
        running_best: List[float] = []
        best_fn = max if self.config.maximize else min
        for i, o in enumerate(objs):
            if i == 0:
                running_best.append(o)
            else:
                running_best.append(best_fn(running_best[-1], o))

        # Margins
        ml, mr, mt, mb = 80, 40, 40, 60
        pw = width - ml - mr
        ph = height - mt - mb

        y_min = min(objs) - abs(min(objs)) * 0.05 - 1e-6
        y_max = max(objs) + abs(max(objs)) * 0.05 + 1e-6
        if abs(y_max - y_min) < 1e-12:
            y_min -= 1
            y_max += 1

        def sx(i: int) -> float:
            return ml + (i / max(n - 1, 1)) * pw

        def sy(v: float) -> float:
            return mt + ph - ((v - y_min) / (y_max - y_min)) * ph

        svg_parts: List[str] = []
        svg_parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
            f' viewBox="0 0 {width} {height}">'
        )
        # Background
        svg_parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')
        # Axes
        svg_parts.append(
            f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="black" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="black" stroke-width="1"/>'
        )
        # Title
        svg_parts.append(
            f'<text x="{width // 2}" y="20" text-anchor="middle" font-size="14" font-family="sans-serif">'
            f"Sweep: {self.config.strategy.value} | Objective: {self.config.objective}</text>"
        )
        # X label
        svg_parts.append(
            f'<text x="{ml + pw // 2}" y="{height - 10}" text-anchor="middle" font-size="12"'
            f' font-family="sans-serif">Trial</text>'
        )
        # Y label
        svg_parts.append(
            f'<text x="15" y="{mt + ph // 2}" text-anchor="middle" font-size="12"'
            f' font-family="sans-serif" transform="rotate(-90 15 {mt + ph // 2})">'
            f"{self.config.objective}</text>"
        )

        # Y-axis ticks
        n_ticks = 5
        for t in range(n_ticks + 1):
            yv = y_min + t * (y_max - y_min) / n_ticks
            yp = sy(yv)
            svg_parts.append(
                f'<line x1="{ml - 5}" y1="{yp:.1f}" x2="{ml}" y2="{yp:.1f}" stroke="black"/>'
            )
            svg_parts.append(
                f'<text x="{ml - 8}" y="{yp + 4:.1f}" text-anchor="end" font-size="10"'
                f' font-family="sans-serif">{yv:.3g}</text>'
            )

        # X-axis ticks
        x_ticks = min(n, 10)
        for t in range(x_ticks):
            xi = int(t * (n - 1) / max(x_ticks - 1, 1))
            xp = sx(xi)
            svg_parts.append(
                f'<line x1="{xp:.1f}" y1="{mt + ph}" x2="{xp:.1f}" y2="{mt + ph + 5}" stroke="black"/>'
            )
            svg_parts.append(
                f'<text x="{xp:.1f}" y="{mt + ph + 18}" text-anchor="middle" font-size="10"'
                f' font-family="sans-serif">{xi}</text>'
            )

        # Grid lines
        for t in range(n_ticks + 1):
            yv = y_min + t * (y_max - y_min) / n_ticks
            yp = sy(yv)
            svg_parts.append(
                f'<line x1="{ml}" y1="{yp:.1f}" x2="{ml + pw}" y2="{yp:.1f}"'
                f' stroke="#eee" stroke-width="0.5"/>'
            )

        # Scatter points
        for i, o in enumerate(objs):
            xp = sx(i)
            yp = sy(o)
            svg_parts.append(
                f'<circle cx="{xp:.1f}" cy="{yp:.1f}" r="3" fill="#4285f4" opacity="0.7"/>'
            )

        # Running best line
        path_d = f"M {sx(0):.1f} {sy(running_best[0]):.1f}"
        for i in range(1, n):
            path_d += f" L {sx(i):.1f} {sy(running_best[i]):.1f}"
        svg_parts.append(
            f'<path d="{path_d}" stroke="#ea4335" stroke-width="2" fill="none"/>'
        )

        # Legend
        lx = ml + pw - 150
        ly = mt + 15
        svg_parts.append(f'<circle cx="{lx}" cy="{ly}" r="3" fill="#4285f4"/>')
        svg_parts.append(
            f'<text x="{lx + 8}" y="{ly + 4}" font-size="10" font-family="sans-serif">Trials</text>'
        )
        svg_parts.append(
            f'<line x1="{lx - 8}" y1="{ly + 18}" x2="{lx + 8}" y2="{ly + 18}"'
            f' stroke="#ea4335" stroke-width="2"/>'
        )
        svg_parts.append(
            f'<text x="{lx + 12}" y="{ly + 22}" font-size="10" font-family="sans-serif">Best so far</text>'
        )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    # -- Text table -----------------------------------------------------------

    def to_table(
        self,
        results: Optional[List[TrialResult]] = None,
        max_rows: int = 50,
        sort_by_objective: bool = True,
    ) -> str:
        """Format results as an ASCII table."""
        if results is None:
            results = self.state.completed_trials

        if not results:
            return "(no results)"

        param_names = [s.name for s in self.config.parameters]
        col_widths: Dict[str, int] = {}
        col_widths["#"] = 5
        col_widths["trial_id"] = 14
        for pn in param_names:
            col_widths[pn] = max(len(pn), 12)
        col_widths["objective"] = 14
        col_widths["duration"] = 10
        col_widths["status"] = 10

        header_cols = ["#", "trial_id"] + param_names + ["objective", "duration", "status"]
        header = " | ".join(c.ljust(col_widths.get(c, 12)) for c in header_cols)
        sep = "-+-".join("-" * col_widths.get(c, 12) for c in header_cols)

        rows_data = list(results)
        if sort_by_objective:
            rows_data.sort(
                key=lambda r: r.objective_value if np.isfinite(r.objective_value) else float("inf"),
                reverse=self.config.maximize,
            )

        lines = [header, sep]
        for rank, r in enumerate(rows_data[:max_rows], 1):
            cells: List[str] = []
            cells.append(str(rank).ljust(col_widths["#"]))
            cells.append(r.trial_id[:12].ljust(col_widths["trial_id"]))
            for pn in param_names:
                cells.append(_format_value(r.parameters.get(pn, "?"), col_widths[pn]))
            cells.append(_format_value(r.objective_value, col_widths["objective"]))
            cells.append(f"{r.duration:.2f}s".ljust(col_widths["duration"]))
            cells.append(r.status[:10].ljust(col_widths["status"]))
            lines.append(" | ".join(cells))

        if len(rows_data) > max_rows:
            lines.append(f"... ({len(rows_data) - max_rows} more rows)")

        return "\n".join(lines)

    # -- Export ---------------------------------------------------------------

    def export_results(
        self,
        results: Optional[List[TrialResult]] = None,
        fmt: str = "json",
    ) -> str:
        """Export results as JSON, CSV, or markdown."""
        if results is None:
            results = self.state.completed_trials

        if fmt == "json":
            return self._export_json(results)
        elif fmt == "csv":
            return self._export_csv(results)
        elif fmt == "markdown" or fmt == "md":
            return self._export_markdown(results)
        else:
            raise ValueError(f"Unknown format: {fmt}. Use 'json', 'csv', or 'markdown'.")

    def _export_json(self, results: List[TrialResult]) -> str:
        data = {
            "config": {
                "strategy": self.config.strategy.value,
                "n_trials": self.config.n_trials,
                "objective": self.config.objective,
                "maximize": self.config.maximize,
                "seed": self.config.seed,
                "parameters": [
                    {
                        "name": s.name,
                        "type": s.param_type.value,
                        "low": s.low,
                        "high": s.high,
                        "choices": s.choices,
                    }
                    for s in self.config.parameters
                ],
            },
            "summary": self.summary(),
            "results": [r.to_dict() for r in results],
        }
        return json.dumps(data, indent=2, default=str)

    def _export_csv(self, results: List[TrialResult]) -> str:
        param_names = [s.name for s in self.config.parameters]
        metric_names: set = set()
        for r in results:
            metric_names.update(r.all_metrics.keys())
        metric_names_sorted = sorted(metric_names)

        header = (
            ["trial_id"]
            + param_names
            + ["objective_value", "duration", "status"]
            + metric_names_sorted
        )
        lines = [",".join(header)]

        for r in results:
            row: List[str] = [r.trial_id]
            for pn in param_names:
                row.append(str(r.parameters.get(pn, "")))
            row.append(str(r.objective_value))
            row.append(f"{r.duration:.4f}")
            row.append(r.status)
            for mn in metric_names_sorted:
                row.append(str(r.all_metrics.get(mn, "")))
            lines.append(",".join(row))

        return "\n".join(lines)

    def _export_markdown(self, results: List[TrialResult]) -> str:
        param_names = [s.name for s in self.config.parameters]
        header = ["trial_id"] + param_names + ["objective", "status"]
        md_lines: List[str] = []
        md_lines.append("# Sweep Results")
        md_lines.append("")
        md_lines.append(f"**Strategy**: {self.config.strategy.value}")
        md_lines.append(f"**Trials**: {len(results)}")
        if self.state.best_trial:
            md_lines.append(
                f"**Best objective**: {self.state.best_trial.objective_value:.6g}"
            )
        md_lines.append("")

        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("| " + " | ".join("---" for _ in header) + " |")

        sorted_results = sorted(
            results,
            key=lambda r: r.objective_value if np.isfinite(r.objective_value) else float("inf"),
            reverse=self.config.maximize,
        )
        for r in sorted_results[:50]:
            cells = [r.trial_id[:12]]
            for pn in param_names:
                v = r.parameters.get(pn, "?")
                if isinstance(v, float):
                    cells.append(f"{v:.4g}")
                else:
                    cells.append(str(v))
            cells.append(f"{r.objective_value:.6g}")
            cells.append(r.status)
            md_lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(md_lines)


# ---------------------------------------------------------------------------
# Sensitivity analysis helpers
# ---------------------------------------------------------------------------


class SensitivityAnalyzer:
    """Morris-method & Sobol-index sensitivity analysis for sweep results."""

    def __init__(self, specs: List[ParameterSpec], seed: int = 42) -> None:
        self.specs = specs
        self.d = len(specs)
        self.rng = np.random.RandomState(seed)

    def morris_screening(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], float],
        n_trajectories: int = 10,
        delta: float = 0.1,
    ) -> Dict[str, Dict[str, float]]:
        """Morris one-at-a-time elementary effects screening.

        Returns a dict mapping parameter name -> {mu, mu_star, sigma}.
        """
        effects: Dict[str, List[float]] = {s.name: [] for s in self.specs}

        for _ in range(n_trajectories):
            # Random base point in [0, 1]^d
            x0 = self.rng.rand(self.d)
            base_params = self._decode(x0)
            base_val = evaluate_fn(base_params)

            # Permute dimensions
            perm = self.rng.permutation(self.d)
            x_current = x0.copy()
            current_val = base_val

            for j in perm:
                x_new = x_current.copy()
                # Perturb dimension j
                if x_new[j] + delta <= 1.0:
                    x_new[j] += delta
                else:
                    x_new[j] -= delta

                new_params = self._decode(x_new)
                new_val = evaluate_fn(new_params)

                ee = (new_val - current_val) / delta
                effects[self.specs[j].name].append(ee)

                x_current = x_new
                current_val = new_val

        result: Dict[str, Dict[str, float]] = {}
        for name, ee_list in effects.items():
            arr = np.array(ee_list) if ee_list else np.array([0.0])
            result[name] = {
                "mu": float(np.mean(arr)),
                "mu_star": float(np.mean(np.abs(arr))),
                "sigma": float(np.std(arr)),
            }
        return result

    def sobol_indices(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], float],
        n_samples: int = 256,
    ) -> Dict[str, Dict[str, float]]:
        """Estimate first-order and total Sobol sensitivity indices.

        Uses the Saltelli sampling scheme with 2*n_samples*(d+1) evaluations.
        """
        d = self.d
        A = self.rng.rand(n_samples, d)
        B = self.rng.rand(n_samples, d)

        f_A = np.array([evaluate_fn(self._decode(A[i])) for i in range(n_samples)])
        f_B = np.array([evaluate_fn(self._decode(B[i])) for i in range(n_samples)])

        total_var = np.var(np.concatenate([f_A, f_B]))
        if total_var < 1e-15:
            return {s.name: {"S1": 0.0, "ST": 0.0} for s in self.specs}

        result: Dict[str, Dict[str, float]] = {}
        for j in range(d):
            # AB_j: A with column j replaced by B's column j
            AB_j = A.copy()
            AB_j[:, j] = B[:, j]
            f_AB_j = np.array(
                [evaluate_fn(self._decode(AB_j[i])) for i in range(n_samples)]
            )

            # First-order: S1_j
            v_j = np.mean(f_B * (f_AB_j - f_A))
            S1 = v_j / max(total_var, 1e-15)

            # Total-order: ST_j
            vt_j = 0.5 * np.mean((f_A - f_AB_j) ** 2)
            ST = vt_j / max(total_var, 1e-15)

            result[self.specs[j].name] = {
                "S1": float(np.clip(S1, 0.0, 1.0)),
                "ST": float(np.clip(ST, 0.0, 1.0)),
            }

        return result

    def _decode(self, x: np.ndarray) -> Dict[str, Any]:
        return {spec.name: spec.decode(float(x[j])) for j, spec in enumerate(self.specs)}


# ---------------------------------------------------------------------------
# Parallel trial evaluation helper
# ---------------------------------------------------------------------------


class _ParallelEvaluator:
    """Evaluate trials using a thread-pool (stdlib concurrent.futures)."""

    def __init__(self, n_parallel: int = 1) -> None:
        self.n_parallel = max(n_parallel, 1)

    def evaluate_batch(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        param_list: List[Dict[str, Any]],
        start_idx: int,
        objective: str,
    ) -> List[TrialResult]:
        if self.n_parallel <= 1:
            results = []
            for i, params in enumerate(param_list):
                results.append(
                    _evaluate_trial(evaluate_fn, params, start_idx + i, objective)
                )
            return results

        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: List[TrialResult] = [None] * len(param_list)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=self.n_parallel) as pool:
            futures = {}
            for i, params in enumerate(param_list):
                fut = pool.submit(
                    _evaluate_trial, evaluate_fn, params, start_idx + i, objective
                )
                futures[fut] = i
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        return results


# ---------------------------------------------------------------------------
# Configuration builder helpers
# ---------------------------------------------------------------------------


def make_sweep_config(
    strategy: str = "random",
    n_trials: int = 50,
    objective: str = "loss",
    maximize: bool = False,
    seed: int = 42,
    **kwargs: Any,
) -> SweepConfig:
    """Convenience factory for SweepConfig."""
    strat = SweepStrategy(strategy)
    return SweepConfig(
        strategy=strat,
        n_trials=n_trials,
        objective=objective,
        maximize=maximize,
        seed=seed,
        **kwargs,
    )


def make_parameter(
    name: str,
    type: str = "continuous",
    low: Optional[float] = None,
    high: Optional[float] = None,
    choices: Optional[List[Any]] = None,
    default: Optional[Any] = None,
    log_scale: bool = False,
    step: Optional[float] = None,
) -> ParameterSpec:
    """Convenience factory for ParameterSpec."""
    pt = ParameterType(type)
    return ParameterSpec(
        name=name,
        param_type=pt,
        low=low,
        high=high,
        choices=choices,
        default=default,
        log_scale=log_scale,
        step=step,
    )


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------


class ConvergenceDiagnostics:
    """Tools for diagnosing whether a sweep has converged."""

    @staticmethod
    def running_best(objectives: List[float], maximize: bool = False) -> List[float]:
        """Compute the running best objective."""
        out: List[float] = []
        fn = max if maximize else min
        for i, v in enumerate(objectives):
            if i == 0:
                out.append(v)
            else:
                out.append(fn(out[-1], v))
        return out

    @staticmethod
    def improvement_rate(
        objectives: List[float], window: int = 10, maximize: bool = False
    ) -> List[float]:
        """Rate of improvement in a sliding window."""
        rb = ConvergenceDiagnostics.running_best(objectives, maximize)
        rates: List[float] = []
        for i in range(len(rb)):
            if i < window:
                rates.append(float("nan"))
            else:
                diff = rb[i] - rb[i - window]
                if not maximize:
                    diff = -diff
                rates.append(diff / window)
        return rates

    @staticmethod
    def effective_sample_size(objectives: List[float]) -> float:
        """Estimate effective sample size via autocorrelation."""
        arr = np.array(objectives)
        n = len(arr)
        if n < 4:
            return float(n)

        arr = arr - np.mean(arr)
        var = np.var(arr)
        if var < 1e-15:
            return float(n)

        max_lag = min(n // 2, 50)
        rho_sum = 0.0
        for lag in range(1, max_lag):
            rho = np.mean(arr[: n - lag] * arr[lag:]) / var
            if rho < 0.05:
                break
            rho_sum += rho

        ess = n / (1 + 2 * rho_sum)
        return max(1.0, float(ess))

    @staticmethod
    def convergence_test(
        objectives: List[float],
        threshold: float = 0.01,
        window: int = 20,
        maximize: bool = False,
    ) -> Dict[str, Any]:
        """Test whether the sweep has converged.

        Returns a dict with 'converged' (bool), 'relative_improvement',
        and 'n_since_best'.
        """
        if len(objectives) < window:
            return {
                "converged": False,
                "relative_improvement": float("nan"),
                "n_since_best": 0,
            }

        rb = ConvergenceDiagnostics.running_best(objectives, maximize)
        best = rb[-1]
        best_idx = 0
        for i, v in enumerate(rb):
            if v == best:
                best_idx = i

        n_since_best = len(rb) - 1 - best_idx

        old_best = rb[max(0, len(rb) - window - 1)]
        if abs(old_best) < 1e-15:
            rel_imp = abs(best - old_best)
        else:
            rel_imp = abs(best - old_best) / abs(old_best)

        converged = rel_imp < threshold and n_since_best >= window

        return {
            "converged": converged,
            "relative_improvement": float(rel_imp),
            "n_since_best": n_since_best,
        }


# ---------------------------------------------------------------------------
# Warm-starting helper
# ---------------------------------------------------------------------------


class WarmStarter:
    """Transfer knowledge from previous sweeps to warm-start a new one."""

    def __init__(self, specs: List[ParameterSpec]) -> None:
        self.specs = specs
        self._prior_results: List[TrialResult] = []

    def add_results(self, results: List[TrialResult]) -> None:
        """Add results from a previous sweep."""
        self._prior_results.extend(results)

    def suggest_initial(
        self, n: int, maximize: bool = False
    ) -> List[Dict[str, Any]]:
        """Suggest initial configurations based on prior results."""
        if not self._prior_results:
            grid = ParameterGrid(self.specs)
            return list(grid.random_search(n, seed=42))

        valid = [
            r for r in self._prior_results
            if r.status == "completed" and np.isfinite(r.objective_value)
        ]
        if not valid:
            grid = ParameterGrid(self.specs)
            return list(grid.random_search(n, seed=42))

        valid.sort(key=lambda r: r.objective_value, reverse=maximize)
        top = valid[: min(n, len(valid))]

        suggestions: List[Dict[str, Any]] = []
        rng = np.random.RandomState(42)
        for r in top:
            # Add the exact point and a small perturbation
            suggestions.append(dict(r.parameters))
            if len(suggestions) >= n:
                break
            perturbed = self._perturb(r.parameters, rng, scale=0.1)
            suggestions.append(perturbed)
            if len(suggestions) >= n:
                break

        return suggestions[:n]

    def _perturb(
        self, params: Dict[str, Any], rng: np.random.RandomState, scale: float
    ) -> Dict[str, Any]:
        """Slightly perturb a parameter configuration."""
        result: Dict[str, Any] = {}
        for spec in self.specs:
            val = params.get(spec.name, spec.default)
            enc = spec.encode(val)
            enc += rng.normal(0, scale)
            enc = float(np.clip(enc, 0.0, 1.0))
            result[spec.name] = spec.decode(enc)
        return result


# ---------------------------------------------------------------------------
# Result aggregation and comparison
# ---------------------------------------------------------------------------


class SweepComparator:
    """Compare results from multiple sweep runs."""

    def __init__(self) -> None:
        self._sweeps: Dict[str, List[TrialResult]] = {}

    def add_sweep(self, name: str, results: List[TrialResult]) -> None:
        self._sweeps[name] = list(results)

    def compare(self, maximize: bool = False) -> Dict[str, Dict[str, Any]]:
        """Compare all added sweeps and return summary statistics."""
        comparison: Dict[str, Dict[str, Any]] = {}
        for name, results in self._sweeps.items():
            valid = [
                r for r in results
                if r.status == "completed" and np.isfinite(r.objective_value)
            ]
            if not valid:
                comparison[name] = {
                    "n_trials": len(results),
                    "n_valid": 0,
                    "best": None,
                    "mean": None,
                    "std": None,
                    "median": None,
                }
                continue

            objs = np.array([r.objective_value for r in valid])
            best_fn = np.max if maximize else np.min
            comparison[name] = {
                "n_trials": len(results),
                "n_valid": len(valid),
                "best": float(best_fn(objs)),
                "mean": float(np.mean(objs)),
                "std": float(np.std(objs)),
                "median": float(np.median(objs)),
                "total_duration": sum(r.duration for r in valid),
            }

        return comparison

    def rank_sweeps(self, maximize: bool = False) -> List[Tuple[str, float]]:
        """Rank sweeps by their best objective value."""
        comp = self.compare(maximize)
        ranked = []
        for name, stats in comp.items():
            if stats["best"] is not None:
                ranked.append((name, stats["best"]))
        ranked.sort(key=lambda x: x[1], reverse=maximize)
        return ranked

    def comparison_table(self, maximize: bool = False) -> str:
        """Format comparison as a text table."""
        comp = self.compare(maximize)
        cols = ["Sweep", "Trials", "Valid", "Best", "Mean", "Std", "Median", "Duration"]
        widths = [20, 8, 8, 14, 14, 14, 14, 12]

        header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
        sep = "-+-".join("-" * w for w in widths)

        lines = [header, sep]
        for name, stats in comp.items():
            row = [
                name[:20].ljust(20),
                str(stats["n_trials"]).ljust(8),
                str(stats["n_valid"]).ljust(8),
                _format_value(stats["best"], 14) if stats["best"] is not None else "N/A".ljust(14),
                _format_value(stats["mean"], 14) if stats["mean"] is not None else "N/A".ljust(14),
                _format_value(stats["std"], 14) if stats["std"] is not None else "N/A".ljust(14),
                _format_value(stats["median"], 14) if stats["median"] is not None else "N/A".ljust(14),
                f"{stats.get('total_duration', 0):.2f}s".ljust(12),
            ]
            lines.append(" | ".join(row))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parameter space utilities
# ---------------------------------------------------------------------------


class ParameterSpaceUtils:
    """Utilities for working with parameter spaces."""

    @staticmethod
    def distance(
        p1: Dict[str, Any],
        p2: Dict[str, Any],
        specs: List[ParameterSpec],
    ) -> float:
        """Euclidean distance in encoded (unit-cube) space."""
        d = 0.0
        for spec in specs:
            e1 = spec.encode(p1.get(spec.name, spec.default))
            e2 = spec.encode(p2.get(spec.name, spec.default))
            d += (e1 - e2) ** 2
        return math.sqrt(d)

    @staticmethod
    def diversity_score(
        points: List[Dict[str, Any]], specs: List[ParameterSpec]
    ) -> float:
        """Average pairwise distance (higher = more diverse)."""
        n = len(points)
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += ParameterSpaceUtils.distance(points[i], points[j], specs)
                count += 1
        return total / count

    @staticmethod
    def coverage(
        points: List[Dict[str, Any]],
        specs: List[ParameterSpec],
        n_bins: int = 5,
    ) -> float:
        """Fraction of bins in the encoded space that are covered."""
        d = len(specs)
        if d == 0 or not points:
            return 0.0

        total_bins = n_bins**d
        occupied: set = set()

        for pt in points:
            bin_key = []
            for spec in specs:
                enc = spec.encode(pt.get(spec.name, spec.default))
                b = min(int(enc * n_bins), n_bins - 1)
                bin_key.append(b)
            occupied.add(tuple(bin_key))

        return len(occupied) / total_bins

    @staticmethod
    def nearest_neighbour_distances(
        points: List[Dict[str, Any]], specs: List[ParameterSpec]
    ) -> List[float]:
        """Compute nearest-neighbour distance for each point."""
        n = len(points)
        if n < 2:
            return [0.0] * n

        # Encode all points
        encoded = np.array(
            [[spec.encode(pt.get(spec.name, spec.default)) for spec in specs] for pt in points]
        )

        nn_dists: List[float] = []
        for i in range(n):
            dists = np.sqrt(np.sum((encoded - encoded[i]) ** 2, axis=1))
            dists[i] = float("inf")
            nn_dists.append(float(np.min(dists)))
        return nn_dists


# ---------------------------------------------------------------------------
# Constraint handling
# ---------------------------------------------------------------------------


class ConstraintHandler:
    """Handle parameter constraints and forbidden regions."""

    def __init__(self) -> None:
        self._constraints: List[Callable[[Dict[str, Any]], bool]] = []
        self._forbidden: List[Dict[str, Any]] = []

    def add_constraint(self, fn: Callable[[Dict[str, Any]], bool]) -> None:
        """Add a constraint function. Must return True if the point is valid."""
        self._constraints.append(fn)

    def add_forbidden_region(self, partial_params: Dict[str, Any]) -> None:
        """Add a forbidden parameter combination."""
        self._forbidden.append(partial_params)

    def is_valid(self, params: Dict[str, Any]) -> bool:
        """Check if a parameter configuration satisfies all constraints."""
        for fn in self._constraints:
            try:
                if not fn(params):
                    return False
            except Exception:
                return False

        for forbidden in self._forbidden:
            match = True
            for k, v in forbidden.items():
                if k in params and params[k] != v:
                    match = False
                    break
            if match and forbidden:
                return False

        return True

    def filter_points(
        self, points: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter out invalid points."""
        return [p for p in points if self.is_valid(p)]

    def rejection_sample(
        self,
        specs: List[ParameterSpec],
        n: int,
        seed: int = 42,
        max_attempts: int = 10000,
    ) -> List[Dict[str, Any]]:
        """Sample *n* valid points via rejection sampling."""
        rng = np.random.RandomState(seed)
        grid = ParameterGrid(specs)
        valid: List[Dict[str, Any]] = []
        attempts = 0
        while len(valid) < n and attempts < max_attempts:
            pt = grid._sample_random(rng)
            if self.is_valid(pt):
                valid.append(pt)
            attempts += 1
        return valid


# ---------------------------------------------------------------------------
# Search space transformations
# ---------------------------------------------------------------------------


class SearchSpaceTransformer:
    """Transform between original and warped search spaces."""

    def __init__(self, specs: List[ParameterSpec]) -> None:
        self.specs = specs
        self.d = len(specs)
        self._warping_fns: Dict[str, Callable[[float], float]] = {}
        self._inv_warping_fns: Dict[str, Callable[[float], float]] = {}

    def add_warping(
        self,
        param_name: str,
        forward: Callable[[float], float],
        inverse: Callable[[float], float],
    ) -> None:
        """Register a custom warping for a parameter."""
        self._warping_fns[param_name] = forward
        self._inv_warping_fns[param_name] = inverse

    def to_warped(self, params: Dict[str, Any]) -> np.ndarray:
        """Transform params to warped unit-cube space."""
        x = np.zeros(self.d)
        for j, spec in enumerate(self.specs):
            enc = spec.encode(params[spec.name])
            if spec.name in self._warping_fns:
                enc = self._warping_fns[spec.name](enc)
            x[j] = np.clip(enc, 0.0, 1.0)
        return x

    def from_warped(self, x: np.ndarray) -> Dict[str, Any]:
        """Transform warped unit-cube point back to original space."""
        params: Dict[str, Any] = {}
        for j, spec in enumerate(self.specs):
            val = float(x[j])
            if spec.name in self._inv_warping_fns:
                val = self._inv_warping_fns[spec.name](val)
            val = float(np.clip(val, 0.0, 1.0))
            params[spec.name] = spec.decode(val)
        return params

    def compute_input_warping(
        self,
        results: List[TrialResult],
        specs: List[ParameterSpec],
    ) -> Dict[str, Tuple[float, float]]:
        """Compute CDF-based input warping parameters from observed data.

        Returns beta distribution parameters (alpha, beta) for each parameter
        that, when applied, make the marginal distribution more uniform.
        """
        warping_params: Dict[str, Tuple[float, float]] = {}
        for spec in specs:
            vals = []
            for r in results:
                if r.status == "completed" and spec.name in r.parameters:
                    vals.append(spec.encode(r.parameters[spec.name]))
            if len(vals) < 5:
                warping_params[spec.name] = (1.0, 1.0)
                continue

            arr = np.array(vals)
            mean_val = np.mean(arr)
            var_val = np.var(arr)

            if var_val < 1e-10:
                warping_params[spec.name] = (1.0, 1.0)
                continue

            # Method of moments for beta distribution
            common = mean_val * (1 - mean_val) / max(var_val, 1e-10) - 1
            common = max(common, 0.1)
            alpha = max(0.1, mean_val * common)
            beta = max(0.1, (1 - mean_val) * common)
            warping_params[spec.name] = (alpha, beta)

        return warping_params


# ---------------------------------------------------------------------------
# Multi-objective support
# ---------------------------------------------------------------------------


class MultiObjectiveResult:
    """Container for multi-objective optimisation results."""

    def __init__(self, objective_names: List[str]) -> None:
        self.objective_names = objective_names
        self._points: List[Tuple[Dict[str, Any], np.ndarray]] = []

    def add(self, params: Dict[str, Any], objectives: Dict[str, float]) -> None:
        vec = np.array([objectives.get(n, float("inf")) for n in self.objective_names])
        self._points.append((params, vec))

    def pareto_front(self, maximize: Optional[List[bool]] = None) -> List[Tuple[Dict[str, Any], np.ndarray]]:
        """Compute the Pareto-optimal set."""
        if not self._points:
            return []

        if maximize is None:
            maximize = [False] * len(self.objective_names)

        # Negate objectives to maximize so we always minimise
        adjusted = []
        for params, vec in self._points:
            v = vec.copy()
            for i, m in enumerate(maximize):
                if m:
                    v[i] = -v[i]
            adjusted.append((params, v))

        # Non-dominated sort (first front only)
        n = len(adjusted)
        is_dominated = [False] * n
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                # Check if j dominates i
                _, vi = adjusted[i]
                _, vj = adjusted[j]
                if np.all(vj <= vi) and np.any(vj < vi):
                    is_dominated[i] = True
                    break

        front = []
        for i in range(n):
            if not is_dominated[i]:
                front.append(self._points[i])
        return front

    def hypervolume(
        self, reference: np.ndarray, maximize: Optional[List[bool]] = None
    ) -> float:
        """Compute 2D hypervolume indicator (only for 2 objectives)."""
        front = self.pareto_front(maximize)
        if not front or len(self.objective_names) != 2:
            return 0.0

        if maximize is None:
            maximize = [False, False]

        points_2d = []
        for _, vec in front:
            v = vec.copy()
            for i, m in enumerate(maximize):
                if m:
                    v[i] = -v[i]
            points_2d.append(v)

        # Sort by first objective
        points_2d.sort(key=lambda p: p[0])

        hv = 0.0
        prev_y = reference[1] if not maximize[1] else -reference[1]
        ref_x = reference[0] if not maximize[0] else -reference[0]

        for p in points_2d:
            if p[0] < ref_x and p[1] < prev_y:
                hv += (ref_x - p[0]) * (prev_y - p[1])
                prev_y = p[1]

        return float(hv)


# ---------------------------------------------------------------------------
# Adaptive budget allocation
# ---------------------------------------------------------------------------


class AdaptiveBudgetAllocator:
    """Dynamically allocate evaluation budget based on observed performance."""

    def __init__(
        self,
        total_budget: float,
        n_stages: int = 4,
        allocation_strategy: str = "exponential",
    ) -> None:
        self.total_budget = total_budget
        self.n_stages = n_stages
        self.allocation_strategy = allocation_strategy
        self._used: float = 0.0
        self._stage: int = 0

    def budget_for_stage(self, stage: Optional[int] = None) -> float:
        """Return the budget allocated to the given stage."""
        if stage is None:
            stage = self._stage

        if self.allocation_strategy == "uniform":
            return self.total_budget / self.n_stages

        if self.allocation_strategy == "exponential":
            # Later stages get more budget
            weights = [2.0**i for i in range(self.n_stages)]
            total_w = sum(weights)
            return self.total_budget * weights[min(stage, self.n_stages - 1)] / total_w

        if self.allocation_strategy == "linear":
            weights = [i + 1 for i in range(self.n_stages)]
            total_w = sum(weights)
            return self.total_budget * weights[min(stage, self.n_stages - 1)] / total_w

        return self.total_budget / self.n_stages

    def advance_stage(self, used: float) -> None:
        """Record budget usage and advance to the next stage."""
        self._used += used
        self._stage += 1

    def remaining_budget(self) -> float:
        return max(0.0, self.total_budget - self._used)

    def should_continue(self) -> bool:
        return self._used < self.total_budget and self._stage < self.n_stages


# ---------------------------------------------------------------------------
# Result caching
# ---------------------------------------------------------------------------


class ResultCache:
    """Cache trial results to avoid duplicate evaluations."""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self._memory_cache: Dict[str, TrialResult] = {}
        self._cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _key(self, params: Dict[str, Any]) -> str:
        raw = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, params: Dict[str, Any]) -> Optional[TrialResult]:
        key = self._key(params)
        if key in self._memory_cache:
            return self._memory_cache[key]

        if self._cache_dir:
            path = os.path.join(self._cache_dir, f"{key}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    result = pickle.load(f)
                self._memory_cache[key] = result
                return result

        return None

    def put(self, params: Dict[str, Any], result: TrialResult) -> None:
        key = self._key(params)
        self._memory_cache[key] = result

        if self._cache_dir:
            path = os.path.join(self._cache_dir, f"{key}.pkl")
            with open(path, "wb") as f:
                pickle.dump(result, f)

    def cached_evaluate(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        params: Dict[str, Any],
        trial_idx: int,
        objective: str,
    ) -> TrialResult:
        """Evaluate with caching."""
        cached = self.get(params)
        if cached is not None:
            return cached
        result = _evaluate_trial(evaluate_fn, params, trial_idx, objective)
        self.put(params, result)
        return result

    def stats(self) -> Dict[str, int]:
        n_disk = 0
        if self._cache_dir and os.path.isdir(self._cache_dir):
            n_disk = len([f for f in os.listdir(self._cache_dir) if f.endswith(".pkl")])
        return {
            "memory_entries": len(self._memory_cache),
            "disk_entries": n_disk,
        }


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


class ConfigValidator:
    """Validate sweep configurations before running."""

    @staticmethod
    def validate(config: SweepConfig) -> List[str]:
        """Return a list of validation error messages (empty if valid)."""
        errors: List[str] = []

        if not config.parameters:
            errors.append("No parameters defined.")

        names = set()
        for spec in config.parameters:
            if spec.name in names:
                errors.append(f"Duplicate parameter name: '{spec.name}'.")
            names.add(spec.name)

            if spec.param_type != ParameterType.CATEGORICAL:
                if spec.low is not None and spec.high is not None:
                    if spec.low > spec.high:
                        errors.append(
                            f"Parameter '{spec.name}': low ({spec.low}) > high ({spec.high})."
                        )
                if spec.param_type == ParameterType.LOG_UNIFORM:
                    if spec.low is not None and spec.low <= 0:
                        errors.append(
                            f"Parameter '{spec.name}': LOG_UNIFORM requires positive bounds."
                        )

        if config.n_trials < 1:
            errors.append(f"n_trials must be >= 1, got {config.n_trials}.")

        if config.strategy == SweepStrategy.GRID:
            total = ParameterGrid(config.parameters).total_combinations()
            if total > 1_000_000:
                errors.append(
                    f"Grid search would produce {total} combinations. Consider random or Bayesian."
                )

        if config.strategy == SweepStrategy.BAYESIAN:
            if len(config.parameters) > 20:
                errors.append(
                    "Bayesian optimisation with >20 dimensions is likely ineffective."
                )

        if config.budget <= 0:
            errors.append(f"Budget must be positive, got {config.budget}.")

        return errors

    @staticmethod
    def validate_or_raise(config: SweepConfig) -> None:
        errors = ConfigValidator.validate(config)
        if errors:
            raise ValueError("Invalid sweep config:\n" + "\n".join(f"  - {e}" for e in errors))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class SweepReporter:
    """Generate human-readable reports from sweep results."""

    def __init__(self, sweep: HyperparameterSweep) -> None:
        self.sweep = sweep

    def full_report(self) -> str:
        """Generate a comprehensive text report."""
        sections: List[str] = []
        sections.append("=" * 72)
        sections.append("HYPERPARAMETER SWEEP REPORT")
        sections.append("=" * 72)
        sections.append("")

        # Summary
        summary = self.sweep.summary()
        sections.append("SUMMARY")
        sections.append("-" * 40)
        for k, v in summary.items():
            if k == "best_params":
                sections.append(f"  {k}:")
                for pk, pv in (v or {}).items():
                    sections.append(f"    {pk}: {pv}")
            else:
                sections.append(f"  {k}: {v}")
        sections.append("")

        # Parameter importance
        results = self.sweep.state.completed_trials
        if len(results) >= 3:
            sections.append("PARAMETER IMPORTANCE")
            sections.append("-" * 40)
            importance = self.sweep.parameter_importance(results)
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for name, imp in sorted_imp:
                bar_len = int(imp * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                sections.append(f"  {name:20s} {bar} {imp:.3f}")
            sections.append("")

        # Interaction effects
        if len(results) >= 5:
            interactions = self.sweep.interaction_effects(results)
            if interactions:
                sections.append("INTERACTION EFFECTS")
                sections.append("-" * 40)
                sorted_int = sorted(interactions.items(), key=lambda x: x[1], reverse=True)
                for (p1, p2), strength in sorted_int[:10]:
                    sections.append(f"  {p1} x {p2}: {strength:.4f}")
                sections.append("")

        # Convergence
        objs = [
            r.objective_value
            for r in results
            if r.status == "completed" and np.isfinite(r.objective_value)
        ]
        if objs:
            conv = ConvergenceDiagnostics.convergence_test(
                objs, maximize=self.sweep.config.maximize
            )
            sections.append("CONVERGENCE")
            sections.append("-" * 40)
            sections.append(f"  Converged: {conv['converged']}")
            sections.append(f"  Relative improvement: {conv['relative_improvement']:.6f}")
            sections.append(f"  Trials since best: {conv['n_since_best']}")
            ess = ConvergenceDiagnostics.effective_sample_size(objs)
            sections.append(f"  Effective sample size: {ess:.1f}")
            sections.append("")

        # Results table
        sections.append("TOP RESULTS")
        sections.append("-" * 40)
        sections.append(self.sweep.to_table(results, max_rows=20))
        sections.append("")

        sections.append("=" * 72)
        return "\n".join(sections)

    def latex_table(self, n_rows: int = 10) -> str:
        """Generate a LaTeX table of the top results."""
        results = sorted(
            [r for r in self.sweep.state.completed_trials if r.status == "completed"],
            key=lambda r: r.objective_value,
            reverse=self.sweep.config.maximize,
        )[:n_rows]

        param_names = [s.name for s in self.sweep.config.parameters]

        lines: List[str] = []
        cols = "l" + "r" * len(param_names) + "r"
        lines.append(r"\begin{tabular}{" + cols + "}")
        lines.append(r"\toprule")

        header = "Rank & " + " & ".join(param_names) + r" & Objective \\"
        lines.append(header)
        lines.append(r"\midrule")

        for rank, r in enumerate(results, 1):
            cells = [str(rank)]
            for pn in param_names:
                v = r.parameters.get(pn, "?")
                if isinstance(v, float):
                    cells.append(f"{v:.4g}")
                else:
                    cells.append(str(v))
            cells.append(f"{r.objective_value:.6g}")
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI argument parsing (stdlib argparse)
# ---------------------------------------------------------------------------


def _build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        prog="sweep",
        description="Hyperparameter sweep orchestration for Diversity Decoding Arena",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    run_p = sub.add_parser("run", help="Execute a sweep from a JSON config file")
    run_p.add_argument("config_file", help="Path to JSON sweep configuration")
    run_p.add_argument("--output", "-o", default=None, help="Output file for results")
    run_p.add_argument("--format", "-f", default="json", choices=["json", "csv", "markdown"])
    run_p.add_argument("--checkpoint-dir", default=None, help="Directory for checkpoints")
    run_p.add_argument("--verbose", "-v", action="store_true")

    # --- status ---
    status_p = sub.add_parser("status", help="Show status of a running/completed sweep")
    status_p.add_argument("checkpoint_dir", help="Checkpoint directory")

    # --- compare ---
    cmp_p = sub.add_parser("compare", help="Compare multiple sweep results")
    cmp_p.add_argument("result_files", nargs="+", help="Result JSON files to compare")
    cmp_p.add_argument("--maximize", action="store_true")

    # --- validate ---
    val_p = sub.add_parser("validate", help="Validate a sweep configuration")
    val_p.add_argument("config_file", help="Path to JSON sweep configuration")

    # --- importance ---
    imp_p = sub.add_parser("importance", help="Compute parameter importance from results")
    imp_p.add_argument("result_file", help="Result JSON file")

    # --- plot ---
    plot_p = sub.add_parser("plot", help="Generate SVG plot of sweep results")
    plot_p.add_argument("result_file", help="Result JSON file")
    plot_p.add_argument("--output", "-o", default="sweep_plot.svg")

    return parser


def _load_config_from_json(path: str) -> SweepConfig:
    """Load a SweepConfig from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    params: List[ParameterSpec] = []
    for p in data.get("parameters", []):
        params.append(
            ParameterSpec(
                name=p["name"],
                param_type=ParameterType(p.get("type", "continuous")),
                low=p.get("low"),
                high=p.get("high"),
                choices=p.get("choices"),
                default=p.get("default"),
                log_scale=p.get("log_scale", False),
                step=p.get("step"),
            )
        )

    return SweepConfig(
        strategy=SweepStrategy(data.get("strategy", "random")),
        parameters=params,
        n_trials=data.get("n_trials", 50),
        n_parallel=data.get("n_parallel", 1),
        budget=data.get("budget", float("inf")),
        objective=data.get("objective", "loss"),
        maximize=data.get("maximize", False),
        seed=data.get("seed", 42),
        early_stopping_rounds=data.get("early_stopping_rounds", 0),
        patience=data.get("patience", 10),
        checkpoint_dir=data.get("checkpoint_dir"),
    )


def _load_results_from_json(path: str) -> Tuple[SweepConfig, List[TrialResult]]:
    """Load sweep config and results from an exported JSON file."""
    with open(path) as f:
        data = json.load(f)

    config_data = data.get("config", {})
    params: List[ParameterSpec] = []
    for p in config_data.get("parameters", []):
        params.append(
            ParameterSpec(
                name=p["name"],
                param_type=ParameterType(p.get("type", "continuous")),
                low=p.get("low"),
                high=p.get("high"),
                choices=p.get("choices"),
            )
        )

    config = SweepConfig(
        strategy=SweepStrategy(config_data.get("strategy", "random")),
        parameters=params,
        n_trials=config_data.get("n_trials", 50),
        objective=config_data.get("objective", "loss"),
        maximize=config_data.get("maximize", False),
        seed=config_data.get("seed", 42),
    )

    results: List[TrialResult] = []
    for r in data.get("results", []):
        results.append(
            TrialResult(
                trial_id=r["trial_id"],
                parameters=r["parameters"],
                objective_value=r["objective_value"],
                all_metrics=r.get("all_metrics", {}),
                duration=r.get("duration", 0.0),
                status=r.get("status", "completed"),
                error=r.get("error"),
            )
        )

    return config, results


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    import sys

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "validate":
        config = _load_config_from_json(args.config_file)
        errors = ConfigValidator.validate(config)
        if errors:
            print("Validation FAILED:")
            for e in errors:
                print(f"  ✗ {e}")
            return 1
        print("Configuration is valid ✓")
        return 0

    if args.command == "status":
        ckpt_path = os.path.join(args.checkpoint_dir, "sweep_state.pkl")
        if not os.path.exists(ckpt_path):
            print(f"No checkpoint found at {ckpt_path}")
            return 1
        with open(ckpt_path, "rb") as f:
            state: SweepState = pickle.load(f)
        print(f"Completed trials: {len(state.completed_trials)}")
        print(f"Iteration: {state.iteration}")
        print(f"Budget used: {state.total_budget_used:.2f}")
        if state.best_trial:
            print(f"Best objective: {state.best_trial.objective_value:.6g}")
            print(f"Best params: {state.best_trial.parameters}")
        return 0

    if args.command == "compare":
        comparator = SweepComparator()
        for path in args.result_files:
            name = os.path.splitext(os.path.basename(path))[0]
            _, results = _load_results_from_json(path)
            comparator.add_sweep(name, results)
        print(comparator.comparison_table(maximize=args.maximize))
        print()
        print("Ranking:")
        for rank, (name, val) in enumerate(
            comparator.rank_sweeps(maximize=args.maximize), 1
        ):
            print(f"  {rank}. {name}: {val:.6g}")
        return 0

    if args.command == "importance":
        config, results = _load_results_from_json(args.result_file)
        sweep = HyperparameterSweep(config)
        importance = sweep.parameter_importance(results)
        print("Parameter Importance:")
        for name, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            bar_len = int(imp * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"  {name:20s} {bar} {imp:.4f}")
        return 0

    if args.command == "plot":
        config, results = _load_results_from_json(args.result_file)
        sweep = HyperparameterSweep(config)
        sweep.state.completed_trials = results
        for r in results:
            sweep.state.update_best(r, config.maximize)
        svg = sweep.plot_sweep_results(results)
        with open(args.output, "w") as f:
            f.write(svg)
        print(f"Plot saved to {args.output}")
        return 0

    if args.command == "run":
        config = _load_config_from_json(args.config_file)
        if args.checkpoint_dir:
            config.checkpoint_dir = args.checkpoint_dir

        ConfigValidator.validate_or_raise(config)

        # For CLI runs we need an evaluate_fn. We look for a user-provided
        # evaluator module, or default to a dummy that warns.
        evaluate_fn = _make_cli_evaluate_fn(config)

        sweep = HyperparameterSweep(config)
        if args.verbose:
            print(f"Starting {config.strategy.value} sweep with {config.n_trials} trials...")
            print(f"Parameters: {[s.name for s in config.parameters]}")

        state = sweep.run(evaluate_fn)

        if args.verbose:
            reporter = SweepReporter(sweep)
            print(reporter.full_report())

        output_str = sweep.export_results(state.completed_trials, fmt=args.format)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_str)
            print(f"Results saved to {args.output}")
        else:
            print(output_str)

        return 0

    parser.print_help()
    return 0


def _make_cli_evaluate_fn(
    config: SweepConfig,
) -> Callable[[Dict[str, Any]], Dict[str, float]]:
    """Create an evaluation function for CLI usage.

    Looks for an 'evaluate_module' field in the config's checkpoint_dir,
    otherwise returns a dummy evaluator that computes a synthetic objective.
    """

    def _dummy_evaluate(params: Dict[str, Any]) -> Dict[str, float]:
        # Synthetic Rosenbrock-like objective for testing
        values: List[float] = []
        for spec in config.parameters:
            if spec.param_type == ParameterType.CATEGORICAL:
                idx = spec.choices.index(params[spec.name]) if params[spec.name] in spec.choices else 0
                values.append(float(idx) / max(len(spec.choices) - 1, 1))
            else:
                values.append(spec.encode(params[spec.name]))

        if len(values) < 2:
            values.append(0.5)

        obj = 0.0
        for i in range(len(values) - 1):
            obj += 100 * (values[i + 1] - values[i] ** 2) ** 2 + (1 - values[i]) ** 2

        noise = np.random.normal(0, 0.01)
        obj += noise

        return {config.objective: obj, "secondary": obj * 0.5 + noise}

    return _dummy_evaluate


# ---------------------------------------------------------------------------
# Scheduling utilities for distributed sweeps
# ---------------------------------------------------------------------------


class TrialScheduler:
    """Schedule and track trial execution for potentially distributed sweeps."""

    def __init__(self, n_parallel: int = 1) -> None:
        self.n_parallel = max(n_parallel, 1)
        self._pending: List[Dict[str, Any]] = []
        self._running: Dict[str, Dict[str, Any]] = {}
        self._completed: List[TrialResult] = []
        self._failed: List[TrialResult] = []
        self._trial_counter: int = 0

    def enqueue(self, params: Dict[str, Any]) -> str:
        """Add a trial to the pending queue. Returns a trial ID."""
        tid = _trial_id(params, self._trial_counter)
        self._pending.append({"trial_id": tid, "params": params, "idx": self._trial_counter})
        self._trial_counter += 1
        return tid

    def enqueue_batch(self, param_list: List[Dict[str, Any]]) -> List[str]:
        return [self.enqueue(p) for p in param_list]

    def next_batch(self) -> List[Dict[str, Any]]:
        """Get the next batch of trials to run."""
        available_slots = self.n_parallel - len(self._running)
        batch: List[Dict[str, Any]] = []
        while self._pending and len(batch) < available_slots:
            trial = self._pending.pop(0)
            self._running[trial["trial_id"]] = trial
            batch.append(trial)
        return batch

    def complete(self, result: TrialResult) -> None:
        """Mark a trial as completed."""
        self._running.pop(result.trial_id, None)
        if result.status == "completed":
            self._completed.append(result)
        else:
            self._failed.append(result)

    def status(self) -> Dict[str, int]:
        return {
            "pending": len(self._pending),
            "running": len(self._running),
            "completed": len(self._completed),
            "failed": len(self._failed),
        }

    def all_done(self) -> bool:
        return len(self._pending) == 0 and len(self._running) == 0

    def results(self) -> List[TrialResult]:
        return list(self._completed) + list(self._failed)


# ---------------------------------------------------------------------------
# Configuration serialisation
# ---------------------------------------------------------------------------


def config_to_json(config: SweepConfig) -> str:
    """Serialise a SweepConfig to JSON."""
    data: Dict[str, Any] = {
        "strategy": config.strategy.value,
        "n_trials": config.n_trials,
        "n_parallel": config.n_parallel,
        "budget": config.budget if config.budget < float("inf") else None,
        "objective": config.objective,
        "maximize": config.maximize,
        "seed": config.seed,
        "early_stopping_rounds": config.early_stopping_rounds,
        "patience": config.patience,
        "checkpoint_dir": config.checkpoint_dir,
        "parameters": [],
    }
    for spec in config.parameters:
        p: Dict[str, Any] = {
            "name": spec.name,
            "type": spec.param_type.value,
        }
        if spec.low is not None:
            p["low"] = spec.low
        if spec.high is not None:
            p["high"] = spec.high
        if spec.choices is not None:
            p["choices"] = spec.choices
        if spec.default is not None:
            p["default"] = spec.default
        if spec.log_scale:
            p["log_scale"] = True
        if spec.step is not None:
            p["step"] = spec.step
        data["parameters"].append(p)
    return json.dumps(data, indent=2)


def config_from_json(text: str) -> SweepConfig:
    """Deserialise a SweepConfig from a JSON string."""
    return _load_config_from_json_data(json.loads(text))


def _load_config_from_json_data(data: Dict[str, Any]) -> SweepConfig:
    params: List[ParameterSpec] = []
    for p in data.get("parameters", []):
        params.append(
            ParameterSpec(
                name=p["name"],
                param_type=ParameterType(p.get("type", "continuous")),
                low=p.get("low"),
                high=p.get("high"),
                choices=p.get("choices"),
                default=p.get("default"),
                log_scale=p.get("log_scale", False),
                step=p.get("step"),
            )
        )
    budget = data.get("budget")
    if budget is None:
        budget = float("inf")
    return SweepConfig(
        strategy=SweepStrategy(data.get("strategy", "random")),
        parameters=params,
        n_trials=data.get("n_trials", 50),
        n_parallel=data.get("n_parallel", 1),
        budget=budget,
        objective=data.get("objective", "loss"),
        maximize=data.get("maximize", False),
        seed=data.get("seed", 42),
        early_stopping_rounds=data.get("early_stopping_rounds", 0),
        patience=data.get("patience", 10),
        checkpoint_dir=data.get("checkpoint_dir"),
    )


# ---------------------------------------------------------------------------
# Quick-start templates
# ---------------------------------------------------------------------------


def template_learning_rate_sweep() -> SweepConfig:
    """Template config for a learning-rate sweep."""
    return SweepConfig(
        strategy=SweepStrategy.BAYESIAN,
        parameters=[
            ParameterSpec("learning_rate", ParameterType.LOG_UNIFORM, low=1e-5, high=1e-1),
            ParameterSpec("weight_decay", ParameterType.LOG_UNIFORM, low=1e-6, high=1e-2),
            ParameterSpec("batch_size", ParameterType.CATEGORICAL, choices=[16, 32, 64, 128, 256]),
            ParameterSpec("warmup_steps", ParameterType.INT_UNIFORM, low=0, high=1000),
        ],
        n_trials=40,
        objective="val_loss",
        maximize=False,
        seed=42,
    )


def template_decoding_sweep() -> SweepConfig:
    """Template config for a diversity decoding sweep."""
    return SweepConfig(
        strategy=SweepStrategy.RANDOM,
        parameters=[
            ParameterSpec("temperature", ParameterType.CONTINUOUS, low=0.1, high=2.0),
            ParameterSpec("top_k", ParameterType.INT_UNIFORM, low=1, high=100),
            ParameterSpec("top_p", ParameterType.CONTINUOUS, low=0.1, high=1.0),
            ParameterSpec("repetition_penalty", ParameterType.CONTINUOUS, low=1.0, high=2.0),
            ParameterSpec("diversity_penalty", ParameterType.CONTINUOUS, low=0.0, high=1.0),
            ParameterSpec(
                "sampling_strategy",
                ParameterType.CATEGORICAL,
                choices=["multinomial", "beam", "contrastive", "typical", "eta"],
            ),
        ],
        n_trials=100,
        objective="diversity_score",
        maximize=True,
        seed=42,
    )


def template_architecture_sweep() -> SweepConfig:
    """Template config for a model architecture sweep."""
    return SweepConfig(
        strategy=SweepStrategy.HYPERBAND,
        parameters=[
            ParameterSpec("n_layers", ParameterType.INT_UNIFORM, low=2, high=24),
            ParameterSpec("hidden_dim", ParameterType.CATEGORICAL, choices=[128, 256, 512, 1024, 2048]),
            ParameterSpec("n_heads", ParameterType.CATEGORICAL, choices=[4, 8, 16, 32]),
            ParameterSpec("dropout", ParameterType.CONTINUOUS, low=0.0, high=0.5),
            ParameterSpec("activation", ParameterType.CATEGORICAL, choices=["relu", "gelu", "silu", "swish"]),
        ],
        n_trials=81,
        budget=81.0,
        objective="val_loss",
        maximize=False,
        seed=42,
    )


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "SweepStrategy",
    "ParameterType",
    "ParameterSpec",
    "SweepConfig",
    "TrialResult",
    "SweepState",
    "ParameterGrid",
    "BayesianOptimizer",
    "SuccessiveHalving",
    "HyperparameterSweep",
    "SensitivityAnalyzer",
    "MultiObjectiveResult",
    "AdaptiveBudgetAllocator",
    "ResultCache",
    "ConfigValidator",
    "SweepReporter",
    "SweepComparator",
    "ParameterSpaceUtils",
    "ConstraintHandler",
    "SearchSpaceTransformer",
    "WarmStarter",
    "ConvergenceDiagnostics",
    "TrialScheduler",
    "config_to_json",
    "config_from_json",
    "make_sweep_config",
    "make_parameter",
    "template_learning_rate_sweep",
    "template_decoding_sweep",
    "template_architecture_sweep",
    "main",
]


if __name__ == "__main__":
    import sys

    sys.exit(main())
