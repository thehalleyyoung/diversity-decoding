"""
Computational cost modeling for the Diversity Decoding Arena.

Provides estimation and tracking of wall-clock time, FLOPs, and memory
usage for each decoding algorithm, along with calibration, optimization,
scaling analysis, and reporting utilities.
"""

from __future__ import annotations

import math
import os
import platform
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AlgorithmName(str, Enum):
    TEMPERATURE = "temperature"
    TOP_K = "top_k"
    NUCLEUS = "nucleus"
    DIVERSE_BEAM = "diverse_beam"
    DPP = "dpp"
    SVD = "svd"
    QD_BS = "qd_bs"


class CostComponent(str, Enum):
    INFERENCE = "inference"
    METRIC_COMPUTATION = "metric_computation"
    KERNEL_BUILD = "kernel_build"
    SORTING = "sorting"
    SAMPLING = "sampling"
    RERANKING = "reranking"
    EMBEDDING = "embedding"
    OTHER = "other"


# ---------------------------------------------------------------------------
# AlgorithmComplexity
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmComplexity:
    """Theoretical complexity descriptor for one decoding algorithm."""

    name: str
    time_formula: str
    space_formula: str
    key_params: List[str]
    _time_fn: Callable[..., float] = field(repr=False, default=lambda **kw: 0.0)
    _space_fn: Callable[..., float] = field(repr=False, default=lambda **kw: 0.0)

    def eval_time(self, **params: float) -> float:
        return self._time_fn(**params)

    def eval_space(self, **params: float) -> float:
        return self._space_fn(**params)


def _temperature_time(T: float = 1, n: float = 1, V: float = 1, **_: Any) -> float:
    return T * n * V


def _temperature_space(n: float = 1, V: float = 1, **_: Any) -> float:
    return n * V


def _topk_time(T: float = 1, n: float = 1, V: float = 1, k: float = 1, **_: Any) -> float:
    return T * n * V * max(math.log2(max(k, 2)), 1.0)


def _topk_space(n: float = 1, V: float = 1, **_: Any) -> float:
    return n * V


def _nucleus_time(T: float = 1, n: float = 1, V: float = 1, **_: Any) -> float:
    return T * n * V * math.log2(max(V, 2))


def _nucleus_space(n: float = 1, V: float = 1, **_: Any) -> float:
    return n * V


def _diverse_beam_time(
    T: float = 1, B: float = 1, G: float = 1, V: float = 1, **_: Any
) -> float:
    return T * B * G * V


def _diverse_beam_space(B: float = 1, T: float = 1, V: float = 1, **_: Any) -> float:
    return B * T * V


def _dpp_time(
    n: float = 1, T: float = 1, V: float = 1, **_: Any
) -> float:
    return n ** 3 + n * T * V


def _dpp_space(n: float = 1, V: float = 1, **_: Any) -> float:
    return n * n + n * V


def _svd_time(
    T: float = 1, n: float = 1, K: float = 1, d: float = 1,
    C_embed: float = 1, **_: Any,
) -> float:
    return T * n ** 2 * (K * d + C_embed)


def _svd_space(
    n: float = 1, d: float = 1, K: float = 1, **_: Any,
) -> float:
    return n * d + n * K


def _qdbs_time(
    T: float = 1, B: float = 1, V: float = 1, C: float = 1, **_: Any,
) -> float:
    return T * B * V * C


def _qdbs_space(
    B: float = 1, T: float = 1, C: float = 1, V: float = 1, **_: Any,
) -> float:
    return B * T + C * V


ALGORITHM_COMPLEXITIES: Dict[str, AlgorithmComplexity] = {
    AlgorithmName.TEMPERATURE: AlgorithmComplexity(
        name="temperature",
        time_formula="O(T * n * V)",
        space_formula="O(n * V)",
        key_params=["T", "n", "V"],
        _time_fn=_temperature_time,
        _space_fn=_temperature_space,
    ),
    AlgorithmName.TOP_K: AlgorithmComplexity(
        name="top_k",
        time_formula="O(T * n * V * log(k))",
        space_formula="O(n * V)",
        key_params=["T", "n", "V", "k"],
        _time_fn=_topk_time,
        _space_fn=_topk_space,
    ),
    AlgorithmName.NUCLEUS: AlgorithmComplexity(
        name="nucleus",
        time_formula="O(T * n * V * log(V))",
        space_formula="O(n * V)",
        key_params=["T", "n", "V"],
        _time_fn=_nucleus_time,
        _space_fn=_nucleus_space,
    ),
    AlgorithmName.DIVERSE_BEAM: AlgorithmComplexity(
        name="diverse_beam",
        time_formula="O(T * B * G * V)",
        space_formula="O(B * T * V)",
        key_params=["T", "B", "G", "V"],
        _time_fn=_diverse_beam_time,
        _space_fn=_diverse_beam_space,
    ),
    AlgorithmName.DPP: AlgorithmComplexity(
        name="dpp",
        time_formula="O(n^3 + n * T * V)",
        space_formula="O(n^2 + n * V)",
        key_params=["n", "T", "V"],
        _time_fn=_dpp_time,
        _space_fn=_dpp_space,
    ),
    AlgorithmName.SVD: AlgorithmComplexity(
        name="svd",
        time_formula="O(T * n^2 * (K*d + C_embed))",
        space_formula="O(n*d + n*K)",
        key_params=["T", "n", "K", "d", "C_embed"],
        _time_fn=_svd_time,
        _space_fn=_svd_space,
    ),
    AlgorithmName.QD_BS: AlgorithmComplexity(
        name="qd_bs",
        time_formula="O(T * B * V * C)",
        space_formula="O(B*T + C*V)",
        key_params=["T", "B", "V", "C"],
        _time_fn=_qdbs_time,
        _space_fn=_qdbs_space,
    ),
}


# ---------------------------------------------------------------------------
# HardwareProfile
# ---------------------------------------------------------------------------

@dataclass
class GPUProfile:
    """Optional GPU descriptor."""

    name: str = "unknown"
    compute_units: int = 0
    clock_mhz: float = 0.0
    memory_bytes: int = 0
    bandwidth_gbps: float = 0.0
    fp32_tflops: float = 0.0
    fp16_tflops: float = 0.0

    @property
    def peak_flops(self) -> float:
        return self.fp32_tflops * 1e12

    @property
    def memory_gb(self) -> float:
        return self.memory_bytes / (1024 ** 3)


@dataclass
class HardwareProfile:
    """Describes the hardware available for cost estimation."""

    cpu_cores: int = 1
    cpu_clock_ghz: float = 2.0
    cache_l1_kb: int = 32
    cache_l2_kb: int = 256
    cache_l3_mb: int = 8
    simd_width: int = 256  # AVX-256 bits
    ram_bytes: int = 8 * (1024 ** 3)
    ram_bandwidth_gbps: float = 25.0
    gpu: Optional[GPUProfile] = None

    @property
    def ram_gb(self) -> float:
        return self.ram_bytes / (1024 ** 3)

    @property
    def peak_cpu_flops(self) -> float:
        ops_per_cycle = self.simd_width / 32  # fp32 ops per SIMD lane
        return self.cpu_cores * self.cpu_clock_ghz * 1e9 * ops_per_cycle * 2  # FMA

    @property
    def has_gpu(self) -> bool:
        return self.gpu is not None and self.gpu.compute_units > 0

    @classmethod
    def detect(cls) -> "HardwareProfile":
        """Best-effort detection of the current machine's hardware."""
        cores = os.cpu_count() or 1
        bits = struct.calcsize("P") * 8
        is_arm = platform.machine().startswith("arm") or platform.machine().startswith("aarch")

        clock_ghz = 2.4
        simd = 128 if is_arm else 256
        cache_l3 = 8

        try:
            import subprocess

            if platform.system() == "Darwin":
                out = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True, timeout=5
                ).strip()
                ram = int(out)
                try:
                    freq = subprocess.check_output(
                        ["sysctl", "-n", "hw.cpufrequency_max"], text=True, timeout=5
                    ).strip()
                    clock_ghz = int(freq) / 1e9
                except Exception:
                    pass
            elif platform.system() == "Linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            ram = int(line.split()[1]) * 1024
                            break
                    else:
                        ram = 8 * (1024 ** 3)
                try:
                    with open("/proc/cpuinfo") as f:
                        for line in f:
                            if "cpu MHz" in line:
                                clock_ghz = float(line.split(":")[1]) / 1000.0
                                break
                except Exception:
                    pass
            else:
                ram = 8 * (1024 ** 3)
        except Exception:
            ram = 8 * (1024 ** 3)

        return cls(
            cpu_cores=cores,
            cpu_clock_ghz=round(clock_ghz, 2),
            cache_l1_kb=32,
            cache_l2_kb=256,
            cache_l3_mb=cache_l3,
            simd_width=simd,
            ram_bytes=ram,
            ram_bandwidth_gbps=25.0,
            gpu=None,
        )

    def summary(self) -> str:
        lines = [
            f"CPU: {self.cpu_cores} cores @ {self.cpu_clock_ghz} GHz  "
            f"(SIMD {self.simd_width}-bit)",
            f"RAM: {self.ram_gb:.1f} GB  bandwidth {self.ram_bandwidth_gbps} GB/s",
            f"Peak CPU FLOPS: {self.peak_cpu_flops:.2e}",
        ]
        if self.has_gpu:
            g = self.gpu
            lines.append(
                f"GPU: {g.name}  {g.compute_units} CUs  "  # type: ignore[union-attr]
                f"{g.memory_gb:.1f} GB  {g.fp32_tflops:.1f} TFLOPS"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Algorithm config helpers
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmConfig:
    """Minimal algorithm configuration for cost prediction."""

    algorithm: str
    num_sequences: int = 10
    max_tokens: int = 128
    vocab_size: int = 50257
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    num_beams: int = 5
    num_beam_groups: int = 2
    embed_dim: int = 768
    num_particles: int = 10
    kernel_components: int = 5
    qd_cells: int = 100

    def to_complexity_params(self) -> Dict[str, float]:
        algo = self.algorithm
        base = {
            "T": float(self.max_tokens),
            "n": float(self.num_sequences),
            "V": float(self.vocab_size),
        }
        if algo == AlgorithmName.TOP_K:
            base["k"] = float(self.top_k)
        elif algo == AlgorithmName.DIVERSE_BEAM:
            base["B"] = float(self.num_beams)
            base["G"] = float(self.num_beam_groups)
        elif algo == AlgorithmName.DPP:
            pass
        elif algo == AlgorithmName.SVD:
            base["K"] = float(self.num_particles)
            base["d"] = float(self.embed_dim)
            base["C_embed"] = float(self.embed_dim) * 4.0
        elif algo == AlgorithmName.QD_BS:
            base["B"] = float(self.num_beams)
            base["C"] = float(self.qd_cells)
        return base


# ---------------------------------------------------------------------------
# CostModel
# ---------------------------------------------------------------------------

class CostModel:
    """Predict wall-clock time, FLOPs, and memory for any algorithm/config."""

    # Empirical constant: how many "complexity units" per second the
    # hardware can chew through.  Calibrated to a baseline machine.
    _DEFAULT_RATE = 2.0e9  # complexity units / second on baseline HW

    def __init__(
        self,
        hardware: Optional[HardwareProfile] = None,
        calibration: Optional[Dict[str, float]] = None,
    ) -> None:
        self.hardware = hardware or HardwareProfile.detect()
        self.calibration: Dict[str, float] = calibration or {}
        self._rate = self._compute_rate()

    def _compute_rate(self) -> float:
        """Scale the default rate by hardware capability."""
        baseline_flops = 2.4e9 * 8 * 8 * 2  # 8-core 2.4 GHz AVX-256 FMA
        ratio = self.hardware.peak_cpu_flops / max(baseline_flops, 1.0)
        return self._DEFAULT_RATE * ratio

    def predict_flops(self, algorithm: str, config: AlgorithmConfig) -> int:
        """Return estimated FLOPs (floating-point operations)."""
        comp = ALGORITHM_COMPLEXITIES.get(algorithm)
        if comp is None:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        params = config.to_complexity_params()
        raw = comp.eval_time(**params)

        # Each "complexity unit" ≈ a small constant number of FP ops
        flops_per_unit = 6.0
        estimated = int(raw * flops_per_unit)

        cal = self.calibration.get(f"{algorithm}_flops_scale", 1.0)
        return int(estimated * cal)

    def predict_time(
        self,
        algorithm: str,
        config: AlgorithmConfig,
        hardware: Optional[HardwareProfile] = None,
    ) -> float:
        """Return estimated wall-clock time in seconds."""
        hw = hardware or self.hardware
        flops = self.predict_flops(algorithm, config)

        peak = hw.peak_cpu_flops
        if hw.has_gpu and algorithm in {
            AlgorithmName.DIVERSE_BEAM, AlgorithmName.DPP, AlgorithmName.SVD
        }:
            peak = max(peak, hw.gpu.peak_flops)  # type: ignore[union-attr]

        # Assume ~15 % of peak is sustained throughput
        sustained = peak * 0.15
        raw_time = flops / max(sustained, 1.0)

        # Memory-bandwidth bottleneck adjustment
        mem_bytes = self.predict_memory(algorithm, config, hw)
        transfer_time = mem_bytes / (hw.ram_bandwidth_gbps * 1e9)
        raw_time = max(raw_time, transfer_time)

        cal = self.calibration.get(f"{algorithm}_time_scale", 1.0)
        return raw_time * cal

    def predict_memory(
        self,
        algorithm: str,
        config: AlgorithmConfig,
        hardware: Optional[HardwareProfile] = None,
    ) -> float:
        """Return estimated peak memory usage in bytes."""
        comp = ALGORITHM_COMPLEXITIES.get(algorithm)
        if comp is None:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        params = config.to_complexity_params()
        raw_units = comp.eval_space(**params)

        # Each unit ≈ one fp32 value (4 bytes) + bookkeeping
        bytes_per_unit = 4.0
        overhead_factor = 1.2  # Python / framework overhead
        estimated = raw_units * bytes_per_unit * overhead_factor

        cal = self.calibration.get(f"{algorithm}_mem_scale", 1.0)
        return estimated * cal

    def predict_all(
        self,
        algorithm: str,
        config: AlgorithmConfig,
        hardware: Optional[HardwareProfile] = None,
    ) -> Dict[str, float]:
        hw = hardware or self.hardware
        return {
            "time_s": self.predict_time(algorithm, config, hw),
            "memory_bytes": self.predict_memory(algorithm, config, hw),
            "flops": float(self.predict_flops(algorithm, config)),
        }

    def compare_algorithms(
        self,
        configs: Dict[str, AlgorithmConfig],
        hardware: Optional[HardwareProfile] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Return predictions keyed by algorithm name."""
        return {
            name: self.predict_all(cfg.algorithm, cfg, hardware)
            for name, cfg in configs.items()
        }


# ---------------------------------------------------------------------------
# CostBenchmark
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    algorithm: str
    config: AlgorithmConfig
    predicted_time: float
    actual_time: float
    predicted_memory: float
    actual_memory: float
    predicted_flops: int
    calibration_factor_time: float
    calibration_factor_memory: float
    iterations: int = 1

    @property
    def time_error_pct(self) -> float:
        if self.actual_time == 0:
            return 0.0
        return abs(self.predicted_time - self.actual_time) / self.actual_time * 100

    @property
    def memory_error_pct(self) -> float:
        if self.actual_memory == 0:
            return 0.0
        return abs(self.predicted_memory - self.actual_memory) / self.actual_memory * 100


class CostBenchmark:
    """Run micro-benchmarks to calibrate the cost model."""

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        iterations: int = 5,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.iterations = iterations
        self.results: List[BenchmarkResult] = []

    # -- Synthetic workload generators ----------------------------------------

    @staticmethod
    def _workload_temperature(config: AlgorithmConfig) -> Callable[[], None]:
        V = min(config.vocab_size, 5000)
        n = config.num_sequences
        T = min(config.max_tokens, 32)
        rng = np.random.default_rng(42)

        def work() -> None:
            logits = rng.standard_normal((n, V)).astype(np.float32)
            for _ in range(T):
                scaled = logits / max(config.temperature, 1e-8)
                mx = scaled.max(axis=-1, keepdims=True)
                exp = np.exp(scaled - mx)
                probs = exp / exp.sum(axis=-1, keepdims=True)
                _ = (probs.cumsum(axis=-1) > rng.random((n, 1))).argmax(axis=-1)
        return work

    @staticmethod
    def _workload_topk(config: AlgorithmConfig) -> Callable[[], None]:
        V = min(config.vocab_size, 5000)
        n = config.num_sequences
        T = min(config.max_tokens, 32)
        k = min(config.top_k, V)
        rng = np.random.default_rng(42)

        def work() -> None:
            logits = rng.standard_normal((n, V)).astype(np.float32)
            for _ in range(T):
                idx = np.argpartition(logits, -k, axis=-1)[:, -k:]
                vals = np.take_along_axis(logits, idx, axis=-1)
                mx = vals.max(axis=-1, keepdims=True)
                exp = np.exp(vals - mx)
                probs = exp / exp.sum(axis=-1, keepdims=True)
                _ = idx[
                    np.arange(n),
                    (probs.cumsum(axis=-1) > rng.random((n, 1))).argmax(axis=-1),
                ]
        return work

    @staticmethod
    def _workload_nucleus(config: AlgorithmConfig) -> Callable[[], None]:
        V = min(config.vocab_size, 5000)
        n = config.num_sequences
        T = min(config.max_tokens, 32)
        p = config.top_p
        rng = np.random.default_rng(42)

        def work() -> None:
            logits = rng.standard_normal((n, V)).astype(np.float32)
            for _ in range(T):
                sorted_idx = np.argsort(logits, axis=-1)[:, ::-1]
                sorted_logits = np.take_along_axis(logits, sorted_idx, axis=-1)
                mx = sorted_logits.max(axis=-1, keepdims=True)
                exp = np.exp(sorted_logits - mx)
                probs = exp / exp.sum(axis=-1, keepdims=True)
                cum = probs.cumsum(axis=-1)
                mask = cum <= p
                mask[:, 0] = True
                filtered = probs * mask
                filtered /= filtered.sum(axis=-1, keepdims=True)
                _ = (filtered.cumsum(axis=-1) > rng.random((n, 1))).argmax(axis=-1)
        return work

    @staticmethod
    def _workload_diverse_beam(config: AlgorithmConfig) -> Callable[[], None]:
        V = min(config.vocab_size, 2000)
        B = config.num_beams
        G = config.num_beam_groups
        T = min(config.max_tokens, 16)
        rng = np.random.default_rng(42)

        def work() -> None:
            beams_per_group = max(B // G, 1)
            for _ in range(T):
                for _g in range(G):
                    scores = rng.standard_normal((beams_per_group, V)).astype(np.float32)
                    best = np.argsort(scores, axis=-1)[:, -beams_per_group:]
                    _ = np.take_along_axis(scores, best, axis=-1)
        return work

    @staticmethod
    def _workload_dpp(config: AlgorithmConfig) -> Callable[[], None]:
        n = min(config.num_sequences, 50)
        d = min(config.embed_dim, 128)
        rng = np.random.default_rng(42)

        def work() -> None:
            features = rng.standard_normal((n, d)).astype(np.float32)
            L = features @ features.T
            eigvals, eigvecs = np.linalg.eigh(L)
            selected: List[int] = []
            probs = eigvals / (eigvals + 1.0)
            mask = rng.random(len(probs)) < probs
            V_sel = eigvecs[:, mask]
            for _ in range(min(int(mask.sum()), n)):
                if V_sel.shape[1] == 0:
                    break
                norms = np.sum(V_sel ** 2, axis=1)
                idx = int(np.argmax(norms))
                selected.append(idx)
                if V_sel.shape[1] <= 1:
                    break
                proj = V_sel[idx] / np.linalg.norm(V_sel[idx])
                V_sel = V_sel - np.outer(V_sel @ proj, proj)
        return work

    @staticmethod
    def _workload_svd(config: AlgorithmConfig) -> Callable[[], None]:
        n = min(config.num_sequences, 20)
        d = min(config.embed_dim, 128)
        T = min(config.max_tokens, 16)
        rng = np.random.default_rng(42)

        def work() -> None:
            particles = rng.standard_normal((n, d)).astype(np.float32)
            for _ in range(T):
                dists = np.sum(
                    (particles[:, None, :] - particles[None, :, :]) ** 2, axis=-1
                )
                kernel = np.exp(-dists / (2.0 * d))
                grad = np.zeros_like(particles)
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            diff = particles[i] - particles[j]
                            grad[i] += kernel[i, j] * diff / d
                particles += 0.01 * grad
        return work

    @staticmethod
    def _workload_qdbs(config: AlgorithmConfig) -> Callable[[], None]:
        B = config.num_beams
        C = min(config.qd_cells, 50)
        V = min(config.vocab_size, 2000)
        T = min(config.max_tokens, 16)
        rng = np.random.default_rng(42)

        def work() -> None:
            archive = np.full(C, -np.inf, dtype=np.float32)
            for _ in range(T):
                for _b in range(B):
                    scores = rng.standard_normal(V).astype(np.float32)
                    best_tok = int(np.argmax(scores))
                    cell = best_tok % C
                    val = scores[best_tok]
                    if val > archive[cell]:
                        archive[cell] = val
        return work

    _WORKLOAD_REGISTRY: Dict[str, Callable[["AlgorithmConfig"], Callable[[], None]]] = {}

    @classmethod
    def _register_workloads(cls) -> None:
        if cls._WORKLOAD_REGISTRY:
            return
        cls._WORKLOAD_REGISTRY = {
            AlgorithmName.TEMPERATURE: cls._workload_temperature.__func__,  # type: ignore[attr-defined]
            AlgorithmName.TOP_K: cls._workload_topk.__func__,  # type: ignore[attr-defined]
            AlgorithmName.NUCLEUS: cls._workload_nucleus.__func__,  # type: ignore[attr-defined]
            AlgorithmName.DIVERSE_BEAM: cls._workload_diverse_beam.__func__,  # type: ignore[attr-defined]
            AlgorithmName.DPP: cls._workload_dpp.__func__,  # type: ignore[attr-defined]
            AlgorithmName.SVD: cls._workload_svd.__func__,  # type: ignore[attr-defined]
            AlgorithmName.QD_BS: cls._workload_qdbs.__func__,  # type: ignore[attr-defined]
        }

    def _measure_time(self, fn: Callable[[], None]) -> float:
        """Return median wall-clock time over self.iterations runs."""
        times: List[float] = []
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return float(np.median(times))

    @staticmethod
    def _estimate_actual_memory(fn: Callable[[], None]) -> float:
        """Rough estimate of peak memory by checking before/after."""
        try:
            import tracemalloc

            tracemalloc.start()
            fn()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return float(peak)
        except Exception:
            return 0.0

    def benchmark_algorithm(
        self, algorithm: str, config: AlgorithmConfig
    ) -> BenchmarkResult:
        self.__class__._register_workloads()
        factory = self._WORKLOAD_REGISTRY.get(algorithm)
        if factory is None:
            raise ValueError(f"No workload for algorithm {algorithm}")

        workload = factory(config)
        predicted_time = self.cost_model.predict_time(algorithm, config)
        predicted_memory = self.cost_model.predict_memory(algorithm, config)
        predicted_flops = self.cost_model.predict_flops(algorithm, config)
        actual_time = self._measure_time(workload)
        actual_memory = self._estimate_actual_memory(workload)

        cal_time = actual_time / max(predicted_time, 1e-15)
        cal_mem = actual_memory / max(predicted_memory, 1e-15) if actual_memory > 0 else 1.0

        result = BenchmarkResult(
            algorithm=algorithm,
            config=config,
            predicted_time=predicted_time,
            actual_time=actual_time,
            predicted_memory=predicted_memory,
            actual_memory=actual_memory,
            predicted_flops=predicted_flops,
            calibration_factor_time=cal_time,
            calibration_factor_memory=cal_mem,
            iterations=self.iterations,
        )
        self.results.append(result)
        return result

    def benchmark_all(
        self, configs: Optional[Dict[str, AlgorithmConfig]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark every algorithm (or a supplied dict)."""
        if configs is None:
            configs = {
                algo: AlgorithmConfig(algorithm=algo)
                for algo in AlgorithmName
            }
        results: List[BenchmarkResult] = []
        for name, cfg in configs.items():
            r = self.benchmark_algorithm(cfg.algorithm, cfg)
            results.append(r)
        return results

    def compute_calibration(self) -> Dict[str, float]:
        """Aggregate calibration factors from collected results."""
        factors: Dict[str, List[float]] = {}
        for r in self.results:
            key_t = f"{r.algorithm}_time_scale"
            key_m = f"{r.algorithm}_mem_scale"
            factors.setdefault(key_t, []).append(r.calibration_factor_time)
            factors.setdefault(key_m, []).append(r.calibration_factor_memory)
        return {k: float(np.median(v)) for k, v in factors.items()}

    def apply_calibration(self) -> None:
        cal = self.compute_calibration()
        self.cost_model.calibration.update(cal)

    def regression_fit(self) -> Dict[str, Tuple[float, float]]:
        """
        Simple linear regression predicted -> actual for each algorithm.

        Returns {algorithm: (slope, intercept)}.
        """
        groups: Dict[str, Tuple[List[float], List[float]]] = {}
        for r in self.results:
            pred, act = groups.setdefault(r.algorithm, ([], []))
            pred.append(r.predicted_time)
            act.append(r.actual_time)

        fits: Dict[str, Tuple[float, float]] = {}
        for algo, (preds, acts) in groups.items():
            if len(preds) < 2:
                fits[algo] = (acts[0] / max(preds[0], 1e-15), 0.0)
                continue
            x = np.array(preds)
            y = np.array(acts)
            A = np.vstack([x, np.ones_like(x)]).T
            result = np.linalg.lstsq(A, y, rcond=None)
            slope, intercept = result[0]
            fits[algo] = (float(slope), float(intercept))
        return fits


# ---------------------------------------------------------------------------
# CostOptimizer
# ---------------------------------------------------------------------------

@dataclass
class OptimizationObjective:
    """Weights for multi-objective optimization."""

    time_weight: float = 1.0
    diversity_weight: float = 1.0
    quality_weight: float = 1.0


@dataclass
class ParetoPoint:
    algorithm: str
    config: AlgorithmConfig
    time_s: float
    diversity_score: float
    quality_score: float
    dominated: bool = False

    @property
    def cost_efficiency(self) -> float:
        if self.time_s <= 0:
            return float("inf")
        return (self.diversity_score + self.quality_score) / self.time_s


class CostOptimizer:
    """Given a time budget, suggest optimal algorithm/config."""

    def __init__(
        self,
        cost_model: CostModel,
        diversity_estimator: Optional[Callable[[AlgorithmConfig], float]] = None,
        quality_estimator: Optional[Callable[[AlgorithmConfig], float]] = None,
    ) -> None:
        self.cost_model = cost_model
        self._diversity_est = diversity_estimator or self._default_diversity
        self._quality_est = quality_estimator or self._default_quality

    @staticmethod
    def _default_diversity(cfg: AlgorithmConfig) -> float:
        """Heuristic diversity estimate: algorithms that do more work tend to
        produce higher diversity."""
        algo_base: Dict[str, float] = {
            AlgorithmName.TEMPERATURE: 0.5,
            AlgorithmName.TOP_K: 0.45,
            AlgorithmName.NUCLEUS: 0.55,
            AlgorithmName.DIVERSE_BEAM: 0.7,
            AlgorithmName.DPP: 0.85,
            AlgorithmName.SVD: 0.90,
            AlgorithmName.QD_BS: 0.80,
        }
        base = algo_base.get(cfg.algorithm, 0.5)
        n_bonus = min(cfg.num_sequences / 50.0, 0.1)
        return min(base + n_bonus, 1.0)

    @staticmethod
    def _default_quality(cfg: AlgorithmConfig) -> float:
        algo_base: Dict[str, float] = {
            AlgorithmName.TEMPERATURE: 0.70,
            AlgorithmName.TOP_K: 0.72,
            AlgorithmName.NUCLEUS: 0.74,
            AlgorithmName.DIVERSE_BEAM: 0.68,
            AlgorithmName.DPP: 0.65,
            AlgorithmName.SVD: 0.63,
            AlgorithmName.QD_BS: 0.66,
        }
        return algo_base.get(cfg.algorithm, 0.65)

    def suggest_algorithm(
        self,
        time_budget: float,
        num_sequences: int = 10,
        max_tokens: int = 128,
        vocab_size: int = 50257,
    ) -> Optional[Tuple[str, AlgorithmConfig, Dict[str, float]]]:
        """Return the best algorithm/config within the time budget."""
        candidates: List[Tuple[float, str, AlgorithmConfig, Dict[str, float]]] = []
        for algo in AlgorithmName:
            cfg = AlgorithmConfig(
                algorithm=algo,
                num_sequences=num_sequences,
                max_tokens=max_tokens,
                vocab_size=vocab_size,
            )
            pred = self.cost_model.predict_all(algo, cfg)
            if pred["time_s"] > time_budget:
                continue
            div = self._diversity_est(cfg)
            qual = self._quality_est(cfg)
            score = div + qual - 0.2 * pred["time_s"] / max(time_budget, 1e-9)
            candidates.append((score, algo, cfg, {**pred, "diversity": div, "quality": qual}))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, best_algo, best_cfg, best_pred = candidates[0]
        return best_algo, best_cfg, best_pred

    def knapsack_allocate(
        self,
        tasks: List[Dict[str, Any]],
        total_budget: float,
    ) -> List[Dict[str, Any]]:
        """
        Allocate a time budget across multiple tasks using a greedy
        fractional-knapsack approach.

        Each task dict must contain "name" and optionally "priority" (float,
        higher is more important, default 1.0).

        Returns a list of dicts with "name", "algorithm", "config", "time_s",
        "fraction" keys.
        """
        n_tasks = len(tasks)
        if n_tasks == 0:
            return []

        per_task_budget = total_budget / n_tasks
        task_costs: List[Tuple[float, int, str, AlgorithmConfig, float]] = []

        for i, task in enumerate(tasks):
            priority = task.get("priority", 1.0)
            result = self.suggest_algorithm(
                time_budget=per_task_budget * 2.0,
                num_sequences=task.get("num_sequences", 10),
                max_tokens=task.get("max_tokens", 128),
                vocab_size=task.get("vocab_size", 50257),
            )
            if result is None:
                continue
            algo, cfg, pred = result
            value = priority * (pred.get("diversity", 0.5) + pred.get("quality", 0.5))
            cost = pred["time_s"]
            efficiency = value / max(cost, 1e-15)
            task_costs.append((efficiency, i, algo, cfg, cost))

        task_costs.sort(key=lambda x: x[0], reverse=True)

        remaining = total_budget
        allocations: List[Dict[str, Any]] = []
        for efficiency, i, algo, cfg, cost in task_costs:
            if remaining <= 0:
                break
            fraction = min(1.0, remaining / max(cost, 1e-15))
            allocated_time = cost * fraction
            remaining -= allocated_time
            allocations.append({
                "name": tasks[i].get("name", f"task_{i}"),
                "algorithm": algo,
                "config": cfg,
                "time_s": allocated_time,
                "fraction": fraction,
            })
        return allocations

    def pareto_frontier(
        self,
        num_sequences: int = 10,
        max_tokens: int = 128,
        vocab_size: int = 50257,
        n_configs_per_algo: int = 5,
    ) -> List[ParetoPoint]:
        """
        Compute the Pareto frontier in (time, diversity, quality) space.

        We sample a grid of configurations for each algorithm and return
        non-dominated points.
        """
        points: List[ParetoPoint] = []

        seq_range = np.linspace(
            max(2, num_sequences // 2), num_sequences * 2, n_configs_per_algo
        ).astype(int)

        for algo in AlgorithmName:
            for ns in seq_range:
                cfg = AlgorithmConfig(
                    algorithm=algo,
                    num_sequences=int(ns),
                    max_tokens=max_tokens,
                    vocab_size=vocab_size,
                )
                pred = self.cost_model.predict_all(algo, cfg)
                div = self._diversity_est(cfg)
                qual = self._quality_est(cfg)
                points.append(ParetoPoint(
                    algorithm=algo,
                    config=cfg,
                    time_s=pred["time_s"],
                    diversity_score=div,
                    quality_score=qual,
                ))

        # Mark dominated points
        for i, pi in enumerate(points):
            for j, pj in enumerate(points):
                if i == j:
                    continue
                if (
                    pj.time_s <= pi.time_s
                    and pj.diversity_score >= pi.diversity_score
                    and pj.quality_score >= pi.quality_score
                    and (
                        pj.time_s < pi.time_s
                        or pj.diversity_score > pi.diversity_score
                        or pj.quality_score > pi.quality_score
                    )
                ):
                    pi.dominated = True
                    break

        return points

    def pareto_front_only(self, **kwargs: Any) -> List[ParetoPoint]:
        all_pts = self.pareto_frontier(**kwargs)
        return [p for p in all_pts if not p.dominated]

    def weighted_score(
        self,
        point: ParetoPoint,
        objective: OptimizationObjective,
        max_time: float = 1.0,
    ) -> float:
        time_norm = 1.0 - min(point.time_s / max(max_time, 1e-9), 1.0)
        return (
            objective.time_weight * time_norm
            + objective.diversity_weight * point.diversity_score
            + objective.quality_weight * point.quality_score
        )


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

@dataclass
class CostRecord:
    algorithm: str
    component: str
    start_time: float
    end_time: float
    tokens_generated: int = 0
    sequences_generated: int = 0
    memory_peak_bytes: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        return self.end_time - self.start_time


class CostTracker:
    """Track actual costs during experiments."""

    def __init__(self, budget_seconds: Optional[float] = None) -> None:
        self.records: List[CostRecord] = []
        self.budget_seconds = budget_seconds
        self._active: Dict[str, float] = {}  # label -> start_time

    def start(self, label: str) -> None:
        self._active[label] = time.perf_counter()

    def stop(
        self,
        label: str,
        algorithm: str = "unknown",
        component: str = CostComponent.OTHER,
        tokens: int = 0,
        sequences: int = 0,
        memory_peak: float = 0.0,
        **metadata: Any,
    ) -> CostRecord:
        start = self._active.pop(label, time.perf_counter())
        end = time.perf_counter()
        rec = CostRecord(
            algorithm=algorithm,
            component=component,
            start_time=start,
            end_time=end,
            tokens_generated=tokens,
            sequences_generated=sequences,
            memory_peak_bytes=memory_peak,
            metadata=metadata,
        )
        self.records.append(rec)
        return rec

    def record(
        self,
        algorithm: str,
        component: str,
        elapsed: float,
        tokens: int = 0,
        sequences: int = 0,
        memory_peak: float = 0.0,
        **metadata: Any,
    ) -> CostRecord:
        now = time.perf_counter()
        rec = CostRecord(
            algorithm=algorithm,
            component=component,
            start_time=now - elapsed,
            end_time=now,
            tokens_generated=tokens,
            sequences_generated=sequences,
            memory_peak_bytes=memory_peak,
            metadata=metadata,
        )
        self.records.append(rec)
        return rec

    @property
    def total_time(self) -> float:
        return sum(r.elapsed for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_generated for r in self.records)

    @property
    def total_sequences(self) -> int:
        return sum(r.sequences_generated for r in self.records)

    @property
    def peak_memory(self) -> float:
        if not self.records:
            return 0.0
        return max(r.memory_peak_bytes for r in self.records)

    @property
    def budget_remaining(self) -> Optional[float]:
        if self.budget_seconds is None:
            return None
        return max(0.0, self.budget_seconds - self.total_time)

    @property
    def budget_fraction_used(self) -> Optional[float]:
        if self.budget_seconds is None or self.budget_seconds <= 0:
            return None
        return min(1.0, self.total_time / self.budget_seconds)

    def cost_per_token(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.total_time / self.total_tokens

    def cost_per_sequence(self) -> float:
        if self.total_sequences == 0:
            return 0.0
        return self.total_time / self.total_sequences

    def cost_by_algorithm(self) -> Dict[str, float]:
        costs: Dict[str, float] = {}
        for r in self.records:
            costs[r.algorithm] = costs.get(r.algorithm, 0.0) + r.elapsed
        return costs

    def cost_by_component(self) -> Dict[str, float]:
        costs: Dict[str, float] = {}
        for r in self.records:
            costs[r.component] = costs.get(r.component, 0.0) + r.elapsed
        return costs

    def cost_per_metric(self, metric_labels: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Return time spent on metric computation, broken down by metric label.

        Looks for records with component == METRIC_COMPUTATION and metadata
        key "metric".
        """
        out: Dict[str, float] = {}
        for r in self.records:
            if r.component != CostComponent.METRIC_COMPUTATION:
                continue
            metric = r.metadata.get("metric", "unknown")
            if metric_labels and metric not in metric_labels:
                continue
            out[metric] = out.get(metric, 0.0) + r.elapsed
        return out

    def cumulative_time(self) -> List[Tuple[float, float]]:
        """Return [(wall_clock, cumulative_elapsed), ...] for plotting."""
        if not self.records:
            return []
        sorted_recs = sorted(self.records, key=lambda r: r.end_time)
        origin = sorted_recs[0].start_time
        cum = 0.0
        points: List[Tuple[float, float]] = []
        for r in sorted_recs:
            cum += r.elapsed
            points.append((r.end_time - origin, cum))
        return points

    def is_over_budget(self) -> bool:
        if self.budget_seconds is None:
            return False
        return self.total_time > self.budget_seconds

    def summary(self) -> Dict[str, Any]:
        return {
            "total_time_s": self.total_time,
            "total_tokens": self.total_tokens,
            "total_sequences": self.total_sequences,
            "peak_memory_bytes": self.peak_memory,
            "cost_per_token_s": self.cost_per_token(),
            "cost_per_sequence_s": self.cost_per_sequence(),
            "budget_seconds": self.budget_seconds,
            "budget_remaining_s": self.budget_remaining,
            "budget_fraction_used": self.budget_fraction_used,
            "by_algorithm": self.cost_by_algorithm(),
            "by_component": self.cost_by_component(),
            "num_records": len(self.records),
        }


# ---------------------------------------------------------------------------
# ScalingAnalysis
# ---------------------------------------------------------------------------

@dataclass
class ScalingPoint:
    param_name: str
    param_value: float
    predicted_time: float
    predicted_flops: float
    predicted_memory: float
    actual_time: Optional[float] = None


class ScalingAnalysis:
    """Analyse how cost scales with key parameters."""

    def __init__(self, cost_model: CostModel) -> None:
        self.cost_model = cost_model

    def _sweep(
        self,
        algorithm: str,
        param: str,
        values: Sequence[float],
        base_config: Optional[AlgorithmConfig] = None,
    ) -> List[ScalingPoint]:
        cfg = base_config or AlgorithmConfig(algorithm=algorithm)
        points: List[ScalingPoint] = []
        for v in values:
            c = AlgorithmConfig(
                algorithm=algorithm,
                num_sequences=int(v) if param == "num_sequences" else cfg.num_sequences,
                max_tokens=int(v) if param == "max_tokens" else cfg.max_tokens,
                vocab_size=int(v) if param == "vocab_size" else cfg.vocab_size,
                top_k=int(v) if param == "top_k" else cfg.top_k,
                num_beams=int(v) if param == "num_beams" else cfg.num_beams,
                num_beam_groups=int(v) if param == "num_beam_groups" else cfg.num_beam_groups,
                num_particles=int(v) if param == "num_particles" else cfg.num_particles,
                embed_dim=int(v) if param == "embed_dim" else cfg.embed_dim,
                qd_cells=int(v) if param == "qd_cells" else cfg.qd_cells,
            )
            pred = self.cost_model.predict_all(algorithm, c)
            points.append(ScalingPoint(
                param_name=param,
                param_value=v,
                predicted_time=pred["time_s"],
                predicted_flops=pred["flops"],
                predicted_memory=pred["memory_bytes"],
            ))
        return points

    def sweep_num_sequences(
        self,
        algorithm: str,
        values: Optional[Sequence[int]] = None,
        base_config: Optional[AlgorithmConfig] = None,
    ) -> List[ScalingPoint]:
        if values is None:
            values = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        return self._sweep(algorithm, "num_sequences", [float(v) for v in values], base_config)

    def sweep_max_tokens(
        self,
        algorithm: str,
        values: Optional[Sequence[int]] = None,
        base_config: Optional[AlgorithmConfig] = None,
    ) -> List[ScalingPoint]:
        if values is None:
            values = [16, 32, 64, 128, 256, 512, 1024]
        return self._sweep(algorithm, "max_tokens", [float(v) for v in values], base_config)

    def sweep_vocab_size(
        self,
        algorithm: str,
        values: Optional[Sequence[int]] = None,
        base_config: Optional[AlgorithmConfig] = None,
    ) -> List[ScalingPoint]:
        if values is None:
            values = [1000, 5000, 10000, 30000, 50000, 100000]
        return self._sweep(algorithm, "vocab_size", [float(v) for v in values], base_config)

    def fit_power_law(
        self, points: List[ScalingPoint]
    ) -> Tuple[float, float, float]:
        """
        Fit time = a * x^b via log-linear regression.

        Returns (a, b, r_squared).
        """
        xs = np.array([p.param_value for p in points], dtype=np.float64)
        ys = np.array([p.predicted_time for p in points], dtype=np.float64)
        mask = (xs > 0) & (ys > 0)
        xs, ys = xs[mask], ys[mask]
        if len(xs) < 2:
            return (0.0, 0.0, 0.0)
        log_x = np.log(xs)
        log_y = np.log(ys)
        A = np.vstack([log_x, np.ones_like(log_x)]).T
        result = np.linalg.lstsq(A, log_y, rcond=None)
        b, log_a = result[0]
        a = float(np.exp(log_a))
        ss_res = float(np.sum((log_y - A @ result[0]) ** 2))
        ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-15)
        return (a, float(b), r2)

    def empirical_vs_theoretical(
        self,
        algorithm: str,
        param: str,
        values: Sequence[float],
        measured_times: Sequence[float],
        base_config: Optional[AlgorithmConfig] = None,
    ) -> Dict[str, Any]:
        """
        Compare empirical measurements against theoretical predictions.

        Returns regression stats and per-point comparison.
        """
        predicted_pts = self._sweep(algorithm, param, values, base_config)
        pred_times = np.array([p.predicted_time for p in predicted_pts])
        act_times = np.array(measured_times, dtype=np.float64)

        abs_errors = np.abs(pred_times - act_times)
        rel_errors = abs_errors / np.maximum(act_times, 1e-15)

        a_pred, b_pred, r2_pred = self.fit_power_law(predicted_pts)

        emp_points = [
            ScalingPoint(param_name=param, param_value=v,
                         predicted_time=t, predicted_flops=0, predicted_memory=0,
                         actual_time=t)
            for v, t in zip(values, measured_times)
        ]
        a_emp, b_emp, r2_emp = self.fit_power_law(emp_points)

        return {
            "param": param,
            "values": list(values),
            "predicted_times": pred_times.tolist(),
            "actual_times": act_times.tolist(),
            "abs_errors": abs_errors.tolist(),
            "rel_errors": rel_errors.tolist(),
            "mean_rel_error": float(np.mean(rel_errors)),
            "max_rel_error": float(np.max(rel_errors)),
            "predicted_power_law": {"a": a_pred, "b": b_pred, "r2": r2_pred},
            "empirical_power_law": {"a": a_emp, "b": b_emp, "r2": r2_emp},
            "exponent_diff": abs(b_pred - b_emp),
        }

    def identify_bottleneck(
        self,
        algorithm: str,
        config: Optional[AlgorithmConfig] = None,
    ) -> Dict[str, Any]:
        """
        Determine which parameter is the primary cost driver.

        Sweeps each relevant parameter by 2x and measures predicted cost
        change.
        """
        cfg = config or AlgorithmConfig(algorithm=algorithm)
        base_time = self.cost_model.predict_time(algorithm, cfg)
        base_mem = self.cost_model.predict_memory(algorithm, cfg)

        sensitivities: Dict[str, Dict[str, float]] = {}
        params_to_sweep = {
            "num_sequences": cfg.num_sequences,
            "max_tokens": cfg.max_tokens,
            "vocab_size": cfg.vocab_size,
        }
        if algorithm == AlgorithmName.TOP_K:
            params_to_sweep["top_k"] = cfg.top_k
        elif algorithm == AlgorithmName.DIVERSE_BEAM:
            params_to_sweep["num_beams"] = cfg.num_beams
            params_to_sweep["num_beam_groups"] = cfg.num_beam_groups
        elif algorithm == AlgorithmName.SVD:
            params_to_sweep["num_particles"] = cfg.num_particles
            params_to_sweep["embed_dim"] = cfg.embed_dim
        elif algorithm == AlgorithmName.QD_BS:
            params_to_sweep["num_beams"] = cfg.num_beams
            params_to_sweep["qd_cells"] = cfg.qd_cells

        for pname, pval in params_to_sweep.items():
            doubled = AlgorithmConfig(
                algorithm=algorithm,
                num_sequences=cfg.num_sequences * 2 if pname == "num_sequences" else cfg.num_sequences,
                max_tokens=cfg.max_tokens * 2 if pname == "max_tokens" else cfg.max_tokens,
                vocab_size=cfg.vocab_size * 2 if pname == "vocab_size" else cfg.vocab_size,
                top_k=cfg.top_k * 2 if pname == "top_k" else cfg.top_k,
                num_beams=cfg.num_beams * 2 if pname == "num_beams" else cfg.num_beams,
                num_beam_groups=cfg.num_beam_groups * 2 if pname == "num_beam_groups" else cfg.num_beam_groups,
                num_particles=cfg.num_particles * 2 if pname == "num_particles" else cfg.num_particles,
                embed_dim=cfg.embed_dim * 2 if pname == "embed_dim" else cfg.embed_dim,
                qd_cells=cfg.qd_cells * 2 if pname == "qd_cells" else cfg.qd_cells,
            )
            new_time = self.cost_model.predict_time(algorithm, doubled)
            new_mem = self.cost_model.predict_memory(algorithm, doubled)

            time_ratio = new_time / max(base_time, 1e-15)
            mem_ratio = new_mem / max(base_mem, 1e-15)

            # Estimate scaling exponent: if ratio=2^k when param doubles, k is exponent
            time_exponent = math.log2(max(time_ratio, 1e-15))
            mem_exponent = math.log2(max(mem_ratio, 1e-15))

            sensitivities[pname] = {
                "base_value": pval,
                "time_ratio_2x": time_ratio,
                "mem_ratio_2x": mem_ratio,
                "time_exponent": time_exponent,
                "mem_exponent": mem_exponent,
            }

        # Identify the top bottleneck
        bottleneck_time = max(
            sensitivities.items(), key=lambda kv: kv[1]["time_exponent"]
        )
        bottleneck_mem = max(
            sensitivities.items(), key=lambda kv: kv[1]["mem_exponent"]
        )

        return {
            "algorithm": algorithm,
            "base_time_s": base_time,
            "base_memory_bytes": base_mem,
            "sensitivities": sensitivities,
            "time_bottleneck": bottleneck_time[0],
            "memory_bottleneck": bottleneck_mem[0],
        }

    def scaling_table(
        self,
        algorithm: str,
        param: str,
        values: Sequence[float],
        base_config: Optional[AlgorithmConfig] = None,
    ) -> List[Dict[str, Any]]:
        points = self._sweep(algorithm, param, values, base_config)
        rows: List[Dict[str, Any]] = []
        for p in points:
            rows.append({
                "param": param,
                "value": p.param_value,
                "time_s": p.predicted_time,
                "flops": p.predicted_flops,
                "memory_bytes": p.predicted_memory,
            })
        return rows


# ---------------------------------------------------------------------------
# CostReport
# ---------------------------------------------------------------------------

class CostReport:
    """Generate human-readable cost reports."""

    def __init__(
        self,
        cost_model: CostModel,
        tracker: Optional[CostTracker] = None,
    ) -> None:
        self.cost_model = cost_model
        self.tracker = tracker

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if seconds < 1e-3:
            return f"{seconds * 1e6:.1f} µs"
        if seconds < 1.0:
            return f"{seconds * 1e3:.2f} ms"
        if seconds < 60.0:
            return f"{seconds:.3f} s"
        minutes = int(seconds // 60)
        secs = seconds - minutes * 60
        return f"{minutes}m {secs:.1f}s"

    @staticmethod
    def _fmt_bytes(b: float) -> str:
        if b < 1024:
            return f"{b:.0f} B"
        if b < 1024 ** 2:
            return f"{b / 1024:.1f} KB"
        if b < 1024 ** 3:
            return f"{b / (1024 ** 2):.1f} MB"
        return f"{b / (1024 ** 3):.2f} GB"

    @staticmethod
    def _fmt_flops(f: float) -> str:
        if f < 1e6:
            return f"{f:.0f}"
        if f < 1e9:
            return f"{f / 1e6:.2f} MFLOP"
        if f < 1e12:
            return f"{f / 1e9:.2f} GFLOP"
        return f"{f / 1e12:.2f} TFLOP"

    def comparison_table(
        self,
        configs: Dict[str, AlgorithmConfig],
        hardware: Optional[HardwareProfile] = None,
    ) -> str:
        """Produce a plaintext table comparing algorithms."""
        preds = self.cost_model.compare_algorithms(configs, hardware)

        col_name = 14
        col_time = 14
        col_mem = 14
        col_flops = 16
        header = (
            f"{'Algorithm':<{col_name}}"
            f"{'Time':<{col_time}}"
            f"{'Memory':<{col_mem}}"
            f"{'FLOPs':<{col_flops}}"
        )
        sep = "-" * len(header)
        lines = [header, sep]

        for name, pred in sorted(preds.items()):
            lines.append(
                f"{name:<{col_name}}"
                f"{self._fmt_time(pred['time_s']):<{col_time}}"
                f"{self._fmt_bytes(pred['memory_bytes']):<{col_mem}}"
                f"{self._fmt_flops(pred['flops']):<{col_flops}}"
            )
        lines.append(sep)
        return "\n".join(lines)

    def breakdown_table(self) -> str:
        """Component-level breakdown from the tracker."""
        if self.tracker is None or not self.tracker.records:
            return "(no tracking data)"

        by_comp = self.tracker.cost_by_component()
        total = self.tracker.total_time

        col_comp = 22
        col_time = 14
        col_pct = 10
        header = f"{'Component':<{col_comp}}{'Time':<{col_time}}{'%':<{col_pct}}"
        sep = "-" * len(header)
        lines = [header, sep]

        for comp, t in sorted(by_comp.items(), key=lambda kv: -kv[1]):
            pct = t / max(total, 1e-15) * 100
            lines.append(
                f"{comp:<{col_comp}}"
                f"{self._fmt_time(t):<{col_time}}"
                f"{pct:<{col_pct}.1f}"
            )
        lines.append(sep)
        lines.append(f"{'TOTAL':<{col_comp}}{self._fmt_time(total):<{col_time}}{'100.0':<{col_pct}}")
        return "\n".join(lines)

    def algorithm_breakdown(self) -> str:
        """Per-algorithm breakdown from the tracker."""
        if self.tracker is None or not self.tracker.records:
            return "(no tracking data)"

        by_algo = self.tracker.cost_by_algorithm()
        total = self.tracker.total_time

        col_algo = 18
        col_time = 14
        col_pct = 10
        col_tok = 12
        col_seq = 12
        header = (
            f"{'Algorithm':<{col_algo}}"
            f"{'Time':<{col_time}}"
            f"{'%':<{col_pct}}"
            f"{'Tokens':<{col_tok}}"
            f"{'Seqs':<{col_seq}}"
        )
        sep = "-" * len(header)
        lines = [header, sep]

        algo_tokens: Dict[str, int] = {}
        algo_seqs: Dict[str, int] = {}
        for r in self.tracker.records:
            algo_tokens[r.algorithm] = algo_tokens.get(r.algorithm, 0) + r.tokens_generated
            algo_seqs[r.algorithm] = algo_seqs.get(r.algorithm, 0) + r.sequences_generated

        for algo, t in sorted(by_algo.items(), key=lambda kv: -kv[1]):
            pct = t / max(total, 1e-15) * 100
            lines.append(
                f"{algo:<{col_algo}}"
                f"{self._fmt_time(t):<{col_time}}"
                f"{pct:<{col_pct}.1f}"
                f"{algo_tokens.get(algo, 0):<{col_tok}}"
                f"{algo_seqs.get(algo, 0):<{col_seq}}"
            )
        lines.append(sep)
        return "\n".join(lines)

    def three_d_analysis(
        self,
        configs: Dict[str, AlgorithmConfig],
        diversity_estimator: Optional[Callable[[AlgorithmConfig], float]] = None,
        quality_estimator: Optional[Callable[[AlgorithmConfig], float]] = None,
    ) -> str:
        """
        Print a time / diversity / quality analysis for each config.
        """
        div_fn = diversity_estimator or CostOptimizer._default_diversity
        qual_fn = quality_estimator or CostOptimizer._default_quality

        col_name = 16
        col_time = 14
        col_div = 12
        col_qual = 12
        col_eff = 14
        header = (
            f"{'Algorithm':<{col_name}}"
            f"{'Time':<{col_time}}"
            f"{'Diversity':<{col_div}}"
            f"{'Quality':<{col_qual}}"
            f"{'Efficiency':<{col_eff}}"
        )
        sep = "-" * len(header)
        lines = [header, sep]

        for name, cfg in sorted(configs.items()):
            pred = self.cost_model.predict_all(cfg.algorithm, cfg)
            d = div_fn(cfg)
            q = qual_fn(cfg)
            eff = (d + q) / max(pred["time_s"], 1e-15)
            lines.append(
                f"{name:<{col_name}}"
                f"{self._fmt_time(pred['time_s']):<{col_time}}"
                f"{d:<{col_div}.3f}"
                f"{q:<{col_qual}.3f}"
                f"{eff:<{col_eff}.2f}"
            )
        lines.append(sep)
        return "\n".join(lines)

    def budget_report(self) -> str:
        """Report on budget usage from the tracker."""
        if self.tracker is None:
            return "(no tracker)"

        s = self.tracker.summary()
        lines = [
            "=== Budget Report ===",
            f"Total time used   : {self._fmt_time(s['total_time_s'])}",
            f"Total tokens      : {s['total_tokens']}",
            f"Total sequences   : {s['total_sequences']}",
            f"Peak memory       : {self._fmt_bytes(s['peak_memory_bytes'])}",
            f"Cost / token      : {self._fmt_time(s['cost_per_token_s'])}",
            f"Cost / sequence   : {self._fmt_time(s['cost_per_sequence_s'])}",
        ]
        if s["budget_seconds"] is not None:
            lines.append(f"Budget            : {self._fmt_time(s['budget_seconds'])}")
            lines.append(f"Remaining         : {self._fmt_time(s['budget_remaining_s'])}")
            lines.append(f"Fraction used     : {s['budget_fraction_used']:.1%}")
        lines.append("=" * 30)
        return "\n".join(lines)

    def full_report(
        self,
        configs: Optional[Dict[str, AlgorithmConfig]] = None,
        hardware: Optional[HardwareProfile] = None,
    ) -> str:
        """Produce a comprehensive report combining all sub-reports."""
        sections: List[str] = []

        if hardware:
            sections.append("=== Hardware ===\n" + hardware.summary())
        else:
            sections.append("=== Hardware ===\n" + self.cost_model.hardware.summary())

        if configs:
            sections.append("=== Predicted Costs ===\n" + self.comparison_table(configs, hardware))
            sections.append("=== Time / Diversity / Quality ===\n" + self.three_d_analysis(configs))

        if self.tracker and self.tracker.records:
            sections.append("=== Actual Costs (by component) ===\n" + self.breakdown_table())
            sections.append("=== Actual Costs (by algorithm) ===\n" + self.algorithm_breakdown())
            sections.append(self.budget_report())

        return "\n\n".join(sections)

    def to_dict(
        self,
        configs: Optional[Dict[str, AlgorithmConfig]] = None,
        hardware: Optional[HardwareProfile] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "hardware": {
                "cpu_cores": self.cost_model.hardware.cpu_cores,
                "cpu_clock_ghz": self.cost_model.hardware.cpu_clock_ghz,
                "ram_gb": self.cost_model.hardware.ram_gb,
                "peak_cpu_flops": self.cost_model.hardware.peak_cpu_flops,
                "has_gpu": self.cost_model.hardware.has_gpu,
            }
        }
        if configs:
            result["predictions"] = self.cost_model.compare_algorithms(configs, hardware)
        if self.tracker:
            result["actual"] = self.tracker.summary()
        return result


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_default_configs(
    num_sequences: int = 10,
    max_tokens: int = 128,
    vocab_size: int = 50257,
) -> Dict[str, AlgorithmConfig]:
    """Return one AlgorithmConfig per algorithm with sensible defaults."""
    return {
        algo.value: AlgorithmConfig(
            algorithm=algo,
            num_sequences=num_sequences,
            max_tokens=max_tokens,
            vocab_size=vocab_size,
        )
        for algo in AlgorithmName
    }


def quick_cost_report(
    num_sequences: int = 10,
    max_tokens: int = 128,
    vocab_size: int = 50257,
) -> str:
    """One-liner to get a full predicted-cost comparison."""
    hw = HardwareProfile.detect()
    model = CostModel(hardware=hw)
    configs = create_default_configs(num_sequences, max_tokens, vocab_size)
    report = CostReport(cost_model=model)
    return report.full_report(configs=configs, hardware=hw)
