"""
ONNX Runtime inference backend for the Diversity Decoding Arena.

Provides ``ONNXLogitSource`` — an ONNX Runtime–based implementation of
:class:`LogitSource` — along with model management, quantization, profiling,
and KV-cache helpers.

Classes:
    ONNXConfig: Session configuration for ONNX Runtime.
    ONNXLogitSource: Logit source backed by an ONNX Runtime session.
    ONNXModelManager: Download, convert, validate, and optimise ONNX models.
    ONNXQuantizer: Dynamic and static quantization utilities.
    ONNXProfiler: Latency, throughput, and memory profiling.
    KVCacheONNX: Manages past_key_values for incremental decoding.

Functions:
    create_onnx_source: High-level factory for ONNXLogitSource.
    benchmark_onnx_source: Throughput / latency benchmark runner.
    validate_onnx_outputs: Compare ONNX outputs against a reference source.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.logit_source.base import LogitBatch, LogitSource, LogitSourceConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard optional imports
# ---------------------------------------------------------------------------

try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore[assignment]
    _ORT_AVAILABLE = False

try:
    import onnx

    _ONNX_AVAILABLE = True
except ImportError:
    onnx = None  # type: ignore[assignment]
    _ONNX_AVAILABLE = False

try:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )

    _ORT_QUANT_AVAILABLE = True
except ImportError:
    _ORT_QUANT_AVAILABLE = False

try:
    from onnxruntime.transformers.optimizer import optimize_model as _ort_optimize

    _ORT_OPTIM_AVAILABLE = True
except ImportError:
    _ORT_OPTIM_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer

    _TOKENIZERS_AVAILABLE = True
except ImportError:
    _TOKENIZERS_AVAILABLE = False


def _require_ort() -> None:
    if not _ORT_AVAILABLE:
        raise ImportError(
            "onnxruntime is required for ONNXLogitSource. "
            "Install with: pip install onnxruntime"
        )


def _require_onnx() -> None:
    if not _ONNX_AVAILABLE:
        raise ImportError(
            "onnx is required for model validation. "
            "Install with: pip install onnx"
        )


# ---------------------------------------------------------------------------
# Optimisation-level mapping
# ---------------------------------------------------------------------------

_OPT_LEVEL_MAP: Dict[str, int] = {
    "ORT_DISABLE_ALL": 0,
    "ORT_ENABLE_BASIC": 1,
    "ORT_ENABLE_EXTENDED": 2,
    "ORT_ENABLE_ALL": 99,
}


# ---------------------------------------------------------------------------
# ONNXConfig
# ---------------------------------------------------------------------------


@dataclass
class ONNXConfig:
    """Session configuration for ONNX Runtime inference.

    Attributes:
        model_path: Filesystem path to the ``.onnx`` model file.
        num_threads: Intra-op thread count (parallelism within a single op).
        inter_op_threads: Inter-op thread count (parallelism across ops).
        execution_providers: Ordered list of execution providers.
        graph_optimization_level: One of the ``ORT_*`` optimisation levels.
        enable_profiling: Whether to enable ORT profiling.
        quantization_type: Optional quantization mode (``"int8"`` or ``"int4"``).
        log_severity_level: ORT log severity (0=VERBOSE … 4=FATAL).
        memory_limit: Optional memory limit for the session (bytes).
    """

    model_path: str = ""
    num_threads: int = 4
    inter_op_threads: int = 1
    execution_providers: List[str] = field(
        default_factory=lambda: ["CPUExecutionProvider"]
    )
    graph_optimization_level: str = "ORT_ENABLE_ALL"
    enable_profiling: bool = False
    quantization_type: Optional[str] = None
    log_severity_level: int = 3
    memory_limit: Optional[int] = None

    def __post_init__(self) -> None:
        if self.graph_optimization_level not in _OPT_LEVEL_MAP:
            raise ValueError(
                f"graph_optimization_level must be one of {list(_OPT_LEVEL_MAP)}, "
                f"got {self.graph_optimization_level!r}"
            )
        if self.quantization_type is not None and self.quantization_type not in (
            "int8",
            "int4",
        ):
            raise ValueError(
                f"quantization_type must be None, 'int8', or 'int4', "
                f"got {self.quantization_type!r}"
            )
        if self.num_threads < 0:
            raise ValueError(f"num_threads must be >= 0, got {self.num_threads}")
        if self.inter_op_threads < 0:
            raise ValueError(
                f"inter_op_threads must be >= 0, got {self.inter_op_threads}"
            )
        if self.log_severity_level not in range(5):
            raise ValueError(
                f"log_severity_level must be 0–4, got {self.log_severity_level}"
            )
        if self.memory_limit is not None and self.memory_limit <= 0:
            raise ValueError(
                f"memory_limit must be positive or None, got {self.memory_limit}"
            )

    def copy(self, **overrides: Any) -> "ONNXConfig":
        """Return a shallow copy, optionally overriding fields."""
        import dataclasses

        return dataclasses.replace(self, **overrides)


# ---------------------------------------------------------------------------
# KVCacheONNX
# ---------------------------------------------------------------------------


class KVCacheONNX:
    """Manages ``past_key_values`` arrays for ONNX models that support
    incremental (KV-cache) decoding.

    The cache stores NumPy arrays keyed by ``(layer, k_or_v)`` — e.g.
    ``"past_key_values.0.key"``, ``"past_key_values.0.value"``.

    Typical workflow::

        cache = KVCacheONNX(num_layers=12, num_heads=12, head_dim=64)
        outputs = session.run(None, {**inputs, **cache.get_past()})
        cache.update(new_keys, new_values)
    """

    def __init__(
        self,
        num_layers: int = 12,
        num_heads: int = 12,
        head_dim: int = 64,
        batch_size: int = 1,
        dtype: np.dtype = np.float32,
        max_length: int = 1024,
    ) -> None:
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._batch_size = batch_size
        self._dtype = np.dtype(dtype)
        self._max_length = max_length

        # keys[layer] and values[layer] have shape [batch, heads, seq, head_dim]
        self._keys: List[np.ndarray] = []
        self._values: List[np.ndarray] = []

        self._seq_len: int = 0
        self.clear()

    # -- public API ---------------------------------------------------------

    def update(
        self,
        new_keys: List[np.ndarray],
        new_values: List[np.ndarray],
    ) -> None:
        """Append new key/value slices for every layer.

        Args:
            new_keys: Per-layer arrays of shape ``[batch, heads, new_seq, head_dim]``.
            new_values: Same shape as *new_keys*.

        Raises:
            ValueError: If the number of layers does not match.
        """
        if len(new_keys) != self._num_layers:
            raise ValueError(
                f"Expected {self._num_layers} layers, got {len(new_keys)} key arrays"
            )
        if len(new_values) != self._num_layers:
            raise ValueError(
                f"Expected {self._num_layers} layers, got {len(new_values)} value arrays"
            )

        added_seq = new_keys[0].shape[2] if new_keys[0].ndim == 4 else 1

        for layer_idx in range(self._num_layers):
            k = np.asarray(new_keys[layer_idx], dtype=self._dtype)
            v = np.asarray(new_values[layer_idx], dtype=self._dtype)

            if self._keys[layer_idx].shape[2] == 0:
                self._keys[layer_idx] = k
                self._values[layer_idx] = v
            else:
                self._keys[layer_idx] = np.concatenate(
                    [self._keys[layer_idx], k], axis=2
                )
                self._values[layer_idx] = np.concatenate(
                    [self._values[layer_idx], v], axis=2
                )

        self._seq_len += added_seq

        # Auto-trim if we exceed max_length
        if self._seq_len > self._max_length:
            self.trim(self._max_length)

    def get_past(self) -> Dict[str, np.ndarray]:
        """Return the cache as a flat dict suitable for ONNX session input.

        Keys follow the naming convention used by Optimum-exported models:
        ``past_key_values.{layer}.key`` and ``past_key_values.{layer}.value``.
        """
        result: Dict[str, np.ndarray] = {}
        for layer_idx in range(self._num_layers):
            result[f"past_key_values.{layer_idx}.key"] = self._keys[layer_idx]
            result[f"past_key_values.{layer_idx}.value"] = self._values[layer_idx]
        return result

    def trim(self, max_length: int) -> None:
        """Trim the cache to at most *max_length* positions (keeping the most
        recent ones)."""
        if max_length <= 0:
            self.clear()
            return
        for layer_idx in range(self._num_layers):
            current_len = self._keys[layer_idx].shape[2]
            if current_len > max_length:
                start = current_len - max_length
                self._keys[layer_idx] = self._keys[layer_idx][:, :, start:, :]
                self._values[layer_idx] = self._values[layer_idx][:, :, start:, :]
        self._seq_len = min(self._seq_len, max_length)

    def clear(self) -> None:
        """Reset the cache to empty."""
        empty_shape = (self._batch_size, self._num_heads, 0, self._head_dim)
        self._keys = [
            np.zeros(empty_shape, dtype=self._dtype)
            for _ in range(self._num_layers)
        ]
        self._values = [
            np.zeros(empty_shape, dtype=self._dtype)
            for _ in range(self._num_layers)
        ]
        self._seq_len = 0

    # -- introspection ------------------------------------------------------

    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        return self._seq_len

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def is_empty(self) -> bool:
        return self._seq_len == 0

    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        total = 0
        for layer_idx in range(self._num_layers):
            total += self._keys[layer_idx].nbytes
            total += self._values[layer_idx].nbytes
        return total

    def __repr__(self) -> str:
        return (
            f"KVCacheONNX(layers={self._num_layers}, heads={self._num_heads}, "
            f"head_dim={self._head_dim}, seq_len={self._seq_len}, "
            f"mem={self.memory_bytes() / 1024:.1f} KB)"
        )


# ---------------------------------------------------------------------------
# ONNXLogitSource
# ---------------------------------------------------------------------------


class ONNXLogitSource(LogitSource):
    """Logit source backed by an ONNX Runtime :class:`InferenceSession`.

    Handles session creation, input preparation, inference, output
    post-processing, optional KV-cache management, and profiling.
    """

    def __init__(
        self,
        config: ONNXConfig,
        logit_source_config: Optional[LogitSourceConfig] = None,
    ) -> None:
        _require_ort()

        if logit_source_config is None:
            logit_source_config = LogitSourceConfig()

        super().__init__(logit_source_config)

        self._onnx_config = config
        self._session: Optional[ort.InferenceSession] = None
        self._session_options: Optional[ort.SessionOptions] = None
        self._tokenizer: Optional[Any] = None
        self._kv_cache: Optional[KVCacheONNX] = None
        self._profiling_started: bool = False
        self._profile_file: Optional[str] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._supports_kv_cache: bool = False

        if config.model_path and os.path.isfile(config.model_path):
            self._create_session()

        # Attempt to load tokenizer
        self._load_tokenizer()

    # -- session management -------------------------------------------------

    def _configure_session_options(self) -> "ort.SessionOptions":
        """Build an ``ort.SessionOptions`` object from :attr:`_onnx_config`."""
        opts = ort.SessionOptions()

        # Thread configuration
        opts.intra_op_num_threads = self._onnx_config.num_threads
        opts.inter_op_num_threads = self._onnx_config.inter_op_threads

        # Graph optimisation
        opt_int = _OPT_LEVEL_MAP.get(
            self._onnx_config.graph_optimization_level, 99
        )
        opts.graph_optimization_level = ort.GraphOptimizationLevel(opt_int)

        # Profiling
        if self._onnx_config.enable_profiling:
            opts.enable_profiling = True

        # Logging
        opts.log_severity_level = self._onnx_config.log_severity_level

        # Memory limit via arena configuration (if supported)
        if self._onnx_config.memory_limit is not None:
            try:
                opts.add_session_config_entry(
                    "arena_extend_strategy", "kSameAsRequested"
                )
            except Exception:
                logger.debug("Memory limit configuration not supported in this ORT build")

        return opts

    def _create_session(self) -> None:
        """Create the ONNX Runtime inference session."""
        model_path = self._onnx_config.model_path
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        logger.info("Creating ONNX session from %s", model_path)
        t0 = time.monotonic()

        self._session_options = self._configure_session_options()

        try:
            self._session = ort.InferenceSession(
                model_path,
                sess_options=self._session_options,
                providers=self._onnx_config.execution_providers,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create ONNX session from {model_path}: {exc}"
            ) from exc

        # Cache input/output names
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

        # Detect KV-cache support
        self._supports_kv_cache = any(
            "past" in name.lower() for name in self._input_names
        )

        elapsed = time.monotonic() - t0
        logger.info(
            "ONNX session ready in %.2fs — inputs=%s, outputs=%s, kv_cache=%s",
            elapsed,
            self._input_names,
            self._output_names,
            self._supports_kv_cache,
        )

    def _load_tokenizer(self) -> None:
        """Attempt to load a HuggingFace tokenizer for encode/decode."""
        if not _TOKENIZERS_AVAILABLE:
            logger.debug("transformers not available; encode/decode will require manual token IDs")
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._config.model_name, use_fast=True
            )
            logger.debug("Loaded tokenizer for %s", self._config.model_name)
        except Exception as exc:
            logger.debug("Could not load tokenizer for %s: %s", self._config.model_name, exc)
            self._tokenizer = None

    # -- input preparation --------------------------------------------------

    def _prepare_inputs(
        self,
        input_ids: List[int],
        use_cache: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Prepare the feed dict for a single sequence.

        Args:
            input_ids: Token-ID sequence.
            use_cache: If True and the model supports KV-cache, include
                ``past_key_values`` in the feed dict.

        Returns:
            Dictionary mapping ONNX input names to NumPy arrays.
        """
        ids_array = np.array([input_ids], dtype=np.int64)
        attention_mask = np.ones_like(ids_array, dtype=np.int64)

        feed: Dict[str, np.ndarray] = {}

        # Map available input names
        name_lower_map = {n.lower(): n for n in self._input_names}

        if "input_ids" in name_lower_map:
            feed[name_lower_map["input_ids"]] = ids_array
        elif self._input_names:
            feed[self._input_names[0]] = ids_array

        if "attention_mask" in name_lower_map:
            feed[name_lower_map["attention_mask"]] = attention_mask

        # Position IDs (some models require this)
        if "position_ids" in name_lower_map:
            seq_len = ids_array.shape[1]
            if use_cache and self._kv_cache is not None and not self._kv_cache.is_empty:
                pos_start = self._kv_cache.seq_len
                position_ids = np.arange(pos_start, pos_start + seq_len, dtype=np.int64)
            else:
                position_ids = np.arange(seq_len, dtype=np.int64)
            feed[name_lower_map["position_ids"]] = position_ids.reshape(1, -1)

        # KV-cache inputs
        if use_cache and self._supports_kv_cache and self._kv_cache is not None:
            past = self._kv_cache.get_past()
            for k, v in past.items():
                if k in self._input_names:
                    feed[k] = v

            # Extend attention mask to include cached positions
            if "attention_mask" in name_lower_map and not self._kv_cache.is_empty:
                cache_len = self._kv_cache.seq_len
                cache_mask = np.ones(
                    (1, cache_len), dtype=np.int64
                )
                feed[name_lower_map["attention_mask"]] = np.concatenate(
                    [cache_mask, attention_mask], axis=1
                )

        # Token type IDs (BERT-style models)
        if "token_type_ids" in name_lower_map:
            feed[name_lower_map["token_type_ids"]] = np.zeros_like(
                ids_array, dtype=np.int64
            )

        return feed

    def _prepare_batch_inputs(
        self,
        input_ids_batch: List[List[int]],
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Prepare a padded batch of inputs.

        Args:
            input_ids_batch: List of *B* token-ID sequences of varying length.

        Returns:
            Tuple of (feed dict, attention_mask) where sequences are
            right-padded to the maximum length in the batch.
        """
        batch_size = len(input_ids_batch)
        max_len = max(len(ids) for ids in input_ids_batch)

        # Pad sequences
        padded_ids = np.zeros((batch_size, max_len), dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)

        for i, ids in enumerate(input_ids_batch):
            seq_len = len(ids)
            padded_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1

        feed: Dict[str, np.ndarray] = {}

        name_lower_map = {n.lower(): n for n in self._input_names}

        if "input_ids" in name_lower_map:
            feed[name_lower_map["input_ids"]] = padded_ids
        elif self._input_names:
            feed[self._input_names[0]] = padded_ids

        if "attention_mask" in name_lower_map:
            feed[name_lower_map["attention_mask"]] = attention_mask

        if "position_ids" in name_lower_map:
            position_ids = np.tile(
                np.arange(max_len, dtype=np.int64), (batch_size, 1)
            )
            # Zero out positions that are padding
            position_ids = position_ids * attention_mask
            feed[name_lower_map["position_ids"]] = position_ids

        if "token_type_ids" in name_lower_map:
            feed[name_lower_map["token_type_ids"]] = np.zeros_like(
                padded_ids, dtype=np.int64
            )

        return feed, attention_mask

    # -- inference ----------------------------------------------------------

    def _run_inference(
        self,
        inputs: Dict[str, np.ndarray],
        output_names: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """Execute the ONNX Runtime session.

        Args:
            inputs: Feed dict mapping input names to arrays.
            output_names: Which outputs to request (``None`` = all).

        Returns:
            List of output arrays.

        Raises:
            RuntimeError: If the session is not initialised or inference fails.
        """
        if self._session is None:
            raise RuntimeError(
                "ONNX session not initialised. Call _create_session() first or "
                "provide a valid model_path in ONNXConfig."
            )

        if output_names is None:
            output_names = self._output_names

        try:
            outputs = self._session.run(output_names, inputs)
        except Exception as exc:
            # Provide a helpful error with input shapes
            shape_info = {k: v.shape for k, v in inputs.items()}
            raise RuntimeError(
                f"ONNX inference failed. Input shapes: {shape_info}. "
                f"Error: {exc}"
            ) from exc

        return outputs

    def _postprocess_outputs(
        self,
        outputs: List[np.ndarray],
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extract logits from ONNX model outputs.

        Most transformer models output logits as the first output tensor with
        shape ``[batch, seq, vocab]``.  This method handles extraction and
        optional masking.

        Args:
            outputs: Raw list of ONNX output arrays.
            attention_mask: Optional mask for locating the last real token
                per sequence (used in batch mode).

        Returns:
            Logits array.  Shape depends on context:
            - Full sequence:  ``[seq, vocab]`` (single) or ``[batch, seq, vocab]``.
            - Last-token only: ``[batch, vocab]`` when *attention_mask* is given.
        """
        if not outputs:
            raise RuntimeError("ONNX session returned no outputs")

        logits = outputs[0]  # [batch, seq, vocab] or [batch, vocab]

        if logits.ndim == 3 and attention_mask is not None:
            # Extract last real token per sequence
            batch_size = logits.shape[0]
            seq_lengths = attention_mask.sum(axis=1).astype(int)  # [batch]
            last_logits = np.empty(
                (batch_size, logits.shape[2]), dtype=logits.dtype
            )
            for i in range(batch_size):
                last_idx = max(seq_lengths[i] - 1, 0)
                last_logits[i] = logits[i, last_idx]
            return last_logits

        return logits

    def _extract_kv_cache_from_outputs(
        self,
        outputs: List[np.ndarray],
    ) -> None:
        """If the model returns ``present_key_values``, update the KV cache."""
        if not self._supports_kv_cache or self._kv_cache is None:
            return

        # Identify which outputs are KV-cache tensors
        kv_outputs: Dict[int, np.ndarray] = {}
        for idx, name in enumerate(self._output_names):
            if "present" in name.lower() or "past_key_values" in name.lower():
                if idx < len(outputs):
                    kv_outputs[idx] = outputs[idx]

        if not kv_outputs:
            return

        # Group into keys and values by layer
        sorted_indices = sorted(kv_outputs.keys())
        new_keys: List[np.ndarray] = []
        new_values: List[np.ndarray] = []

        for i in range(0, len(sorted_indices), 2):
            if i + 1 < len(sorted_indices):
                new_keys.append(kv_outputs[sorted_indices[i]])
                new_values.append(kv_outputs[sorted_indices[i + 1]])

        if new_keys and new_values:
            # Replace cache entirely — ONNX models with KV-cache typically
            # return the full present state, not just the delta.
            self._kv_cache.clear()
            self._kv_cache.update(new_keys, new_values)

    # -- LogitSource interface ----------------------------------------------

    def get_logits(self, input_ids: List[int]) -> np.ndarray:
        """Return logits for every position in *input_ids*.

        Returns:
            Array of shape ``[S, vocab_size]``.
        """
        self._check_closed()
        if not input_ids:
            raise ValueError("input_ids must be non-empty")

        inputs = self._prepare_inputs(input_ids, use_cache=False)
        outputs = self._run_inference(inputs)
        logits = self._postprocess_outputs(outputs)

        # Ensure shape is [S, vocab]
        if logits.ndim == 3:
            logits = logits[0]  # remove batch dim → [S, vocab]
        elif logits.ndim == 1:
            logits = logits.reshape(1, -1)

        return logits.astype(self._config.numpy_dtype)

    def get_logits_batch(
        self,
        input_ids_batch: List[List[int]],
    ) -> LogitBatch:
        """Batched logit inference.

        Returns:
            :class:`LogitBatch` with logits of shape ``[B, vocab_size]``
            (last-position logits for each sequence).
        """
        self._check_closed()
        if not input_ids_batch:
            raise ValueError("input_ids_batch must be non-empty")

        t0 = time.monotonic()

        feed, attention_mask = self._prepare_batch_inputs(input_ids_batch)
        outputs = self._run_inference(feed)
        logits = self._postprocess_outputs(outputs, attention_mask=attention_mask)

        # Ensure [B, vocab]
        if logits.ndim == 3:
            # Extract last-token logits using attention_mask
            batch_size = logits.shape[0]
            seq_lengths = attention_mask.sum(axis=1).astype(int)
            last_logits = np.empty(
                (batch_size, logits.shape[2]), dtype=logits.dtype
            )
            for i in range(batch_size):
                last_idx = max(seq_lengths[i] - 1, 0)
                last_logits[i] = logits[i, last_idx]
            logits = last_logits

        latency = time.monotonic() - t0
        total_tokens = sum(len(ids) for ids in input_ids_batch)
        self._stats.record_call(latency, n_tokens=total_tokens)

        return LogitBatch(
            logits=logits.astype(self._config.numpy_dtype),
            token_ids=input_ids_batch,
            attention_mask=attention_mask,
            metadata={
                "latency_s": latency,
                "batch_size": len(input_ids_batch),
                "total_tokens": total_tokens,
                "backend": "onnxruntime",
            },
        )

    def get_next_token_logits(self, prefix: List[int]) -> np.ndarray:
        """Efficient single next-token prediction.

        If KV-cache is enabled and populated, only the last token is fed to
        the model.  Otherwise falls back to full-sequence inference.

        Returns:
            1-D array of shape ``[vocab_size]``.
        """
        self._check_closed()
        if not prefix:
            raise ValueError("prefix must be non-empty")

        if len(prefix) > self._config.max_seq_len:
            warnings.warn(
                f"prefix length {len(prefix)} exceeds max_seq_len "
                f"{self._config.max_seq_len}; truncating from the left"
            )
            prefix = prefix[-self._config.max_seq_len :]

        t0 = time.monotonic()

        use_cache = (
            self._config.use_kv_cache
            and self._supports_kv_cache
            and self._kv_cache is not None
            and not self._kv_cache.is_empty
        )

        if use_cache:
            # Feed only the new token(s)
            new_token = prefix[-1:]
            inputs = self._prepare_inputs(new_token, use_cache=True)
        else:
            inputs = self._prepare_inputs(prefix, use_cache=False)

        outputs = self._run_inference(inputs)

        # Update KV cache if applicable
        if self._config.use_kv_cache and self._supports_kv_cache:
            if self._kv_cache is None:
                self._init_kv_cache()
            self._extract_kv_cache_from_outputs(outputs)

        logits = self._postprocess_outputs(outputs)

        # Extract last position
        if logits.ndim == 3:
            result = logits[0, -1, :]
        elif logits.ndim == 2:
            result = logits[0] if logits.shape[0] == 1 else logits[-1]
        else:
            result = logits

        latency = time.monotonic() - t0
        self._stats.record_call(latency, n_tokens=len(prefix))

        return result.astype(self._config.numpy_dtype)

    def encode(self, text: str) -> List[int]:
        """Tokenise *text* into a list of integer token IDs."""
        if self._tokenizer is None:
            raise RuntimeError(
                "No tokenizer available. Install transformers and ensure the "
                f"model name '{self._config.model_name}' is valid."
            )
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into a string."""
        if self._tokenizer is None:
            raise RuntimeError(
                "No tokenizer available. Install transformers and ensure the "
                f"model name '{self._config.model_name}' is valid."
            )
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    # -- KV-cache helpers ---------------------------------------------------

    def _init_kv_cache(self) -> None:
        """Initialise the KV cache based on model metadata heuristics."""
        # Try to infer layer count from output names
        layer_indices: set = set()
        for name in self._output_names:
            lowered = name.lower()
            if "present" in lowered or "past" in lowered:
                parts = name.split(".")
                for p in parts:
                    if p.isdigit():
                        layer_indices.add(int(p))

        num_layers = len(layer_indices) if layer_indices else 12

        # Try to infer head count / dim from output shapes
        num_heads = 12
        head_dim = 64
        if self._session is not None:
            for out in self._session.get_outputs():
                if "present" in out.name.lower() or "past" in out.name.lower():
                    shape = out.shape
                    if shape and len(shape) == 4:
                        # Typical: [batch, heads, seq, head_dim]
                        if isinstance(shape[1], int):
                            num_heads = shape[1]
                        if isinstance(shape[3], int):
                            head_dim = shape[3]
                    break

        self._kv_cache = KVCacheONNX(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=1,
            dtype=self._config.numpy_dtype,
            max_length=self._config.max_seq_len,
        )
        logger.debug(
            "Initialised KV cache: layers=%d, heads=%d, head_dim=%d",
            num_layers,
            num_heads,
            head_dim,
        )

    def reset_kv_cache(self) -> None:
        """Clear the KV cache for a new sequence."""
        if self._kv_cache is not None:
            self._kv_cache.clear()

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release the ONNX Runtime session and associated resources."""
        if self._closed:
            return

        logger.info("Closing ONNXLogitSource (%s)", self._config.model_name)

        # End profiling if active
        if self._profiling_started and self._session is not None:
            try:
                self._profile_file = self._session.end_profiling()
                self._profiling_started = False
            except Exception:
                pass

        # Clear KV cache
        if self._kv_cache is not None:
            self._kv_cache.clear()
            self._kv_cache = None

        # Release session
        self._session = None
        self._session_options = None

        super().close()

    def __enter__(self) -> "ONNXLogitSource":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # -- profiling & metadata -----------------------------------------------

    def profile_report(self) -> Dict[str, Any]:
        """Return profiling statistics if profiling was enabled.

        Returns:
            Dictionary with per-operator timing, total time, and the path
            to the raw profiling JSON file.
        """
        if not self._onnx_config.enable_profiling:
            return {"error": "Profiling is not enabled in ONNXConfig"}

        # End profiling to flush data
        if self._session is not None and self._profiling_started:
            try:
                self._profile_file = self._session.end_profiling()
                self._profiling_started = False
            except Exception as exc:
                return {"error": f"Failed to end profiling: {exc}"}

        if self._profile_file is None or not os.path.isfile(self._profile_file):
            return {"error": "No profiling data available"}

        try:
            with open(self._profile_file, "r") as f:
                raw_profile = json.load(f)
        except Exception as exc:
            return {"error": f"Failed to read profile: {exc}", "file": self._profile_file}

        # Summarise per-operator
        op_times: Dict[str, float] = {}
        op_counts: Dict[str, int] = {}
        total_time_us = 0.0

        events = raw_profile if isinstance(raw_profile, list) else raw_profile.get("traceEvents", [])
        for event in events:
            if not isinstance(event, dict):
                continue
            cat = event.get("cat", "")
            dur = event.get("dur", 0)
            name = event.get("name", "unknown")

            if cat in ("Node", "Op", "kernel"):
                op_times[name] = op_times.get(name, 0.0) + dur
                op_counts[name] = op_counts.get(name, 0) + 1
                total_time_us += dur

        sorted_ops = sorted(op_times.items(), key=lambda x: x[1], reverse=True)

        return {
            "profile_file": self._profile_file,
            "total_time_us": total_time_us,
            "total_time_ms": total_time_us / 1000.0,
            "num_events": len(events),
            "top_operators": [
                {
                    "name": name,
                    "total_us": t,
                    "count": op_counts.get(name, 0),
                    "avg_us": t / max(op_counts.get(name, 1), 1),
                    "pct": (t / total_time_us * 100) if total_time_us > 0 else 0,
                }
                for name, t in sorted_ops[:20]
            ],
            "source_stats": self._stats.summary(),
        }

    def model_metadata(self) -> Dict[str, Any]:
        """Return metadata about the loaded ONNX model.

        Includes input/output names and shapes, execution providers,
        session options, and model-level metadata properties.
        """
        if self._session is None:
            return {"error": "No ONNX session loaded"}

        inputs_info = []
        for inp in self._session.get_inputs():
            inputs_info.append(
                {"name": inp.name, "shape": inp.shape, "type": inp.type}
            )

        outputs_info = []
        for out in self._session.get_outputs():
            outputs_info.append(
                {"name": out.name, "shape": out.shape, "type": out.type}
            )

        # Model metadata
        meta = {}
        try:
            model_meta = self._session.get_modelmeta()
            meta = {
                "description": model_meta.description,
                "domain": model_meta.domain,
                "graph_name": model_meta.graph_name,
                "producer_name": model_meta.producer_name,
                "version": model_meta.version,
                "custom_metadata": dict(model_meta.custom_metadata_map),
            }
        except Exception:
            pass

        providers = []
        try:
            providers = self._session.get_providers()
        except Exception:
            pass

        return {
            "model_path": self._onnx_config.model_path,
            "inputs": inputs_info,
            "outputs": outputs_info,
            "supports_kv_cache": self._supports_kv_cache,
            "execution_providers": providers,
            "model_metadata": meta,
            "config": {
                "num_threads": self._onnx_config.num_threads,
                "inter_op_threads": self._onnx_config.inter_op_threads,
                "optimization_level": self._onnx_config.graph_optimization_level,
                "profiling": self._onnx_config.enable_profiling,
                "quantization": self._onnx_config.quantization_type,
            },
        }

    # -- private helpers ----------------------------------------------------

    def _check_closed(self) -> None:
        """Raise if the source has been closed."""
        if self._closed:
            raise RuntimeError("ONNXLogitSource has been closed")


# ---------------------------------------------------------------------------
# ONNXModelManager
# ---------------------------------------------------------------------------


class ONNXModelManager:
    """Utilities for downloading, converting, validating, and optimising ONNX
    models for use with :class:`ONNXLogitSource`.
    """

    # Well-known HuggingFace model IDs that have Optimum-exported ONNX variants
    _KNOWN_MODELS: List[str] = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "distilgpt2",
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "microsoft/phi-2",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "onnx_arena"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # -- public API ---------------------------------------------------------

    def download_model(
        self,
        model_name: str,
        output_dir: Optional[str] = None,
        opset_version: int = 14,
    ) -> str:
        """Download a HuggingFace model and convert it to ONNX format.

        First attempts to use ``optimum`` for export; falls back to manual
        PyTorch → ONNX conversion.

        Args:
            model_name: HuggingFace model identifier (e.g. ``"gpt2"``).
            output_dir: Directory for the ``.onnx`` file. Defaults to the
                cache directory.
            opset_version: ONNX opset version.

        Returns:
            Path to the exported ``.onnx`` file.
        """
        if output_dir is None:
            safe_name = model_name.replace("/", "_")
            output_dir_path = self._cache_dir / safe_name
        else:
            output_dir_path = Path(output_dir)

        output_dir_path.mkdir(parents=True, exist_ok=True)
        onnx_path = output_dir_path / "model.onnx"

        if onnx_path.exists():
            logger.info("ONNX model already cached at %s", onnx_path)
            return str(onnx_path)

        # Try Optimum export first
        try:
            from optimum.exporters.onnx import main_export

            logger.info("Exporting %s via Optimum", model_name)
            main_export(
                model_name,
                output=str(output_dir_path),
                task="causal-lm",
                opset=opset_version,
            )
            if onnx_path.exists():
                logger.info("Optimum export successful: %s", onnx_path)
                return str(onnx_path)

            # Optimum may use a different filename
            onnx_files = list(output_dir_path.glob("*.onnx"))
            if onnx_files:
                return str(onnx_files[0])
        except ImportError:
            logger.info("optimum not available; falling back to manual export")
        except Exception as exc:
            logger.warning("Optimum export failed (%s); trying manual export", exc)

        # Manual PyTorch → ONNX export
        return self._export_pytorch_to_onnx(
            model_name, str(onnx_path), opset_version
        )

    def _export_pytorch_to_onnx(
        self,
        model_name: str,
        output_path: str,
        opset_version: int = 14,
    ) -> str:
        """Export a HuggingFace PyTorch model to ONNX via ``torch.onnx.export``.

        Args:
            model_name: HuggingFace model identifier.
            output_path: Destination ``.onnx`` file.
            opset_version: ONNX opset version.

        Returns:
            *output_path*.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ONNX export. Install with: pip install torch"
            )
        if not _TOKENIZERS_AVAILABLE:
            raise ImportError(
                "transformers is required for ONNX export. "
                "Install with: pip install transformers"
            )

        from transformers import AutoModelForCausalLM

        logger.info("Loading PyTorch model %s for export", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()

        dummy_text = "Hello world"
        dummy_input = tokenizer(dummy_text, return_tensors="pt")
        input_ids = dummy_input["input_ids"]
        attention_mask = dummy_input["attention_mask"]

        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        }

        logger.info("Exporting to ONNX (opset %d): %s", opset_version, output_path)
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            output_path,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
        )

        logger.info("Export complete: %s (%.1f MB)", output_path, self.model_size(output_path) / 1e6)
        return output_path

    def quantize_model(
        self,
        model_path: str,
        output_path: str,
        quant_type: str = "int8",
    ) -> str:
        """Apply quantization to an ONNX model.

        Args:
            model_path: Source ``.onnx`` file.
            output_path: Destination quantized ``.onnx`` file.
            quant_type: ``"int8"`` or ``"int4"``.

        Returns:
            *output_path*.
        """
        quantizer = ONNXQuantizer()
        return quantizer.quantize_dynamic(model_path, output_path, quant_type)

    def validate_model(self, model_path: str) -> bool:
        """Check that an ONNX model file is valid.

        Uses ``onnx.checker.check_model`` if the ``onnx`` package is
        available; otherwise falls back to attempting a session load.

        Returns:
            ``True`` if validation passes.
        """
        if not os.path.isfile(model_path):
            logger.error("Model file not found: %s", model_path)
            return False

        # Try onnx checker first
        if _ONNX_AVAILABLE:
            try:
                model = onnx.load(model_path)
                onnx.checker.check_model(model)
                logger.info("ONNX model validation passed: %s", model_path)
                return True
            except Exception as exc:
                logger.error("ONNX model validation failed: %s", exc)
                return False

        # Fall back to session load
        _require_ort()
        try:
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            _ = sess.get_inputs()
            logger.info("ONNX model loads successfully: %s", model_path)
            return True
        except Exception as exc:
            logger.error("ONNX model load failed: %s", exc)
            return False

    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Return input/output names, shapes, and metadata for an ONNX model.

        Args:
            model_path: Path to the ``.onnx`` file.

        Returns:
            Dictionary with ``inputs``, ``outputs``, ``opset_version``,
            ``ir_version``, ``producer``, and ``file_size_bytes`` keys.
        """
        result: Dict[str, Any] = {
            "model_path": model_path,
            "file_size_bytes": self.model_size(model_path),
            "file_size_mb": round(self.model_size(model_path) / (1024 * 1024), 2),
        }

        if _ONNX_AVAILABLE:
            try:
                model = onnx.load(model_path)
                graph = model.graph

                result["inputs"] = [
                    {
                        "name": inp.name,
                        "shape": [
                            d.dim_value if d.dim_value > 0 else d.dim_param
                            for d in inp.type.tensor_type.shape.dim
                        ],
                        "dtype": onnx.TensorProto.DataType.Name(
                            inp.type.tensor_type.elem_type
                        ),
                    }
                    for inp in graph.input
                ]
                result["outputs"] = [
                    {
                        "name": out.name,
                        "shape": [
                            d.dim_value if d.dim_value > 0 else d.dim_param
                            for d in out.type.tensor_type.shape.dim
                        ],
                        "dtype": onnx.TensorProto.DataType.Name(
                            out.type.tensor_type.elem_type
                        ),
                    }
                    for out in graph.output
                ]
                result["opset_version"] = [
                    {"domain": opset.domain or "ai.onnx", "version": opset.version}
                    for opset in model.opset_import
                ]
                result["ir_version"] = model.ir_version
                result["producer"] = model.producer_name
                result["doc_string"] = model.doc_string
                result["num_nodes"] = len(graph.node)
                result["num_initializers"] = len(graph.initializer)

                return result
            except Exception as exc:
                logger.warning("Failed to read ONNX model graph: %s", exc)

        # Fall back to ORT session
        _require_ort()
        try:
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            result["inputs"] = [
                {"name": i.name, "shape": i.shape, "type": i.type}
                for i in sess.get_inputs()
            ]
            result["outputs"] = [
                {"name": o.name, "shape": o.shape, "type": o.type}
                for o in sess.get_outputs()
            ]
        except Exception as exc:
            result["error"] = str(exc)

        return result

    def optimize_model(
        self,
        model_path: str,
        output_path: str,
        model_type: str = "gpt2",
        num_heads: int = 12,
        hidden_size: int = 768,
    ) -> str:
        """Apply graph optimizations to an ONNX model.

        Uses ``onnxruntime.transformers.optimizer`` if available; otherwise
        falls back to ORT session-level optimisation and re-export.

        Args:
            model_path: Source ``.onnx`` file.
            output_path: Destination optimised ``.onnx`` file.
            model_type: Model architecture hint (``"gpt2"``, ``"bert"``, etc.).
            num_heads: Number of attention heads.
            hidden_size: Hidden dimension size.

        Returns:
            *output_path*.
        """
        if _ORT_OPTIM_AVAILABLE:
            try:
                optimized = _ort_optimize(
                    model_path,
                    model_type=model_type,
                    num_heads=num_heads,
                    hidden_size=hidden_size,
                    optimization_options=None,
                )
                optimized.save_model_to_file(output_path)
                logger.info("Optimised model saved to %s", output_path)
                return output_path
            except Exception as exc:
                logger.warning("ORT optimizer failed: %s; falling back to session-level", exc)

        # Fallback: session-level optimization
        _require_ort()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.optimized_model_filepath = output_path

        _ = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info("Session-level optimised model saved to %s", output_path)
        return output_path

    def list_available_models(self) -> List[str]:
        """Return the list of known model identifiers that can be exported."""
        return list(self._KNOWN_MODELS)

    def model_size(self, model_path: str) -> int:
        """Return the file size of an ONNX model in bytes."""
        p = Path(model_path)
        if not p.exists():
            return 0
        return p.stat().st_size

    def list_cached_models(self) -> List[Dict[str, Any]]:
        """List all models currently in the local cache directory."""
        results: List[Dict[str, Any]] = []
        if not self._cache_dir.exists():
            return results

        for entry in sorted(self._cache_dir.iterdir()):
            if entry.is_dir():
                onnx_files = list(entry.glob("*.onnx"))
                for f in onnx_files:
                    results.append(
                        {
                            "name": entry.name,
                            "path": str(f),
                            "size_bytes": f.stat().st_size,
                            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                        }
                    )

        return results


# ---------------------------------------------------------------------------
# ONNXQuantizer
# ---------------------------------------------------------------------------


class ONNXQuantizer:
    """Dynamic and static quantization utilities for ONNX models."""

    def quantize_dynamic(
        self,
        model_path: str,
        output_path: str,
        quant_type: str = "int8",
    ) -> str:
        """Apply dynamic quantization.

        Args:
            model_path: Source ``.onnx`` file.
            output_path: Destination quantized ``.onnx`` file.
            quant_type: ``"int8"`` or ``"int4"``.

        Returns:
            *output_path*.
        """
        if not _ORT_QUANT_AVAILABLE:
            raise ImportError(
                "onnxruntime quantization tools are required. "
                "Install with: pip install onnxruntime"
            )

        weight_type = self._resolve_quant_type(quant_type)

        logger.info(
            "Applying dynamic %s quantization: %s → %s",
            quant_type,
            model_path,
            output_path,
        )

        try:
            quantize_dynamic(
                model_input=model_path,
                model_output=output_path,
                weight_type=weight_type,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Dynamic quantization failed: {exc}"
            ) from exc

        original_size = Path(model_path).stat().st_size
        quant_size = Path(output_path).stat().st_size
        reduction = 1.0 - quant_size / original_size if original_size > 0 else 0.0

        logger.info(
            "Quantization complete. Size reduction: %.1f%% (%d → %d bytes)",
            reduction * 100,
            original_size,
            quant_size,
        )
        return output_path

    def quantize_static(
        self,
        model_path: str,
        output_path: str,
        calibration_data: Optional[List[Dict[str, np.ndarray]]] = None,
        calibration_texts: Optional[List[str]] = None,
        tokenizer_name: Optional[str] = None,
        quant_type: str = "int8",
        num_calibration_samples: int = 100,
    ) -> str:
        """Apply static quantization with calibration data.

        Either provide *calibration_data* directly (list of feed dicts) or
        *calibration_texts* + *tokenizer_name* to auto-generate calibration
        inputs.

        Args:
            model_path: Source ``.onnx`` file.
            output_path: Destination quantized ``.onnx`` file.
            calibration_data: Pre-built calibration samples.
            calibration_texts: Raw texts for calibration.
            tokenizer_name: HuggingFace tokenizer for encoding texts.
            quant_type: ``"int8"`` or ``"int4"``.
            num_calibration_samples: Max calibration samples to use.

        Returns:
            *output_path*.
        """
        if not _ORT_QUANT_AVAILABLE:
            raise ImportError(
                "onnxruntime quantization tools are required. "
                "Install with: pip install onnxruntime"
            )

        weight_type = self._resolve_quant_type(quant_type)

        # Build calibration reader
        if calibration_data is not None:
            reader = _ListCalibrationReader(calibration_data[:num_calibration_samples])
        elif calibration_texts is not None and tokenizer_name is not None:
            cal_data = self._prepare_calibration_dataset(
                calibration_texts[:num_calibration_samples],
                tokenizer_name,
            )
            reader = _ListCalibrationReader(cal_data)
        else:
            raise ValueError(
                "Provide either calibration_data or "
                "(calibration_texts + tokenizer_name) for static quantization."
            )

        logger.info(
            "Applying static %s quantization with %d calibration samples",
            quant_type,
            len(reader),
        )

        try:
            quantize_static(
                model_input=model_path,
                model_output=output_path,
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                weight_type=weight_type,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Static quantization failed: {exc}"
            ) from exc

        logger.info("Static quantization complete: %s", output_path)
        return output_path

    def _prepare_calibration_dataset(
        self,
        texts: List[str],
        tokenizer_name: str,
        max_length: int = 128,
    ) -> List[Dict[str, np.ndarray]]:
        """Tokenize texts into calibration feed dicts.

        Args:
            texts: Raw calibration texts.
            tokenizer_name: HuggingFace tokenizer identifier.
            max_length: Maximum token length per sample.

        Returns:
            List of feed dicts with ``input_ids`` and ``attention_mask``.
        """
        if not _TOKENIZERS_AVAILABLE:
            raise ImportError(
                "transformers is required for calibration dataset preparation. "
                "Install with: pip install transformers"
            )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        samples: List[Dict[str, np.ndarray]] = []
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            sample: Dict[str, np.ndarray] = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            samples.append(sample)

        return samples

    def compare_accuracy(
        self,
        original_path: str,
        quantized_path: str,
        test_data: List[Dict[str, np.ndarray]],
        rtol: float = 1e-2,
        atol: float = 1e-3,
    ) -> Dict[str, Any]:
        """Compare outputs of the original and quantized models.

        Args:
            original_path: Path to the original ``.onnx`` model.
            quantized_path: Path to the quantized ``.onnx`` model.
            test_data: List of feed dicts to run through both models.
            rtol: Relative tolerance for ``np.allclose``.
            atol: Absolute tolerance for ``np.allclose``.

        Returns:
            Dictionary with per-sample and aggregate accuracy metrics.
        """
        _require_ort()

        orig_sess = ort.InferenceSession(original_path, providers=["CPUExecutionProvider"])
        quant_sess = ort.InferenceSession(quantized_path, providers=["CPUExecutionProvider"])

        results: List[Dict[str, Any]] = []
        all_diffs: List[float] = []

        for idx, sample in enumerate(test_data):
            try:
                orig_out = orig_sess.run(None, sample)
                quant_out = quant_sess.run(None, sample)

                if not orig_out or not quant_out:
                    results.append({"sample": idx, "error": "empty output"})
                    continue

                orig_logits = orig_out[0].astype(np.float32)
                quant_logits = quant_out[0].astype(np.float32)

                max_diff = float(np.max(np.abs(orig_logits - quant_logits)))
                mean_diff = float(np.mean(np.abs(orig_logits - quant_logits)))
                is_close = bool(np.allclose(orig_logits, quant_logits, rtol=rtol, atol=atol))

                # Top-k agreement
                if orig_logits.ndim >= 2:
                    flat_orig = orig_logits.reshape(-1, orig_logits.shape[-1])
                    flat_quant = quant_logits.reshape(-1, quant_logits.shape[-1])
                    top1_orig = np.argmax(flat_orig, axis=-1)
                    top1_quant = np.argmax(flat_quant, axis=-1)
                    top1_agree = float(np.mean(top1_orig == top1_quant))
                else:
                    top1_agree = float(np.argmax(orig_logits) == np.argmax(quant_logits))

                all_diffs.append(mean_diff)
                results.append(
                    {
                        "sample": idx,
                        "max_diff": max_diff,
                        "mean_diff": mean_diff,
                        "is_close": is_close,
                        "top1_agreement": top1_agree,
                    }
                )
            except Exception as exc:
                results.append({"sample": idx, "error": str(exc)})

        return {
            "num_samples": len(test_data),
            "aggregate": {
                "mean_diff": float(np.mean(all_diffs)) if all_diffs else None,
                "max_diff": float(np.max(all_diffs)) if all_diffs else None,
                "all_close_pct": (
                    sum(1 for r in results if r.get("is_close", False)) / len(results) * 100
                    if results
                    else 0.0
                ),
                "avg_top1_agreement": (
                    float(np.mean([r["top1_agreement"] for r in results if "top1_agreement" in r]))
                    if any("top1_agreement" in r for r in results)
                    else None
                ),
            },
            "per_sample": results,
            "tolerances": {"rtol": rtol, "atol": atol},
        }

    def size_reduction(self, original_path: str, quantized_path: str) -> float:
        """Return the fractional size reduction (0.0–1.0).

        A value of 0.6 means the quantized model is 60% smaller.
        """
        orig_size = Path(original_path).stat().st_size
        quant_size = Path(quantized_path).stat().st_size
        if orig_size == 0:
            return 0.0
        return 1.0 - quant_size / orig_size

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _resolve_quant_type(quant_type: str) -> Any:
        """Map a string quantization type to the ORT enum."""
        if not _ORT_QUANT_AVAILABLE:
            raise ImportError("onnxruntime.quantization is not available")

        mapping = {
            "int8": QuantType.QInt8,
            "uint8": QuantType.QUInt8,
        }

        # int4 support varies by ORT version
        if quant_type == "int4":
            if hasattr(QuantType, "QInt4"):
                return QuantType.QInt4
            else:
                logger.warning(
                    "int4 quantization not available in this ORT version; "
                    "falling back to int8"
                )
                return QuantType.QInt8

        if quant_type not in mapping:
            raise ValueError(
                f"Unsupported quantization type: {quant_type!r}. "
                f"Choose from: {list(mapping)} + ['int4']"
            )
        return mapping[quant_type]


class _ListCalibrationReader:
    """Adapter that wraps a list of feed dicts into a
    ``CalibrationDataReader`` interface for ONNX Runtime static quantization.
    """

    def __init__(self, data: List[Dict[str, np.ndarray]]) -> None:
        self._data = data
        self._index = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._index >= len(self._data):
            return None
        sample = self._data[self._index]
        self._index += 1
        return sample

    def rewind(self) -> None:
        self._index = 0

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# ONNXProfiler
# ---------------------------------------------------------------------------


class ONNXProfiler:
    """Profiling utilities for ONNX Runtime sessions and logit sources.

    Provides latency, throughput, per-op timing, and memory analysis.
    """

    def __init__(self) -> None:
        self._profile_data: Optional[List[Dict[str, Any]]] = None
        self._profile_file: Optional[str] = None

    def start_profiling(self, session: "ort.InferenceSession") -> None:
        """Enable profiling on an existing session.

        .. note::
           For profiling to work, the session must have been created with
           ``enable_profiling=True`` in ``SessionOptions``.  This method
           records the start timestamp for throughput calculations.
        """
        _require_ort()
        self._start_time = time.monotonic()
        self._profile_data = None
        self._profile_file = None
        logger.debug("Profiling started")

    def stop_profiling(
        self,
        session: "ort.InferenceSession",
    ) -> Dict[str, Any]:
        """End profiling and return parsed results.

        Args:
            session: The session whose profiling should be ended.

        Returns:
            Dictionary with raw events, summary, and file path.
        """
        _require_ort()
        elapsed = time.monotonic() - getattr(self, "_start_time", time.monotonic())

        try:
            profile_file = session.end_profiling()
            self._profile_file = profile_file
        except Exception as exc:
            return {"error": f"Failed to end profiling: {exc}"}

        if not os.path.isfile(profile_file):
            return {"error": f"Profile file not found: {profile_file}"}

        try:
            with open(profile_file, "r") as f:
                raw = json.load(f)
        except Exception as exc:
            return {"error": f"Failed to parse profile JSON: {exc}"}

        events = raw if isinstance(raw, list) else raw.get("traceEvents", [])
        self._profile_data = events

        summary = self.summarize_profile(events)
        summary["profile_file"] = profile_file
        summary["wall_time_s"] = elapsed
        return summary

    def summarize_profile(
        self,
        profile_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Summarise per-operator timing from ONNX Runtime profile events.

        Args:
            profile_data: List of Chrome Trace–format event dicts.

        Returns:
            Dictionary with per-op totals, top-N operators, and aggregates.
        """
        op_stats: Dict[str, Dict[str, Any]] = {}
        total_dur_us = 0.0
        node_count = 0

        for event in profile_data:
            if not isinstance(event, dict):
                continue
            cat = event.get("cat", "")
            dur = event.get("dur", 0)
            name = event.get("name", "unknown")

            if cat not in ("Node", "Op", "kernel", "Session"):
                continue

            node_count += 1
            total_dur_us += dur

            if name not in op_stats:
                op_stats[name] = {
                    "total_us": 0.0,
                    "count": 0,
                    "min_us": float("inf"),
                    "max_us": 0.0,
                }
            op_stats[name]["total_us"] += dur
            op_stats[name]["count"] += 1
            op_stats[name]["min_us"] = min(op_stats[name]["min_us"], dur)
            op_stats[name]["max_us"] = max(op_stats[name]["max_us"], dur)

        # Compute averages and percentages
        for name, stats in op_stats.items():
            stats["avg_us"] = stats["total_us"] / max(stats["count"], 1)
            stats["pct"] = (
                stats["total_us"] / total_dur_us * 100 if total_dur_us > 0 else 0.0
            )
            if stats["min_us"] == float("inf"):
                stats["min_us"] = 0.0

        # Sort by total time
        sorted_ops = sorted(op_stats.items(), key=lambda x: x[1]["total_us"], reverse=True)

        return {
            "total_time_us": total_dur_us,
            "total_time_ms": total_dur_us / 1000.0,
            "num_events": node_count,
            "num_unique_ops": len(op_stats),
            "top_operators": [
                {"name": name, **stats} for name, stats in sorted_ops[:20]
            ],
            "all_operators": {name: stats for name, stats in sorted_ops},
        }

    def memory_usage(
        self,
        session: "ort.InferenceSession",
    ) -> Dict[str, Any]:
        """Estimate memory usage of an ONNX Runtime session.

        Uses ``onnxruntime``'s ``get_session_options`` and model metadata to
        produce estimates.  Exact RSS tracking is OS-dependent.

        Args:
            session: Active ONNX Runtime session.

        Returns:
            Dictionary with estimated memory figures.
        """
        _require_ort()

        result: Dict[str, Any] = {}

        # Try to get model size
        try:
            model_path = None
            for inp in session.get_inputs():
                # No direct way to get model path from session; estimate from
                # model metadata.
                break

            # Estimate parameter memory from output/input shapes
            total_params = 0
            for out in session.get_outputs():
                shape = out.shape
                if shape:
                    size = 1
                    for dim in shape:
                        if isinstance(dim, int) and dim > 0:
                            size *= dim
                    if size > 1:
                        total_params += size

            result["estimated_output_params"] = total_params
            result["estimated_output_memory_mb"] = round(total_params * 4 / (1024 * 1024), 2)
        except Exception:
            pass

        # Try process-level memory (platform-dependent)
        try:
            import resource

            rusage = resource.getrusage(resource.RUSAGE_SELF)
            result["process_max_rss_mb"] = round(rusage.ru_maxrss / 1024, 2)
        except ImportError:
            pass
        except Exception:
            pass

        # Try psutil if available
        try:
            import psutil

            proc = psutil.Process(os.getpid())
            mem_info = proc.memory_info()
            result["process_rss_mb"] = round(mem_info.rss / (1024 * 1024), 2)
            result["process_vms_mb"] = round(mem_info.vms / (1024 * 1024), 2)
        except ImportError:
            pass
        except Exception:
            pass

        result["providers"] = session.get_providers()
        return result

    def throughput_test(
        self,
        source: ONNXLogitSource,
        num_iterations: int = 50,
        seq_lengths: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Run a throughput and latency benchmark on an :class:`ONNXLogitSource`.

        Args:
            source: The ONNX logit source to benchmark.
            num_iterations: Number of inference calls per sequence length.
            seq_lengths: List of sequence lengths to test. Defaults to
                ``[8, 32, 64, 128, 256, 512]``.

        Returns:
            Dictionary with per-length and aggregate throughput/latency stats.
        """
        if seq_lengths is None:
            seq_lengths = [8, 32, 64, 128, 256, 512]

        results_per_length: List[Dict[str, Any]] = []
        all_latencies: List[float] = []

        for seq_len in seq_lengths:
            if seq_len > source.max_seq_len:
                logger.info("Skipping seq_len=%d (exceeds max_seq_len=%d)", seq_len, source.max_seq_len)
                continue

            # Random input IDs
            rng = np.random.RandomState(42)
            input_ids = rng.randint(0, source.vocab_size, size=seq_len).tolist()

            latencies: List[float] = []

            # Warmup (3 iterations)
            for _ in range(min(3, num_iterations)):
                try:
                    source.get_next_token_logits(input_ids)
                except Exception:
                    break

            # Timed iterations
            for _ in range(num_iterations):
                t0 = time.monotonic()
                try:
                    source.get_next_token_logits(input_ids)
                    elapsed = time.monotonic() - t0
                    latencies.append(elapsed)
                except Exception as exc:
                    logger.warning("Iteration failed (seq_len=%d): %s", seq_len, exc)

            if latencies:
                arr = np.array(latencies)
                length_result = {
                    "seq_length": seq_len,
                    "num_iterations": len(latencies),
                    "mean_latency_ms": float(np.mean(arr) * 1000),
                    "p50_latency_ms": float(np.percentile(arr, 50) * 1000),
                    "p95_latency_ms": float(np.percentile(arr, 95) * 1000),
                    "p99_latency_ms": float(np.percentile(arr, 99) * 1000),
                    "min_latency_ms": float(np.min(arr) * 1000),
                    "max_latency_ms": float(np.max(arr) * 1000),
                    "std_latency_ms": float(np.std(arr) * 1000),
                    "throughput_tokens_per_s": seq_len / float(np.mean(arr)) if np.mean(arr) > 0 else 0,
                }
                results_per_length.append(length_result)
                all_latencies.extend(latencies)

        aggregate: Dict[str, Any] = {}
        if all_latencies:
            arr = np.array(all_latencies)
            aggregate = {
                "total_iterations": len(all_latencies),
                "overall_mean_latency_ms": float(np.mean(arr) * 1000),
                "overall_p95_latency_ms": float(np.percentile(arr, 95) * 1000),
                "overall_min_latency_ms": float(np.min(arr) * 1000),
                "overall_max_latency_ms": float(np.max(arr) * 1000),
            }

        return {
            "per_length": results_per_length,
            "aggregate": aggregate,
            "config": {
                "num_iterations": num_iterations,
                "seq_lengths_tested": [r["seq_length"] for r in results_per_length],
            },
        }


# ---------------------------------------------------------------------------
# Factory & helper functions
# ---------------------------------------------------------------------------


def create_onnx_source(
    model_name: str,
    quantize: bool = True,
    cache_dir: Optional[str] = None,
    num_threads: int = 4,
    execution_providers: Optional[List[str]] = None,
    max_seq_len: int = 1024,
    vocab_size: int = 50257,
    use_kv_cache: bool = True,
) -> ONNXLogitSource:
    """High-level factory: download (or locate), optionally quantize, and
    return a ready-to-use :class:`ONNXLogitSource`.

    Args:
        model_name: HuggingFace model identifier.
        quantize: Whether to apply dynamic int8 quantization.
        cache_dir: Local cache directory for ONNX models.
        num_threads: Intra-op thread count.
        execution_providers: ONNX Runtime execution providers.
        max_seq_len: Maximum sequence length.
        vocab_size: Vocabulary size.
        use_kv_cache: Whether to enable KV caching.

    Returns:
        Initialised :class:`ONNXLogitSource`.
    """
    _require_ort()

    manager = ONNXModelManager(cache_dir=cache_dir)

    # Download / export
    logger.info("Preparing ONNX model for %s", model_name)
    onnx_path = manager.download_model(model_name)

    # Optionally quantize
    if quantize:
        safe_name = model_name.replace("/", "_")
        quant_dir = manager._cache_dir / safe_name
        quant_path = str(quant_dir / "model_quantized.onnx")

        if not Path(quant_path).exists():
            logger.info("Quantizing model (dynamic int8)")
            try:
                manager.quantize_model(onnx_path, quant_path, quant_type="int8")
                onnx_path = quant_path
            except Exception as exc:
                logger.warning(
                    "Quantization failed (%s); using unquantized model", exc
                )
        else:
            logger.info("Using cached quantized model: %s", quant_path)
            onnx_path = quant_path

    # Validate
    if not manager.validate_model(onnx_path):
        raise RuntimeError(f"ONNX model validation failed: {onnx_path}")

    # Build configs
    if execution_providers is None:
        execution_providers = ["CPUExecutionProvider"]

    onnx_cfg = ONNXConfig(
        model_path=onnx_path,
        num_threads=num_threads,
        execution_providers=execution_providers,
        quantization_type="int8" if quantize else None,
    )

    logit_cfg = LogitSourceConfig(
        model_name=model_name,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        device="cpu",
        use_kv_cache=use_kv_cache,
    )

    source = ONNXLogitSource(config=onnx_cfg, logit_source_config=logit_cfg)
    logger.info(
        "ONNXLogitSource ready for %s (%s)",
        model_name,
        "quantized" if quantize else "fp32",
    )
    return source


def benchmark_onnx_source(
    source: ONNXLogitSource,
    n_iterations: int = 50,
    seq_lengths: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run a throughput / latency benchmark on an :class:`ONNXLogitSource`.

    This is a thin wrapper around :meth:`ONNXProfiler.throughput_test`.

    Args:
        source: The ONNX logit source to benchmark.
        n_iterations: Number of inference calls per sequence length.
        seq_lengths: Sequence lengths to test.

    Returns:
        Benchmark results dict.
    """
    profiler = ONNXProfiler()
    results = profiler.throughput_test(
        source,
        num_iterations=n_iterations,
        seq_lengths=seq_lengths,
    )
    results["source_stats"] = source.stats.summary()
    results["model_metadata"] = source.model_metadata()
    return results


def validate_onnx_outputs(
    source: ONNXLogitSource,
    reference_source: LogitSource,
    test_inputs: List[List[int]],
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> Dict[str, Any]:
    """Compare ONNX source outputs against a reference :class:`LogitSource`.

    Useful for validating that an ONNX-exported or quantized model produces
    outputs consistent with the original backend.

    Args:
        source: The ONNX logit source to validate.
        reference_source: The reference (e.g. PyTorch) logit source.
        test_inputs: List of token-ID sequences to test.
        rtol: Relative tolerance for ``np.allclose``.
        atol: Absolute tolerance for ``np.allclose``.

    Returns:
        Dictionary with per-sample comparison and aggregate accuracy metrics.
    """
    results: List[Dict[str, Any]] = []
    all_max_diffs: List[float] = []
    all_mean_diffs: List[float] = []
    top1_agreements: List[float] = []

    for idx, input_ids in enumerate(test_inputs):
        try:
            onnx_logits = source.get_next_token_logits(input_ids)
            ref_logits = reference_source.get_next_token_logits(input_ids)

            onnx_f32 = onnx_logits.astype(np.float32)
            ref_f32 = ref_logits.astype(np.float32)

            max_diff = float(np.max(np.abs(onnx_f32 - ref_f32)))
            mean_diff = float(np.mean(np.abs(onnx_f32 - ref_f32)))
            is_close = bool(np.allclose(onnx_f32, ref_f32, rtol=rtol, atol=atol))

            # Cosine similarity
            dot = np.dot(onnx_f32, ref_f32)
            norm_onnx = np.linalg.norm(onnx_f32)
            norm_ref = np.linalg.norm(ref_f32)
            cosine_sim = float(dot / (norm_onnx * norm_ref + 1e-12))

            # Top-k agreement
            top1_onnx = int(np.argmax(onnx_f32))
            top1_ref = int(np.argmax(ref_f32))
            top1_match = top1_onnx == top1_ref

            k = min(10, len(onnx_f32))
            topk_onnx = set(np.argsort(onnx_f32)[-k:].tolist())
            topk_ref = set(np.argsort(ref_f32)[-k:].tolist())
            topk_overlap = len(topk_onnx & topk_ref) / k

            # KL divergence (softmax first)
            def _softmax(x: np.ndarray) -> np.ndarray:
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()

            p = _softmax(ref_f32)
            q = _softmax(onnx_f32)
            kl_div = float(np.sum(p * np.log(p / (q + 1e-12) + 1e-12)))

            all_max_diffs.append(max_diff)
            all_mean_diffs.append(mean_diff)
            top1_agreements.append(1.0 if top1_match else 0.0)

            results.append(
                {
                    "sample": idx,
                    "num_tokens": len(input_ids),
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                    "is_close": is_close,
                    "cosine_similarity": cosine_sim,
                    "top1_match": top1_match,
                    "top1_onnx": top1_onnx,
                    "top1_ref": top1_ref,
                    "topk_overlap": topk_overlap,
                    "kl_divergence": kl_div,
                }
            )

        except Exception as exc:
            results.append({"sample": idx, "error": str(exc)})

    # Aggregate
    aggregate: Dict[str, Any] = {}
    if all_max_diffs:
        aggregate = {
            "num_samples": len(test_inputs),
            "num_successful": len(all_max_diffs),
            "num_errors": len(test_inputs) - len(all_max_diffs),
            "avg_max_diff": float(np.mean(all_max_diffs)),
            "worst_max_diff": float(np.max(all_max_diffs)),
            "avg_mean_diff": float(np.mean(all_mean_diffs)),
            "top1_agreement_pct": float(np.mean(top1_agreements) * 100),
            "all_close_pct": (
                sum(1 for r in results if r.get("is_close", False)) / len(results) * 100
            ),
        }

    return {
        "aggregate": aggregate,
        "per_sample": results,
        "tolerances": {"rtol": rtol, "atol": atol},
    }
