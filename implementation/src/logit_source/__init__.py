"""
LogitSource abstraction layer.

Provides a unified interface for obtaining next-token logit distributions
from language models, regardless of whether inference is performed live,
loaded from cache, or read from pre-computed files.

Classes:
    LogitSource: Abstract base class defining the interface.
    LiveLogitSource: Real-time model inference via ONNX Runtime or PyTorch.
    CachedLogitSource: Content-addressed LRU cache wrapping another source.
    ONNXLogitSource: Direct ONNX Runtime session management.
"""

from src.logit_source.base import LogitSource, LogitSourceConfig, LogitBatch
from src.logit_source.live import LiveLogitSource
from src.logit_source.cached import CachedLogitSource
from src.logit_source.onnx_source import ONNXLogitSource

__all__ = [
    "LogitSource",
    "LogitSourceConfig",
    "LogitBatch",
    "LiveLogitSource",
    "CachedLogitSource",
    "ONNXLogitSource",
]
