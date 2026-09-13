"""saccade: real-time edge video perception.

Top-level names resolve lazily, so ``import saccade`` does not pull in torch or
the native extensions; that cost is paid on first attribute access.
"""

from importlib import import_module, metadata
from typing import Any

try:
    __version__ = metadata.version("saccade")
except metadata.PackageNotFoundError:  # source tree without an install
    __version__ = "0.0.0+unknown"

# Each entry points at the leaf module, not a subpackage __init__:
# saccade.perception.tracking.__init__ imports tracker_gpu eagerly, which is
# exactly the torch + native-extension load this indirection defers.
_LAZY: dict[str, tuple[str, str]] = {
    "GPUByteTracker": ("saccade.perception.tracking.tracker_gpu", "GPUByteTracker"),
    "ReorderingBuffer": ("saccade.perception.tracking.reorder", "ReorderingBuffer"),
    "run_eval": ("saccade.perception.eval.evaluator", "run_eval"),
    "EvalConfig": ("saccade.perception.eval.config", "EvalConfig"),
    "MambaGatedDetector": (
        "saccade.perception.temporal_yolo.mamba_gated_detector",
        "MambaGatedDetector",
    ),
}
__all__ = ["__version__", *_LAZY]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'saccade' has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attr)
    globals()[name] = value  # cache: __getattr__ runs once per name
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])
