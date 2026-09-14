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
#
# The third field names the extra whose absence is the expected reason for an
# ImportError, so a core-only install gets told which extra to add rather than
# a traceback ending in an arbitrary transitive module. None means the name
# imports on the default dependency set (tracker core), and an ImportError
# there is a real fault.
_LAZY: dict[str, tuple[str, str, str | None]] = {
    "GPUByteTracker": (
        "saccade.perception.tracking.tracker_gpu",
        "GPUByteTracker",
        None,
    ),
    "ReorderingBuffer": (
        "saccade.perception.tracking.reorder",
        "ReorderingBuffer",
        None,
    ),
    "run_eval": ("saccade.perception.eval.evaluator", "run_eval", "eval"),
    "EvalConfig": ("saccade.perception.eval.config", "EvalConfig", None),
    "MambaGatedDetector": (
        "saccade.perception.temporal_yolo.mamba_gated_detector",
        "MambaGatedDetector",
        None,
    ),
}
__all__ = ["__version__", *_LAZY]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr, extra = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'saccade' has no attribute {name!r}") from None
    try:
        module = import_module(module_name)
    except ImportError as exc:
        if extra is None:
            raise
        raise ImportError(
            f"saccade.{name} needs the {extra!r} extra "
            f"(pip install 'saccade[{extra}]'): {exc}"
        ) from exc
    value = getattr(module, attr)
    globals()[name] = value  # cache: __getattr__ runs once per name
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])
