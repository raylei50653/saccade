"""Runtime path contract for the ``saccade`` package.

Everything the package reads from disk at runtime is one of three kinds, and
each kind has exactly one way of being found:

* **package-owned** -- files that ship inside ``saccade/`` (the research CUDA
  helper under ``perception/eval/_cuda``). A module addresses those relative to
  its own ``__file__``; nothing here is involved.
* **external runtime inputs** -- models, engines, checkpoints, datasets, output
  directories. They are constructor / config / CLI arguments, and a relative
  value is relative to the caller's working directory, as for any command-line
  tool. :func:`runtime_input` is that convention written down once.
* **repository-provided** -- the native build products, the vendored TrackEval
  and, for research provenance tools, the checkout's own sources. These used to
  be inferred per module from the package location (``Path(__file__)
  .parents[4]``), which silently assumed ``src/saccade`` inside a checkout. They
  resolve here, once: an explicit input first, the source checkout second, and
  ``None`` -- never a guess -- when neither applies.

Nothing in this module imports torch or a native extension, and nothing is
resolved at import time; every function reads the environment when called.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "BUILD_PATH_ENV",
    "TRACKEVAL_ROOT_ENV",
    "SourceCheckoutRequired",
    "build_dir",
    "require_source_checkout",
    "runtime_input",
    "source_checkout_root",
    "trackeval_root",
]

#: Native build directory (the ``saccade_*_ext`` modules, the TensorRT scan
#: plugin, ``cuda_reid/``). Honoured by ``scripts/eval/mot17.py`` and the H2
#: tooling already; the package reads the same variable.
BUILD_PATH_ENV = "SACCADE_BUILD_PATH"

#: Root of a TrackEval checkout (the directory containing ``trackeval/``). Only
#: needed when the package is used outside the repository and ``trackeval`` is
#: not otherwise importable.
TRACKEVAL_ROOT_ENV = "SACCADE_TRACKEVAL_ROOT"


class SourceCheckoutRequired(RuntimeError):
    """A repository-only operation was requested from outside a checkout."""


def source_checkout_root() -> Path | None:
    """The repository this package is imported from, or ``None``.

    Only the ``<root>/src/saccade`` layout with ``pyproject.toml`` beside
    ``src/`` qualifies. A copy installed into site-packages, or vendored
    anywhere else, returns ``None`` rather than a directory four levels up.
    """
    package_dir = Path(__file__).resolve().parent
    src_dir = package_dir.parent
    root = src_dir.parent
    if src_dir.name != "src" or not (root / "pyproject.toml").is_file():
        return None
    return root


def require_source_checkout(purpose: str) -> Path:
    """:func:`source_checkout_root`, failing with *purpose* when there is none."""
    root = source_checkout_root()
    if root is None:
        raise SourceCheckoutRequired(
            f"{purpose} needs the saccade source checkout, but saccade is "
            f"imported from {Path(__file__).resolve().parent}, which is not "
            "<checkout>/src/saccade"
        )
    return root


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "")
    if not value.strip():
        return None
    return Path(value).expanduser().resolve()


def build_dir() -> Path | None:
    """Where the native build products live, or ``None``.

    ``SACCADE_BUILD_PATH`` wins when set, whether or not it exists: an explicit
    input that points nowhere is the caller's mistake to see, not one to paper
    over. Otherwise the source checkout's ``build/``; ``None`` outside one.

    Loading the extensions does not depend on this: a site-packages
    ``saccade_build.pth`` (written by ``scripts/native/rebuild.sh``) puts the
    build directory on ``sys.path`` for every interpreter, and callers only
    fall back here when a plain import fails.
    """
    explicit = _env_path(BUILD_PATH_ENV)
    if explicit is not None:
        return explicit
    root = source_checkout_root()
    if root is None:
        return None
    return root / "build"


def trackeval_root() -> Path | None:
    """A directory containing the ``trackeval`` package, or ``None``.

    ``SACCADE_TRACKEVAL_ROOT`` wins when set and must actually hold
    ``trackeval/``; otherwise the checkout's ``third_party/TrackEval`` when it
    does. ``None`` means "nothing to add to ``sys.path``", and the caller may
    still find an installed ``trackeval``.
    """
    explicit = _env_path(TRACKEVAL_ROOT_ENV)
    if explicit is not None:
        if not (explicit / "trackeval").is_dir():
            raise ValueError(
                f"{TRACKEVAL_ROOT_ENV}={explicit} does not contain a trackeval/ package"
            )
        return explicit
    root = source_checkout_root()
    if root is None:
        return None
    candidate = root / "third_party" / "TrackEval"
    if (candidate / "trackeval").is_dir():
        return candidate
    return None


def runtime_input(path: str | os.PathLike[str]) -> Path:
    """An external runtime input as an absolute path.

    Relative values resolve against the current working directory -- the same
    rule a CLI flag follows -- and are never re-rooted at the package or the
    repository. The path is not required to exist.
    """
    return Path(path).expanduser().resolve()
