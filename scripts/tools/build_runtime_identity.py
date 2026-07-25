#!/usr/bin/env python3
"""Build the published four-axis `runtime_identity` of the online track.

    decision_surface  resolved parameter snapshot of the sealed preset + the
                      resolved bridge-policy fingerprint
    implementation    git blob digest over the decision-relevant path set
    environment       build recipe + dependency lock + CUDA/TRT/driver versions
    behavior          the § 4.0 policy-visible digest (h2_behavioral_identity)

Everything physical — artifact SHA-256, ELF build-id, loaded library closure,
GPU serial identity — is recorded as **witness** and carries no decision
authority (declaration § 4.1). That inversion is the whole point: H0 spent five
exactly-once authorizations proving physical membership, and the owner's R5
parity audit showed physical identity is not even a function of source (same
source, different build directory, different `sha256` and build-id). An identity
you cannot recompute cannot be published, and an identity nobody can publish has
to be re-derived by every study under a one-shot budget.

Axis bump semantics (declaration § 8.1 / § 8.4):

  * `behavior` unchanged while `implementation` / `environment` changed
    ⇒ behavior-preserving; sealed claims survive. This is what lets the online
      track move without invalidating research.
  * `decision_surface` or `behavior` changed ⇒ decision-affecting; every state
    captured under the old identity is stale until re-attested.

Usage:
  uv run python scripts/tools/build_runtime_identity.py --emit out/identity.json
  uv run python scripts/tools/build_runtime_identity.py \
      --behavior-from out/h2_behavior/result.json --emit docs/reference/runtime_identity.generated.json
  uv run python scripts/tools/build_runtime_identity.py --run-behavior --build-dir build/h2_a
"""
# status: stable

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_path_partition as partition  # noqa: E402
from h2_behavioral_identity import (  # noqa: E402
    POLICY_PRESET_REL,
    POLICY_PRESET_STEM,
    canonical_json_bytes,
    digest,
)

IDENTITY_SCHEMA = "h2_runtime_identity_v1"

# `environment` axis inputs: the recipe that produces the binaries and the lock
# that pins every wheel they load. Deliberately *not* the loaded closure.
ENVIRONMENT_FILES = ("CMakeLists.txt", "pyproject.toml", "uv.lock")


class IdentityError(RuntimeError):
    pass


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


# --------------------------------------------------------------------------- #
# decision_surface                                                             #
# --------------------------------------------------------------------------- #
def decision_surface_axis() -> dict[str, Any]:
    """Resolved parameter snapshot of the sealed preset, plus its policy fingerprint.

    Reuses the existing golden-snapshot resolver (`scripts/eval/config/gen_golden_snapshot.py`)
    so this axis and `tests/fixtures/golden_config_*.json` cannot drift apart, and
    the bridge-policy resolver that `check_headline_decision_contract.py` and the
    declaration bindings already use.
    """
    for extra in (
        REPO_ROOT / "scripts" / "eval",
        REPO_ROOT / "scripts" / "eval" / "config",
        REPO_ROOT / "src",
    ):
        if extra.as_posix() not in sys.path:
            sys.path.insert(0, extra.as_posix())
    from gen_golden_snapshot import _resolve_config
    from resolved_bridge_policy_config import fingerprint

    resolved = _resolve_config(POLICY_PRESET_STEM)
    return {
        "digest": digest(
            {
                "preset": POLICY_PRESET_REL,
                "resolved_bridge_policy_config_v1": fingerprint(POLICY_PRESET_STEM),
                "resolved_parameters": resolved,
            }
        ),
        "parameter_count": len(resolved),
        "preset": POLICY_PRESET_REL,
        "resolved_bridge_policy_config_v1": fingerprint(POLICY_PRESET_STEM),
    }


# --------------------------------------------------------------------------- #
# implementation                                                               #
# --------------------------------------------------------------------------- #
def decision_relevant_files() -> tuple[str, ...]:
    """Every tracked file the partition calls `decision_relevant`.

    Expanded from git rather than from a literal list: a new file under
    `scripts/eval/config/` is decision-relevant the moment it lands, and an
    identity that missed it would be silently narrower than its own definition.
    """
    tracked = [line for line in _git("ls-files").splitlines() if line]
    return tuple(
        sorted(
            path
            for path in tracked
            if partition.classify(path) == "decision_relevant"
            # Prose under a decision-relevant prefix (e.g. `scripts/eval/config/README.md`)
            # stays *classified* decision-relevant — a retry may not edit it — but it is
            # kept out of the axis so a doc edit cannot bump `implementation`. The axis
            # errs toward over-sensitivity elsewhere; there is no reason to add noise
            # that is knowably behavior-irrelevant.
            and not path.endswith((".md", ".rst", ".txt"))
        )
    )


def implementation_axis() -> dict[str, Any]:
    files = decision_relevant_files()
    if not files:
        raise IdentityError("decision-relevant file set is empty")
    members = []
    for path in files:
        blob = _git("hash-object", "--", path)
        if len(blob) != 40:
            raise IdentityError(f"unexpected blob id for {path}: {blob!r}")
        members.append({"blob": blob, "path": path})
    return {
        "digest": digest(members),
        "file_count": len(members),
        "files": members,
    }


# --------------------------------------------------------------------------- #
# environment                                                                  #
# --------------------------------------------------------------------------- #
def _torch_environment() -> dict[str, Any]:
    """Versions, not file identities. Fails soft: absence is recorded, not fatal."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment probe
        return {"available": False, "error": type(exc).__name__}
    info: dict[str, Any] = {
        "available": True,
        "cuda_compiled": getattr(torch.version, "cuda", None),
        "cudnn": None,
        "torch": torch.__version__,
    }
    try:
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception:  # pragma: no cover
        pass
    try:
        if torch.cuda.is_available():
            info["driver_device_count"] = torch.cuda.device_count()
            info["device_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:  # pragma: no cover
        pass
    try:
        import tensorrt

        info["tensorrt"] = tensorrt.__version__
    except Exception:
        info["tensorrt"] = None
    return info


def environment_axis() -> dict[str, Any]:
    recipe = []
    for name in ENVIRONMENT_FILES:
        path = REPO_ROOT / name
        if not path.is_file():
            raise IdentityError(f"environment input is absent: {name}")
        recipe.append({"blob": _git("hash-object", "--", name), "path": name})
    toolchain = _torch_environment()
    return {
        "digest": digest({"recipe": recipe, "toolchain": toolchain}),
        "recipe": recipe,
        "toolchain": toolchain,
    }


# --------------------------------------------------------------------------- #
# witness — recorded, never predicate                                          #
# --------------------------------------------------------------------------- #
def witness(build_dir: Path | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "note": (
            "Witness fields carry no decision authority (declaration § 4.1). No "
            "terminal may be selected on them. Physical artifact identity is not a "
            "function of source: the same source built in a different directory "
            "yields a different sha256 and ELF build-id."
        ),
        "head": _git("rev-parse", "HEAD"),
        "tree": _git("rev-parse", "HEAD^{tree}"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
    }
    if build_dir is None:
        record["build_artifacts"] = None
        return record
    artifacts = []
    for pattern in ("saccade_tracking_ext*.so", "libsaccade_scan_plugin.so"):
        for path in sorted(build_dir.glob(pattern)):
            data = path.read_bytes()
            import hashlib

            artifacts.append(
                {
                    "length": len(data),
                    "path": path.relative_to(REPO_ROOT).as_posix()
                    if path.is_relative_to(REPO_ROOT)
                    else path.as_posix(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    record["build_artifacts"] = artifacts
    record["build_dir"] = build_dir.as_posix()
    return record


# --------------------------------------------------------------------------- #
# assembly                                                                     #
# --------------------------------------------------------------------------- #
def build_identity(
    *,
    behavior: dict[str, Any] | None,
    build_dir: Path | None,
) -> dict[str, Any]:
    axes = {
        "decision_surface": decision_surface_axis(),
        "environment": environment_axis(),
        "implementation": implementation_axis(),
    }
    if behavior is None:
        axes["behavior"] = {
            "digest": None,
            "state": "not_computed",
            "note": (
                "The behavior axis requires one capture-off identity run on a GPU. "
                "Until it is present this identity is incomplete and must not be "
                "published or cited as a substrate coordinate."
            ),
        }
    else:
        axes["behavior"] = behavior
    complete = axes["behavior"].get("digest") is not None
    return {
        "axes": axes,
        "complete": complete,
        "identity": {name: axes[name].get("digest") for name in sorted(axes)},
        "schema": IDENTITY_SCHEMA,
        "witness": witness(build_dir),
    }


def load_behavior(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "h2_behavior_result_v1":
        raise IdentityError(f"{path}: not an h2_behavior_result_v1 payload")
    if not payload.get("identical", False) and int(payload.get("repeats", 1)) > 1:
        raise IdentityError(
            f"{path}: repeats disagreed — a non-reproducible run cannot define an axis"
        )
    if payload.get("digest") is None:
        raise IdentityError(f"{path}: no digest")
    return {
        "digest": payload["digest"],
        "mode": payload.get("mode"),
        "repeats": payload.get("repeats"),
        "sequence": payload.get("sequence"),
        "state": "computed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", type=Path, default=None, help="write JSON identity")
    parser.add_argument(
        "--behavior-from",
        type=Path,
        default=None,
        help="read the behavior axis from an h2_behavioral_identity result",
    )
    parser.add_argument(
        "--run-behavior",
        action="store_true",
        help="run one identity-mode behavior pass now (needs a GPU)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="record physical build artifacts from this directory as witness",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="exit nonzero unless the behavior axis is present",
    )
    args = parser.parse_args(argv)

    behavior: dict[str, Any] | None = None
    try:
        if args.behavior_from and args.run_behavior:
            parser.error("--behavior-from and --run-behavior are mutually exclusive")
        if args.behavior_from:
            behavior = load_behavior(args.behavior_from)
        elif args.run_behavior:
            from h2_behavioral_identity import IDENTITY_SEQUENCE, run_behavior_inventory

            result = run_behavior_inventory(
                sequence=IDENTITY_SEQUENCE,
                identity_mode=True,
                output_dir=REPO_ROOT / "out" / "h2_behavior" / "identity",
            )
            behavior = {
                "digest": result["digest"],
                "mode": result["mode"],
                "repeats": 1,
                "sequence": result["sequence"],
                "state": "computed",
            }
        identity = build_identity(behavior=behavior, build_dir=args.build_dir)
    except IdentityError as exc:
        print(f"runtime identity failed: {exc}", file=sys.stderr)
        return 1

    payload = canonical_json_bytes(identity) + b"\n"
    if args.emit:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_bytes(payload)
        print(f"wrote {args.emit}")
    else:
        sys.stdout.write(payload.decode("utf-8"))

    for name, value in identity["identity"].items():
        print(f"  {name:18} {value}")
    if args.require_complete and not identity["complete"]:
        print("behavior axis absent — identity incomplete", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
