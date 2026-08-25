#!/usr/bin/env python3
"""Locate the exact jump discontinuities of pooled IDF1 along one bridge-gate axis.

Why this exists instead of another grid sweep
---------------------------------------------
The bridge scale gate admits a pair when ``h_lo <= ratio <= h_hi`` -- the CUDA
kernel rejects only on ``hr < bridge_h_lo || hr > bridge_h_hi``
(``src/tracking/tracker_gpu.cu``), so both bounds are inclusive and a threshold
set exactly to an observed ratio still admits that pair.  It is applied to a
*finite* set of evaluated candidate pairs, so moving a threshold changes nothing
until it crosses one of the ratio values actually observed in the data, at which
point decisions flip and the metric jumps.  The metric surface is therefore
**piecewise constant with jumps**, not a smooth ridge, and a grid sweep reports
an arbitrary sampling of a step function: the "optimum" is a grid artefact and
"distance to the nearest bad cell" carries no stability information (this is
what made the old ``nearest_unsafe_distance=1`` result vacuous).

The right objects are the **plateau** you operate on and the **jump** at its
edge.  This tool finds them by bisection rather than by sampling.

Bisection is available here because the underlying measurement is bit-exact
(``reid_mode: off`` + ``--no-gpu-decode``, so N=1 suffices), which makes
*equality* a usable predicate.  A bracket whose endpoints differ can then be
narrowed to any width in O(log(1/tol)) evaluations instead of O(1/tol) grid
points.  Endpoints are compared on the **per-sequence (idtp, idfp, idfn)
vector**, not pooled IDF1: different decision sets can collide on that scalar.

Measurements are cached under ``--work-dir`` and bound to a context fingerprint
(source tree by content, the native extensions the eval itself would import,
the whole child environment, the preset and every model/engine file it names,
every dataset byte, interpreter/torch/GPU): a mismatch fails closed rather than
mixing two measurement regimes, so point ``--work-dir`` somewhere fresh.

Limits (these are properties of the method, not of the implementation)
----------------------------------------------------------------------
1. ``f(a) == f(b)`` does **not** prove there is no jump inside ``(a, b)`` -- two
   jumps can cancel.  Every "plateau" below means *no jump detected at the scan
   resolution*, never *no jump exists*.
2. If a scan bracket contains several jumps, bisection converges to one of them
   and the others are missed; raise ``--scan`` to separate them.  Each reported
   breakpoint carries the ``reason`` it is or is not narrowed to ``--tol``.
3. Breakpoint locations are **data values**, i.e. a property of this dataset's
   ratio distribution rather than of the gate.  Do not expect them to transfer
   to another dataset; see the 2026-08-08 cross-dataset note.
4. Bit-exactness is a premise, not a finding: the tool refuses a preset that is
   not explicitly reid-off, and pins ``--reid-mode off`` on the eval command.

Usage: one axis per invocation
------------------------------
  # --dry-run prints these commands and exits, touching neither eval nor cache
  .venv/bin/python scripts/eval/diagnostics/bridge_gate_breakpoints.py \
      --axis h_hi --lo 1.2 --hi 1.8 --scan 7 --tol 1e-3 \
      --px 0.4 --h-lo 0.6 \
      --preset mamba_whole_graph_m --detector SDP \
      --work-dir out/bp_h_hi

  # dataset without a detector suffix (MOT20 / DanceTrack)
  ... --data-root datasets/MOT20/MOT20 --split train   # and omit --detector
"""
# status: diagnostic

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Eval occasionally dies inside CUDA-graph capture (DALI touching the stream
# mid-capture).  It is transient and a plain re-run clears it, so a bisection
# should not lose fifteen completed evaluations to one of them.
TRANSIENT_MARKERS = (
    "cudaErrorStreamCaptureInvalidated",
    "cudaErrorStreamCaptureUnsupported",
    "operation failed due to a previous error during capture",
    "operation not permitted when stream is capturing",
)
AXES = ("px", "h_lo", "h_hi")
FLAG = {
    "px": "--relink-bridge-px",
    "h_lo": "--relink-bridge-h-lo",
    "h_hi": "--relink-bridge-h-hi",
}

# Flags this tool owns and therefore refuses to let --eval-arg carry.  mot17.py
# parses with the argparse default (last occurrence wins), so a second copy
# appended after ours would silently decide the run while the report kept
# labelling the point with the value *we* asked for -- a measurement filed under
# the wrong coordinates.  --reid-mode and the decode flags are owned at one
# remove: they are what makes the measurement bit-exact, so moving them
# invalidates the equality predicate the whole bisection rests on.
TOOL_OWNED_FLAGS = (
    "--preset",
    "--data-root",
    "--split",
    "--detector",
    "--output",
    "--double-buffer",
    "--gpu-decode",
    "--no-gpu-decode",
    "--reid-mode",
    "--relink-bridge-px",
    "--relink-bridge-h-lo",
    "--relink-bridge-h-hi",
)

CACHE_SCHEMA = 2
# Source that can change what an eval computes.  Digested by content: the git
# head alone is both too weak (dirty tree) and too strong (a docs commit moves
# it without touching a measurement).
SOURCE_ROOTS = ("include", "src", "scripts/eval", "configs")
SOURCE_SUFFIXES = frozenset(
    {".py", ".pyi", ".cu", ".cuh", ".h", ".hpp", ".cc", ".cpp", ".yaml", ".yml"}
)
NATIVE_MODULES = (
    "saccade_tracking_ext",
    "saccade_perception_ext",
    "saccade_eval_ext",
    "saccade_media_ext",
)
# The child environment is bound in full, minus names nothing on the eval path
# can read.  An allowlist would have to be right about every knob that reaches
# the tracker (there are >60 SACCADE_* alone, several of which move boxes and
# scores); a denylist only has to be right about the handful it names, and each
# of these is terminal or session decoration.
ENV_IGNORED_NAMES = frozenset(
    {
        "_",
        "COLORTERM",
        "COLUMNS",
        "LINES",
        "OLDPWD",
        "PWD",
        "SHLVL",
        "TERM",
        "TERM_PROGRAM",
        "TERM_PROGRAM_VERSION",
        "TERM_SESSION_ID",
        "WINDOWID",
    }
)
ENV_IGNORED_PREFIXES = ("SSH_", "TMUX", "CLAUDE", "VSCODE_", "ITERM")

# The probe must resolve the extensions the *eval* will import, so it repeats
# scripts/eval/mot17.py's sys.path prologue verbatim -- SACCADE_BUILD_PATH and
# all.  Resolving them against this process's own sys.path instead reported the
# .so in build/ while the eval loaded a different one.
_PROBE_SRC = """
import importlib.util, json, os, sys

root = sys.argv[2]
sys.path.insert(0, root)
src = os.path.join(root, "src")
if os.path.exists(src):
    sys.path.insert(0, src)
build = os.environ.get("SACCADE_BUILD_PATH", os.path.join(root, "build"))
if os.path.exists(build):
    sys.path.insert(0, build)

out = {
    "executable": sys.executable,
    "python_version": sys.version.split()[0],
    "build_path": build,
}
mods = {}
for name in json.loads(sys.argv[1]):
    try:
        spec = importlib.util.find_spec(name)
        mods[name] = spec.origin if spec is not None else None
    except Exception as exc:  # noqa: BLE001 - reported, not handled
        mods[name] = "unimportable: %s" % (exc,)
out["modules"] = mods
try:
    import torch

    out["torch"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
except Exception as exc:  # noqa: BLE001 - reported, not handled
    out["torch"] = "unimportable: %s" % (exc,)
    out["torch_cuda"] = None
json.dump(out, sys.stdout)
"""


class CacheContextMismatch(RuntimeError):
    """The cache under --work-dir was measured under a different context."""


@dataclass
class Measurement:
    """One evaluated threshold value."""

    value: float
    idf1: float
    per_seq: dict[str, dict[str, int]]
    out_dir: str

    @property
    def fingerprint(self) -> tuple:
        """Bit-exact identity of the decision outcome (see module docstring)."""
        return tuple(
            (seq, c["idtp"], c["idfp"], c["idfn"])
            for seq, c in sorted(self.per_seq.items())
        )

    def per_seq_idf1(self) -> dict[str, float]:
        return {seq: _idf1(c) for seq, c in sorted(self.per_seq.items())}


def _idf1(counts: dict[str, int]) -> float:
    denom = 2 * counts["idtp"] + counts["idfp"] + counts["idfn"]
    return 0.0 if denom == 0 else 100.0 * 2 * counts["idtp"] / denom


def _pooled_idf1(per_seq: dict[str, dict[str, int]]) -> float:
    """IDF1 is a ratio: pool the counts, never average the per-sequence values."""
    total = {"idtp": 0, "idfp": 0, "idfn": 0}
    for counts in per_seq.values():
        for key in total:
            total[key] += counts[key]
    return _idf1(total)


# --------------------------------------------------------------------------
# eval command construction (shared by the real runner and --dry-run)
# --------------------------------------------------------------------------


def params_for(args: argparse.Namespace, value: float) -> dict[str, float]:
    params = {"px": args.px, "h_lo": args.h_lo, "h_hi": args.h_hi}
    params[args.axis] = value
    return params


def out_dir_for(args: argparse.Namespace, params: dict[str, float]) -> Path:
    tag = "_".join(f"{k}{params[k]:.10g}" for k in AXES)
    return Path(args.work_dir) / f"run_{tag}"


def eval_command(
    args: argparse.Namespace, params: dict[str, float], out_dir: Path
) -> list[str]:
    cmd = [
        args.python,
        "scripts/eval/mot17.py",
        "--preset",
        args.preset,
        "--data-root",
        args.data_root,
        "--split",
        args.split,
        "--output",
        str(out_dir),
    ]
    if args.detector:
        cmd += ["--detector", args.detector]
    # --reid-mode off is pinned on the command line, where it outranks every
    # config layer: the bit-exactness the equality predicate rests on is not
    # something to leave to whatever the preset happened to say.  main() also
    # refuses a preset that does not already say off, so the pin never silently
    # re-purposes a run the caller asked for.
    cmd += ["--double-buffer", "--no-gpu-decode", "--reid-mode", "off"]
    for axis in AXES:
        cmd += [FLAG[axis], f"{params[axis]:.10g}"]
    cmd += list(args.eval_arg)
    return cmd


def eval_env() -> dict[str, str]:
    """The one child environment used for every subprocess this tool starts.

    Built once and shared by the runtime probe and the eval runs, so what was
    fingerprinted is what ran.
    """
    env = dict(os.environ)
    torch_lib = REPO_ROOT / ".venv/lib/python3.12/site-packages/torch/lib"
    if torch_lib.is_dir():
        env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(
            ":"
        )
    return env


def _env_component(env: dict[str, str]) -> dict:
    """Bind the whole child environment; several SACCADE_* knobs move boxes.

    Values are digested rather than listed -- an environment carries
    credentials -- except SACCADE_* overrides, which are what a reader needs to
    see and are never secrets.
    """
    bound = {
        name: value
        for name, value in env.items()
        if name not in ENV_IGNORED_NAMES and not name.startswith(ENV_IGNORED_PREFIXES)
    }
    return {
        "bound_variables": len(bound),
        "ignored": sorted(set(env) - set(bound)),
        "saccade_overrides": {
            k: v for k, v in sorted(bound.items()) if k.startswith("SACCADE_")
        },
        "digest": _digest(sorted(bound.items())),
    }


def rejected_eval_args(eval_args: list[str]) -> list[tuple[str, str]]:
    """Tokens in --eval-arg that could resolve to a flag this tool owns.

    mot17.py's parser leaves ``allow_abbrev`` on, so ``--relink-bridge-h-h`` is
    just as effective an override as the full flag; the check is therefore on
    prefixes, not on exact names.
    """
    bad: list[tuple[str, str]] = []
    for token in eval_args:
        name = token.split("=", 1)[0]
        if not name.startswith("--"):
            continue
        for owned in TOOL_OWNED_FLAGS:
            if name == owned or owned.startswith(name):
                bad.append((token, owned))
                break
    return bad


def resolve_reid_mode(args: argparse.Namespace) -> tuple[str, str]:
    """Return (reid_mode, where it was read from) for the preset being measured."""
    import yaml

    preset_path = REPO_ROOT / "configs" / "presets" / f"{args.preset}.yaml"
    if not preset_path.is_file():
        return "", f"no such preset: {preset_path}"
    loaded = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    if "reid_mode" not in loaded:
        return "", f"{preset_path.relative_to(REPO_ROOT)} does not set reid_mode"
    return str(loaded["reid_mode"]), str(preset_path.relative_to(REPO_ROOT))


def premise_violation(args: argparse.Namespace) -> str | None:
    """Why this run cannot support bisection, or None.

    Bisection here is licensed by one thing: the measurement is bit-exact, so
    N=1 suffices and *equality* is a usable predicate. That holds with ReID off;
    with ReID doing appearance work it does not, and every plateau and jump the
    tool reported would be an artefact of run-to-run variation. The tool has no
    way to tell that apart after the fact, so it refuses up front rather than
    publishing numbers whose premise is false.
    """
    mode, origin = resolve_reid_mode(args)
    if mode != "off":
        return (
            f"--preset {args.preset} resolves to reid_mode={mode or '<unset>'} "
            f"({origin}); this tool requires an explicitly reid-off preset, "
            f"because bisection on a non-bit-exact metric measures noise"
        )
    return None


# --------------------------------------------------------------------------
# context fingerprint
# --------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _source_component() -> dict:
    digest = hashlib.sha256()
    files = 0
    for root in SOURCE_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
                continue
            digest.update(str(path.relative_to(REPO_ROOT)).encode())
            digest.update(_sha256_file(path).encode())
            files += 1
    return {"roots": list(SOURCE_ROOTS), "files": files, "digest": digest.hexdigest()}


def _file_digests(paths: list[Path]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for path in paths:
        key = str(path if path.is_absolute() else REPO_ROOT / path)
        out[key] = _sha256_file(path) if path.is_file() else None
    return out


def _config_component(args: argparse.Namespace) -> dict:
    import yaml

    preset_path = REPO_ROOT / "configs" / "presets" / f"{args.preset}.yaml"
    referenced: list[Path] = []
    if preset_path.is_file():
        loaded = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
        for value in loaded.values():
            if not isinstance(value, str) or not value:
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            if candidate.is_file():
                referenced.append(candidate)
    # Anything the caller forwarded that names a file is part of the config too.
    for token in args.eval_arg:
        raw = token.split("=", 1)[-1]
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if raw and candidate.is_file():
            referenced.append(candidate)
    return {
        "preset": args.preset,
        "preset_sha256": _sha256_file(preset_path) if preset_path.is_file() else None,
        "referenced_files": _file_digests(sorted(set(referenced))),
        "extra_eval_args": list(args.eval_arg),
    }


def _dataset_component(args: argparse.Namespace) -> dict:
    root = Path(args.data_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    split_dir = root / args.split
    digest = hashlib.sha256()
    sequences = 0
    files = 0
    payload = 0
    dirs = (
        sorted(p for p in split_dir.iterdir() if p.is_dir())
        if split_dir.is_dir()
        else []
    )
    # The bound unit is the whole sequence directory -- frames, ground truth,
    # public detections, labels, seqinfo -- digested by content.  Picking out
    # "the files that matter" is the judgement call that let a repainted frame
    # through when frames were bound by (name, size).
    for seq in dirs:
        if args.detector and not seq.name.endswith(args.detector):
            continue
        sequences += 1
        digest.update(seq.name.encode())
        for path in sorted(seq.rglob("*")):
            if not path.is_file():
                continue
            files += 1
            payload += path.stat().st_size
            digest.update(str(path.relative_to(seq)).encode())
            digest.update(_sha256_file(path).encode())
    return {
        "split_dir": str(split_dir),
        "detector": args.detector,
        "sequences": sequences,
        "files": files,
        "bytes": payload,
        "digest": digest.hexdigest(),
        "coverage": "sha256 of every file under each matched sequence directory",
    }


def _gpu_identity() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"
    if proc.returncode != 0:
        return f"unavailable: nvidia-smi exit {proc.returncode}"
    return " | ".join(line.strip() for line in proc.stdout.strip().splitlines())


def _runtime_component(args: argparse.Namespace, env: dict[str, str]) -> dict:
    proc = subprocess.run(
        [
            args.python,
            "-c",
            _PROBE_SRC,
            json.dumps(list(NATIVE_MODULES)),
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"runtime probe failed ({proc.returncode}): {proc.stderr.strip()[-2000:]}"
        )
    info = json.loads(proc.stdout)
    natives: dict[str, dict] = {}
    for name, origin in sorted(info["modules"].items()):
        path = Path(origin) if origin else None
        natives[name] = {
            "origin": origin,
            "sha256": _sha256_file(path)
            if path is not None and path.is_file()
            else None,
        }
    return {
        "interpreter": info["executable"],
        "python_version": info["python_version"],
        "build_path": info["build_path"],
        "torch": info["torch"],
        "torch_cuda": info["torch_cuda"],
        "native_extensions": natives,
        "gpu": _gpu_identity(),
    }


def context_fingerprint(args: argparse.Namespace, env: dict[str, str]) -> dict:
    """Everything a cached measurement is only valid under.

    ``env`` must be the same environment the eval runs get, or the fingerprint
    describes a run that never happened.

    ``bound`` is what the cache is checked against; ``informational`` is
    recorded for the reader and deliberately excluded, because a git head moves
    for reasons that cannot change a measurement.
    """
    bound = {
        "source": _source_component(),
        "config": _config_component(args),
        "dataset": _dataset_component(args),
        "environment": _env_component(env),
        "runtime": _runtime_component(args, env),
    }
    return {
        "schema": CACHE_SCHEMA,
        "bound": bound,
        "component_digests": {name: _digest(part) for name, part in bound.items()},
        "digest": _digest(bound),
        "informational": {"git_head": _git_head()},
    }


def _git_head() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"
    return proc.stdout.strip() if proc.returncode == 0 else "unavailable"


class Runner:
    """Runs eval at a threshold value and scores it, with an on-disk cache."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.work_dir = Path(args.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.work_dir / "cache.json"
        self.env = eval_env()
        self.context = context_fingerprint(args, self.env)
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            stored = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self._check_context(stored)
            self.cache = stored["entries"]
        self.runs = 0
        # Every point measured this session, fresh or served from cache.  The
        # report is derived from all of them, never from the scan grid alone.
        self.seen: dict[float, Measurement] = {}

    def _check_context(self, stored: dict) -> None:
        """Refuse a cache measured under a different context (fail closed)."""
        if not isinstance(stored, dict) or stored.get("schema") != CACHE_SCHEMA:
            raise CacheContextMismatch(
                f"{self.cache_path} was written by a different version of this "
                f"tool and carries no context fingerprint; its numbers cannot be "
                f"shown to be comparable. Use a fresh --work-dir."
            )
        old = stored.get("context", {})
        if old.get("digest") == self.context["digest"]:
            return
        changed = [
            name
            for name, digest in self.context["component_digests"].items()
            if old.get("component_digests", {}).get(name) != digest
        ]
        raise CacheContextMismatch(
            f"{self.cache_path} was measured under a different context "
            f"(changed: {', '.join(changed) or 'unknown'}). Cached and fresh "
            f"numbers would not be comparable. Use a fresh --work-dir."
        )

    def _key(self, value: float) -> str:
        params = params_for(self.args, value)
        return json.dumps(
            {
                "preset": self.args.preset,
                "data_root": self.args.data_root,
                "split": self.args.split,
                "detector": self.args.detector,
                "eval_arg": self.args.eval_arg,
                **{k: f"{v:.10g}" for k, v in params.items()},
            },
            sort_keys=True,
        )

    def adopt_cached_range(self, lo: float, hi: float) -> int:
        """Pull every already-measured point in [lo, hi] into this report.

        The cache outlives a single invocation, so a targeted refinement run
        leaves behind points that a later full-range run would never visit. Not
        adopting them makes the report understate what has actually been
        measured -- it would re-label an already-bisected jump UNREFINED.

        Only reachable once the context check above has passed, so every point
        adopted here was measured under this exact context.
        """
        adopted = 0
        for key, hit in self.cache.items():
            entry = json.loads(key)
            if entry.get(self.args.axis) is None:
                continue
            value = float(entry[self.args.axis])
            if not lo <= value <= hi or value in self.seen:
                continue
            same_context = all(
                entry.get(field) == expected
                for field, expected in (
                    ("preset", self.args.preset),
                    ("data_root", self.args.data_root),
                    ("split", self.args.split),
                    ("detector", self.args.detector),
                    ("eval_arg", self.args.eval_arg),
                )
            )
            fixed_match = all(
                entry.get(axis) == f"{getattr(self.args, axis):.10g}"
                for axis in AXES
                if axis != self.args.axis
            )
            if not (same_context and fixed_match):
                continue
            self.seen[value] = Measurement(
                value, hit["idf1"], hit["per_seq"], hit["out_dir"]
            )
            adopted += 1
        return adopted

    def measure(self, value: float) -> Measurement:
        key = self._key(value)
        if key in self.cache:
            hit = self.cache[key]
            cached = Measurement(value, hit["idf1"], hit["per_seq"], hit["out_dir"])
            self.seen[value] = cached
            return cached

        params = params_for(self.args, value)
        out_dir = out_dir_for(self.args, params)
        self._run_eval(params, out_dir)
        per_seq = self._score(out_dir)
        if not per_seq:
            raise RuntimeError(f"no scored sequences in {out_dir}")
        measurement = Measurement(value, _pooled_idf1(per_seq), per_seq, str(out_dir))
        self.cache[key] = {
            "idf1": measurement.idf1,
            "per_seq": per_seq,
            "out_dir": str(out_dir),
        }
        self._save_cache()
        self.runs += 1
        self.seen[value] = measurement
        return measurement

    def _save_cache(self) -> None:
        self.cache_path.write_text(
            json.dumps(
                {
                    "schema": CACHE_SCHEMA,
                    "context": self.context,
                    "entries": self.cache,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _run_eval(self, params: dict[str, float], out_dir: Path) -> None:
        cmd = eval_command(self.args, params, out_dir)
        env = self.env

        print(f"  run {' '.join(cmd[-8:])}", file=sys.stderr, flush=True)
        for attempt in range(1, self.args.retries + 2):
            suffix = "" if attempt == 1 else f".attempt{attempt}"
            log_path = out_dir.parent / f"{out_dir.name}{suffix}.log"
            # A retry must not score leftovers from the failed attempt.
            if out_dir.exists():
                for stale in out_dir.glob("*.txt"):
                    stale.unlink()
            out_dir.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if proc.returncode == 0:
                return

            tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
            transient = next((m for m in TRANSIENT_MARKERS if m in tail), None)
            if transient is None or attempt > self.args.retries:
                raise RuntimeError(f"eval failed ({proc.returncode}); see {log_path}")
            print(
                f"    transient failure ({transient}); "
                f"retry {attempt}/{self.args.retries}",
                file=sys.stderr,
                flush=True,
            )

    def _score(self, out_dir: Path) -> dict[str, dict[str, int]]:
        """Recompute IDF1 counts at full precision from the saved MOT output."""
        src = str(REPO_ROOT / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from saccade.perception.eval.metrics import _evaluate_single_sequence

        gt_root = Path(self.args.data_root) / self.args.split
        per_seq: dict[str, dict[str, int]] = {}
        for ts_path in sorted(out_dir.glob("*.txt")):
            seq = ts_path.stem
            if seq.startswith("_"):
                continue
            gt_path = gt_root / seq / "gt" / "gt.txt"
            if not gt_path.exists():
                continue
            summary = _evaluate_single_sequence(seq, str(gt_path), str(ts_path))
            per_seq[seq] = {
                "idtp": int(summary["idtp"]),
                "idfp": int(summary["idfp"]),
                "idfn": int(summary["idfn"]),
            }
        return per_seq


@dataclass
class Bracket:
    low: Measurement
    high: Measurement
    refined: bool = False
    attempted: bool = False
    history: list[float] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.high.value - self.low.value


def scan(runner: Runner, lo: float, hi: float, points: int) -> list[Measurement]:
    if points < 2:
        raise ValueError("--scan must be at least 2")
    step = (hi - lo) / (points - 1)
    values = [lo + i * step for i in range(points)]
    out = []
    for i, value in enumerate(values, 1):
        print(f"[scan {i}/{points}] {value:.6g}", file=sys.stderr, flush=True)
        out.append(runner.measure(value))
    return out


def brackets_from_scan(samples: list[Measurement]) -> list[Bracket]:
    return [
        Bracket(a, b)
        for a, b in zip(samples, samples[1:])
        if a.fingerprint != b.fingerprint
    ]


def bisect(runner: Runner, bracket: Bracket, tol: float) -> Bracket:
    bracket.attempted = True
    low, high = bracket.low, bracket.high
    while high.value - low.value > tol:
        mid_value = 0.5 * (low.value + high.value)
        if mid_value <= low.value or mid_value >= high.value:
            break  # float resolution exhausted
        print(
            f"  [bisect] ({low.value:.8g}, {high.value:.8g}) -> {mid_value:.8g}",
            file=sys.stderr,
            flush=True,
        )
        mid = runner.measure(mid_value)
        bracket.history.append(mid_value)
        if mid.fingerprint == low.fingerprint:
            low = mid
        else:
            high = mid
    bracket.low, bracket.high, bracket.refined = low, high, True
    return bracket


def _refinement_reason(
    low: Measurement,
    high: Measurement,
    args: argparse.Namespace,
    bisected_edges: set[tuple[float, float]],
    unfinished: list[Bracket],
    aborted: str | None,
) -> str:
    """Say why this gap is (or is not) narrowed to --tol -- never guess.

    'Wider than --tol' has several distinct causes, and reporting the most
    interesting one for all of them (a scan bracket holding several jumps) is a
    claim about the data that the run may not have made.
    """
    width = high.value - low.value
    bisected = (low.value, high.value) in bisected_edges
    if width <= args.tol:
        if bisected:
            return "bisected to <= --tol"
        return "already within --tol at the scan resolution"
    if bisected:
        return "bisection stopped early: float resolution exhausted before --tol"
    if args.scan_only:
        return "not bisected: --scan-only"
    enclosing = next(
        (
            b
            for b in unfinished
            if b.low.value <= low.value and high.value <= b.high.value
        ),
        None,
    )
    if enclosing is not None:
        # A bracket that was being bisected when the run died has measured
        # points inside it and is not the same story as one nobody reached.
        if enclosing.attempted:
            detail = f" ({aborted})" if aborted is not None else ""
            return f"not narrowed to --tol: bisection of this bracket stopped part-way{detail}"
        if aborted is not None:
            return f"not bisected: run aborted before reaching this bracket ({aborted})"
        return "not bisected: bisection did not reach this bracket"
    return (
        "residual gap: the enclosing scan bracket held more than one jump and "
        "bisection isolated a different one -- raise --scan to separate them"
    )


def build_report(
    args: argparse.Namespace,
    seen: list[Measurement],
    runs: int,
    brackets: list[Bracket] | None = None,
    aborted: str | None = None,
    context: dict | None = None,
) -> dict:
    """Derive plateaus and jumps from *every* measured point.

    Deriving them from the scan grid alone lets the report assert a plateau that
    the bisection's own midpoints refute -- the tool would be claiming more than
    it measured.  Grouping all measured points by fingerprint cannot do that: a
    plateau is exactly a maximal run of consecutive points that behaved
    identically, and a jump is exactly the gap between two such runs.
    """
    brackets = list(brackets or [])
    bisected_edges = {(b.low.value, b.high.value) for b in brackets if b.refined}
    unfinished = [b for b in brackets if not b.refined]

    points = sorted(seen, key=lambda m: m.value)
    groups: list[list[Measurement]] = []
    for point in points:
        if groups and groups[-1][0].fingerprint == point.fingerprint:
            groups[-1].append(point)
        else:
            groups.append([point])

    plateaus = [
        {
            "from": g[0].value,
            "to": g[-1].value,
            "width": g[-1].value - g[0].value,
            "idf1": g[0].idf1,
            "per_seq_idf1": g[0].per_seq_idf1(),
            "measured_points": len(g),
        }
        for g in groups
    ]

    breakpoints = []
    for below, above in zip(groups, groups[1:]):
        low, high = below[-1], above[0]
        breakpoints.append(
            {
                "lower_bound": low.value,
                "upper_bound": high.value,
                "width": high.value - low.value,
                "idf1_below": low.idf1,
                "idf1_above": high.idf1,
                "delta_idf1": high.idf1 - low.idf1,
                "per_seq_delta_idf1": {
                    seq: high.per_seq_idf1()[seq] - value
                    for seq, value in low.per_seq_idf1().items()
                },
                "refined": (high.value - low.value) <= args.tol,
                "reason": _refinement_reason(
                    low, high, args, bisected_edges, unfinished, aborted
                ),
            }
        )

    report = {
        "tool": "bridge_gate_breakpoints",
        "schema_version": 2,
        "axis": args.axis,
        "fixed_axes": {a: getattr(args, a) for a in AXES if a != args.axis},
        "dataset": {
            "preset": args.preset,
            "data_root": args.data_root,
            "split": args.split,
            "detector": args.detector,
            "extra_eval_args": list(args.eval_arg),
        },
        "equality_predicate": "per-sequence (idtp, idfp, idfn) vector",
        "gate_contract": "h_lo <= ratio <= h_hi (both bounds inclusive)",
        "scan": {"lo": args.lo, "hi": args.hi, "points": args.scan},
        "measurements": [
            {"value": m.value, "idf1": m.idf1, "out_dir": m.out_dir} for m in points
        ],
        "tolerance": args.tol,
        "eval_runs_executed": runs,
        "breakpoints": breakpoints,
        "plateaus": plateaus,
        "limits": [
            "equal endpoints do not prove the absence of a jump between them; "
            "plateaus mean 'no jump detected at the scan resolution'",
            "a jump wider than --tol is not by itself evidence of anything: "
            "each breakpoint carries a 'reason' saying why it was not narrowed",
            "breakpoint locations are data values, so they are a property of "
            "this dataset's feature distribution, not of the gate",
        ],
    }
    if context is not None:
        report["context"] = context
    return report


def print_table(report: dict) -> None:
    print(f"\naxis={report['axis']}  fixed={report['fixed_axes']}")
    print(f"eval runs executed: {report['eval_runs_executed']}")

    print("\nplateaus (identical outcome across every measured point inside):")
    print(f"  {'from':>12}  {'to':>12}  {'width':>10}  {'IDF1':>9}  {'pts':>4}")
    for p in report["plateaus"]:
        print(
            f"  {p['from']:>12.7g}  {p['to']:>12.7g}  "
            f"{p['width']:>10.4g}  {p['idf1']:>9.3f}  {p['measured_points']:>4d}"
        )

    print("\nbreakpoints (jump localised to):")
    if not report["breakpoints"]:
        print("  none detected at this scan resolution")
    for b in report["breakpoints"]:
        print(
            f"  ({b['lower_bound']:.8g}, {b['upper_bound']:.8g}]  "
            f"width={b['width']:.3g}  "
            f"IDF1 {b['idf1_below']:.3f} -> {b['idf1_above']:.3f}  "
            f"(delta {b['delta_idf1']:+.3f})"
        )
        print(f"      {'refined' if b['refined'] else 'UNREFINED'}: {b['reason']}")
        worst = sorted(b["per_seq_delta_idf1"].items(), key=lambda kv: kv[1])[:2]
        for seq, delta in worst:
            if abs(delta) >= 0.05:
                print(f"      {seq}: {delta:+.3f}")

    print("\nlimits:")
    for limit in report["limits"]:
        print(f"  - {limit}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--axis", choices=AXES, required=True, help="Threshold to vary.")
    p.add_argument("--lo", type=float, required=True, help="Scan range lower end.")
    p.add_argument("--hi", type=float, required=True, help="Scan range upper end.")
    p.add_argument(
        "--scan", type=int, default=7, help="Initial scan points (>=2). Default 7."
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Bisect until each jump is bracketed to this width. Default 1e-3.",
    )
    p.add_argument("--px", type=float, default=0.4, help="relink_bridge_px.")
    p.add_argument("--h-lo", dest="h_lo", type=float, default=0.6)
    p.add_argument("--h-hi", dest="h_hi", type=float, default=1.7)
    p.add_argument("--preset", default="mamba_whole_graph_m")
    p.add_argument("--data-root", default="datasets/MOT17")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--detector",
        default="SDP",
        help="Detector suffix. Pass empty for MOT20/DanceTrack.",
    )
    p.add_argument(
        "--eval-arg",
        action="append",
        default=[],
        help=(
            "Extra argument forwarded verbatim to mot17.py (repeatable). Use "
            "the equals form for values starting with '-', e.g. "
            "--eval-arg=--seq-limit --eval-arg=2. Flags this tool owns "
            "(gate/preset/dataset/output/decode/reid-mode) are rejected."
        ),
    )
    p.add_argument("--work-dir", required=True, help="Run outputs and result cache.")
    p.add_argument("--json", help="Write the full report here.")
    p.add_argument("--python", default=".venv/bin/python")
    p.add_argument(
        "--retries",
        type=int,
        default=2,
        help=(
            "Re-run an eval this many times when it dies with a known "
            "transient CUDA-graph capture error. Default 2."
        ),
    )
    p.add_argument(
        "--scan-only",
        action="store_true",
        help="Find brackets but skip bisection.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the planned scan-grid eval commands and exit. Runs nothing, "
            "reads nothing from the cache and writes nothing to it."
        ),
    )
    args = p.parse_args(argv)
    bad = rejected_eval_args(args.eval_arg)
    if bad:
        p.error(
            "--eval-arg may not carry flags this tool owns: "
            + "; ".join(f"{token!r} resolves to {owned}" for token, owned in bad)
        )
    return args


def dry_run(args: argparse.Namespace) -> int:
    """Print the scan-grid commands. No eval, no cache, no report.

    A report would have to invent measurements, and the bisection's own
    evaluations cannot be planned at all: which midpoints get measured is
    decided by what the earlier measurements say.
    """
    step = (args.hi - args.lo) / (args.scan - 1)
    print(f"# {args.scan} scan point(s) on --axis {args.axis}; nothing is executed")
    for i in range(args.scan):
        value = args.lo + i * step
        params = params_for(args, value)
        cmd = eval_command(args, params, out_dir_for(args, params))
        print(f"[scan {i + 1}/{args.scan}] {value:.6g}")
        print("  " + " ".join(cmd))
    print(
        "# bisection evaluations depend on these results and cannot be listed "
        "in advance"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.hi <= args.lo:
        print("--hi must exceed --lo", file=sys.stderr)
        return 2
    if args.scan < 2:
        print("--scan must be at least 2", file=sys.stderr)
        return 2
    violation = premise_violation(args)
    if violation is not None:
        print(f"REFUSING TO MEASURE: {violation}", file=sys.stderr)
        return 2
    if args.dry_run:
        return dry_run(args)

    try:
        runner = Runner(args)
    except CacheContextMismatch as exc:
        print(f"REFUSING CACHE: {exc}", file=sys.stderr)
        return 2

    samples = scan(runner, args.lo, args.hi, args.scan)
    brackets = brackets_from_scan(samples)
    print(f"\n{len(brackets)} bracket(s) contain a jump", file=sys.stderr, flush=True)

    failure: str | None = None
    if not args.scan_only:
        for i, bracket in enumerate(brackets, 1):
            print(f"[refine {i}/{len(brackets)}]", file=sys.stderr, flush=True)
            try:
                bisect(runner, bracket, args.tol)
            except RuntimeError as exc:
                # Do not throw away the brackets already refined: report what is
                # known, leave the rest unrefined, and fail loudly at the end.
                failure = str(exc)
                print(f"  ABORTED: {exc}", file=sys.stderr, flush=True)
                break

    adopted = runner.adopt_cached_range(args.lo, args.hi)
    if adopted:
        print(
            f"adopted {adopted} previously measured point(s) from the cache",
            file=sys.stderr,
        )

    report = build_report(
        args,
        list(runner.seen.values()),
        runner.runs,
        brackets=brackets,
        aborted=failure,
        context=runner.context,
    )
    if failure is not None:
        report["incomplete"] = failure
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.json}", file=sys.stderr)
    print_table(report)
    if failure is not None:
        print(f"\nINCOMPLETE: {failure}", file=sys.stderr)
        print("cached results are kept; re-run to resume", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
