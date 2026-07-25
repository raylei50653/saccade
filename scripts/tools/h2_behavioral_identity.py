#!/usr/bin/env python3
"""Compute the H2 `behavior` axis: a policy-visible digest of one eval run.

The axis digests exactly the four capture-off-computable A7.6 members
(declaration § 4.0), in canonical key order:

    active_tid_slot_pairs   per frame, sorted by slot
    final_track_rows        frame, row index, raw binary32 bits, class, track ID
    mot_output              complete MOT file bytes: length + SHA-256
    relink_debug_raw        the complete 13-integer vector

These are the members A7.6 requires to be equal between capture-off and every
capture-on run — the policy-visible state. `proposal_projection` and
`winner_commit_projection` are trace-only and cannot exist capture-off (A7.6
forbids fabricating them); `overflow_vector` is a capture-on zero predicate. So
this runs **capture-off**, and needs no research instrumentation enabled.

Why not import `run_h0_phase_a_child.py`: that module is hash-pinned inside
historical H0 freeze artifacts (`verify_h0_preseal_freeze.IMPLEMENTATIONS`) and
is hardwired to MOT17-04-SDP, the m preset, a sanitized environment, and a
single sealed invocation. H2 reimplements the same instrumentation technique
parameterized, and edits no frozen plumbing.

What this tool is for, and what it is not:

  * **Change detector.** In identity mode (`--identity-mode`) the known
    nondeterminism sources are pinned off, so a change to decision-relevant code
    provably moves the digest. This is *not* a claim that identity mode
    reproduces production behavior — treating it as one would repeat the
    `R`-operator error the fidelity protocol forbids.
  * **Repeat-equality probe.** With `--repeats N --require-identical` on the
    production policy target it answers, cheaply and before any seal, whether
    production is byte-reproducible at all (gate G2).

Usage:
  uv run python scripts/tools/h2_behavioral_identity.py --identity-mode
  uv run python scripts/tools/h2_behavioral_identity.py \
      --sequences MOT17-04-SDP --repeats 3 --require-identical
  uv run python scripts/tools/h2_behavioral_identity.py --identity-mode \
      --emit /tmp/behavior.json
"""
# status: stable

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

BEHAVIOR_SCHEMA = "h2_behavior_inventory_v1"

# Declaration § 3.1 / H0 Amendment 5. The identity axis and the measurement share
# one policy target; only the fixture and the determinism pinning differ.
POLICY_PRESET_REL = "configs/presets/mamba_whole_graph_m.yaml"
POLICY_PRESET_STEM = "mamba_whole_graph_m"

# Declaration § 3.2. Shortest 7-seq member; owner-overridable.
IDENTITY_SEQUENCE = "MOT17-09-SDP"

# Declaration § 4.0 — the exact member set, frozen.
BEHAVIOR_MEMBERS = (
    "active_tid_slot_pairs",
    "final_track_rows",
    "mot_output",
    "relink_debug_raw",
)


class BehavioralIdentityError(RuntimeError):
    """Fail closed: never emit a digest we cannot stand behind."""


def canonical_json_bytes(value: object) -> bytes:
    """H0's `h0_phase_a_execution_v1` convention, consumed unchanged."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _binary32_bits(value: float) -> int:
    return struct.unpack("!I", struct.pack("!f", value))[0]


def resolve_build_dir() -> Path:
    """The build directory under test.

    `SACCADE_BUILD_PATH` is honoured by `scripts/eval/mot17.py` only; a
    site-packages `.pth` (`saccade_build.pth`) pins `<repo>/build` into
    `sys.path` for every interpreter. Without an explicit override the env var
    is silently ignored and every run loads the same extension — which is
    exactly how a two-build comparison can appear to pass while testing one
    binary twice. G1 is untestable unless the build directory is selectable.
    """
    return Path(
        os.environ.get("SACCADE_BUILD_PATH", (REPO_ROOT / "build").as_posix())
    ).resolve()


def _assert_extension_consumed(build_dir: Path) -> dict[str, Any]:
    """Prove the loaded extension is the one in *build_dir*, and record it.

    This is the cheap in-process form of H0's "operations prove runtime
    consumption" predicate: it does not enumerate any closure, it just asks the
    interpreter which file it actually imported. R5 lost its authorization at a
    load-membership check whose record was never persisted; this one is checked
    and recorded before any digest is emitted.
    """
    module = sys.modules.get("saccade_tracking_ext")
    if module is None or not getattr(module, "__file__", None):
        raise BehavioralIdentityError("saccade_tracking_ext was never imported")
    loaded = Path(module.__file__).resolve()
    if loaded.parent != build_dir:
        raise BehavioralIdentityError(
            f"extension under test is {build_dir}, but the interpreter loaded "
            f"{loaded} — the digest would not describe the requested build"
        )
    data = loaded.read_bytes()
    return {
        "build_dir": build_dir.as_posix(),
        "extension_length": len(data),
        "extension_path": loaded.as_posix(),
        "extension_sha256": hashlib.sha256(data).hexdigest(),
    }


def _import_eval_stack() -> tuple[Any, ...]:
    # The build directory must win over the `.pth`-injected default.
    build_dir = resolve_build_dir()
    if not build_dir.is_dir():
        raise BehavioralIdentityError(f"build directory is absent: {build_dir}")
    if sys.path and sys.path[0] != build_dir.as_posix():
        sys.path.insert(0, build_dir.as_posix())
    for extra in (REPO_ROOT / "scripts" / "eval", REPO_ROOT / "src"):
        if extra.as_posix() not in sys.path:
            sys.path.insert(0, extra.as_posix())
    import yaml
    from mot17_args import build_parser, configure_runtime_env
    from resolved_bridge_policy_config import fingerprint
    from saccade.perception.eval import evaluator as evaluator_module
    from saccade.perception.eval import stages as stages_module
    from saccade.perception.eval.pipeline import EvalPipeline
    from saccade.perception.temporal_yolo.mamba_gated_detector import (
        build_mamba_gated_detector,
        set_postprocess_compile,
    )

    return (
        yaml,
        build_parser,
        configure_runtime_env,
        fingerprint,
        evaluator_module,
        stages_module,
        EvalPipeline,
        build_mamba_gated_detector,
        set_postprocess_compile,
    )


def _image_size(tiling: str) -> int:
    for token, size in (("1280", 1280), ("1024", 1024), ("960", 960)):
        if token in tiling:
            return size
    return 640


def run_behavior_inventory(
    *,
    sequence: str,
    identity_mode: bool,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one capture-off eval and return the § 4.0 inventory.

    Instrumentation is observation-only: `_run_frame` and `_fast_emit_mot_lines`
    are wrapped to record what the pipeline already produced, and both are
    restored in a `finally`. No tracker setter is touched, so no research capture
    is enabled and no decision can move.
    """
    (
        yaml,
        build_parser,
        configure_runtime_env,
        fingerprint,
        evaluator_module,
        stages_module,
        EvalPipeline,
        build_mamba_gated_detector,
        set_postprocess_compile,
    ) = _import_eval_stack()

    build_witness = _assert_extension_consumed(resolve_build_dir())

    preset_path = REPO_ROOT / POLICY_PRESET_REL
    defaults = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    if not isinstance(defaults, dict):
        raise BehavioralIdentityError("policy preset is not a mapping")

    argv: list[str] = ["--sequences", sequence, "--output", output_dir.as_posix()]
    if identity_mode:
        # Declaration § 3.2: pin the known nondeterminism sources. Both are
        # recorded in the returned inventory, so an identity digest can never be
        # mistaken for a production-mode one.
        argv += ["--no-gpu-decode", "--cpp-threads", "1"]

    parser = build_parser()
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)

    resolved = fingerprint(POLICY_PRESET_STEM)
    configure_runtime_env(args, os.environ)

    tiling = getattr(args, "tiling", "native_640")
    detector = build_mamba_gated_detector(
        yolo_pt_path=args.mamba_yolo_weights,
        teacher_ckpt=args.mamba_teacher_ckpt,
        mamba_ckpt=args.mamba_ckpt,
        img_size=_image_size(tiling),
        device="cuda",
        conf_thr=0.001,
        max_det=getattr(args, "max_det", 300),
        trt_backbone_engine=getattr(args, "fpn_backbone_engine", ""),
        trt_head_engine=getattr(args, "mamba_head_engine", ""),
        temporal_T_override=0 if getattr(args, "no_temporal", False) else None,
        use_cuda_graph=bool(getattr(args, "use_cuda_graph", False)),
        use_whole_graph=bool(getattr(args, "use_whole_graph", False)),
        small_p3_max_threshold=getattr(args, "mamba_small_p3_max_threshold", 0.0),
    )
    set_postprocess_compile(True)
    detector.mamba_head.set_head_compile(True)
    detector.mamba_head.set_block_compile(True)

    active_pairs: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    pipelines: list[Any] = []
    emitted: list[tuple[str, tuple[str, ...]]] = []

    original_init = EvalPipeline.__init__
    original_run_frame = evaluator_module._run_frame
    original_evaluator_emit = evaluator_module._fast_emit_mot_lines
    original_stages_emit = stages_module._fast_emit_mot_lines

    def observed_init(self: Any, *positional: Any, **keywords: Any) -> None:
        original_init(self, *positional, **keywords)
        if self.seq != sequence or pipelines:
            raise BehavioralIdentityError(
                "evaluator constructed a second or unexpected sequence"
            )
        pipelines.append(self)

    def observed_run_frame(
        state: Any, *, frame_id: int, prepared_detection: Any = None
    ) -> bool:
        outcome = original_run_frame(
            state, frame_id=frame_id, prepared_detection=prepared_detection
        )
        native = state.detector.tracker.tracker
        raw = [
            [int(tid), int(slot)] for tid, slot in native.get_active_tid_slot_pairs()
        ]
        # A7.6's canonical value is "the pairs **sorted by slot**, with no omitted
        # active slot" — a normalization the recorder owes, not a property the
        # native call provides. `get_active_tid_slot_pairs()` iterates
        # `std::unordered_map<int,int> h_tid_to_slot_` (tracker_gpu.cu:5084) in
        # track-id bucket order, so slot order holds only by coincidence.
        # `run_h0_phase_a_child.py` asserts the raw order instead of sorting it;
        # that assertion is false in general and, under H0's structure, was only
        # reachable inside a sealed measurement. See declaration § 5.1.1.
        pairs = sorted(raw, key=lambda pair: pair[1])
        slots = [slot for _tid, slot in pairs]
        if len(set(slots)) != len(slots):
            raise BehavioralIdentityError(
                f"duplicate slot in active tid/slot pairs at frame {frame_id}: {pairs}"
            )
        active_pairs.append({"frame": int(frame_id), "pairs": pairs})
        return outcome

    def observing_emit(
        original: Callable[..., list[str]], **keywords: Any
    ) -> list[str]:
        lines = original(**keywords)
        track_results = keywords["track_results"]
        count = int(track_results["count"])
        if len(lines) != count:
            raise BehavioralIdentityError("emission cardinality mismatch")
        if keywords["seq"] != sequence:
            raise BehavioralIdentityError("emission sequence mismatch")
        boxes = track_results["boxes"][:count].numpy()
        scores = track_results["scores"][:count].numpy()
        classes_value = track_results.get("classes")
        if classes_value is None:
            classes = [int(getattr(args, "person_class", 0))] * count
        else:
            classes = [int(value) for value in classes_value[:count].numpy()]
        frame = int(keywords["frame_id"])
        row_base = sum(1 for row in final_rows if row["frame"] == frame)
        for offset, (line, box, score, class_id) in enumerate(
            zip(lines, boxes, scores, classes, strict=True)
        ):
            fields = line.split(",")
            if len(fields) != 10 or int(fields[0]) != frame:
                raise BehavioralIdentityError("emission/MOT row mismatch")
            x1, y1, x2, y2 = (float(value) for value in box)
            values = (x1, y1, x2 - x1, y2 - y1, float(score))
            final_rows.append(
                {
                    "binary32_bits": [_binary32_bits(value) for value in values],
                    "class": class_id,
                    "frame": frame,
                    "row_index": row_base + offset,
                    "track_id": int(fields[1]),
                }
            )
        return lines

    def sequence_callback(name: str, lines: tuple[str, ...]) -> None:
        if emitted or name != sequence or not isinstance(lines, tuple):
            raise BehavioralIdentityError("sequence callback cardinality/type mismatch")
        emitted.append((name, lines))

    EvalPipeline.__init__ = observed_init
    evaluator_module.EvalPipeline = EvalPipeline
    evaluator_module._run_frame = observed_run_frame
    evaluator_module._fast_emit_mot_lines = lambda **kw: observing_emit(
        original_evaluator_emit, **kw
    )
    stages_module._fast_emit_mot_lines = lambda **kw: observing_emit(
        original_stages_emit, **kw
    )

    skip_keys = {
        "module_detection",
        "module_geometry",
        "module_motion",
        "module_reid",
        "module_semantic",
        "module_trigger",
        "module_lifecycle",
        "mamba_ckpt",
        "mamba_teacher_ckpt",
        "mamba_yolo_weights",
        "teacher_head_ckpt",
        "visualize",
        "visualize_scale",
        "visualize_fps",
        "no_visualize_score",
        "visualize_trail_len",
    }
    eval_kwargs: dict[str, Any] = {
        key: value for key, value in vars(args).items() if key not in skip_keys
    }
    eval_kwargs["detector"] = detector
    eval_kwargs["tiling"] = tiling
    eval_kwargs["engine"] = "mamba"
    eval_kwargs["sequence_result_callback"] = sequence_callback

    try:
        evaluator_module.run_eval(**eval_kwargs)
        relink_debug = [
            int(value) for value in pipelines[0].detector.tracker.get_relink_debug()
        ]
    finally:
        EvalPipeline.__init__ = original_init
        evaluator_module.EvalPipeline = EvalPipeline
        evaluator_module._run_frame = original_run_frame
        evaluator_module._fast_emit_mot_lines = original_evaluator_emit
        stages_module._fast_emit_mot_lines = original_stages_emit

    if len(emitted) != 1 or len(pipelines) != 1:
        raise BehavioralIdentityError("evaluator did not produce exactly one sequence")
    if len(relink_debug) != 13:
        raise BehavioralIdentityError(
            f"relink_debug_raw is not the complete 13-integer vector: {relink_debug}"
        )

    mot_bytes = ("\n".join(emitted[0][1]) + "\n").encode("utf-8")
    inventory: dict[str, Any] = {
        "active_tid_slot_pairs": active_pairs,
        "final_track_rows": final_rows,
        "mot_output": {
            "length": len(mot_bytes),
            "sha256": hashlib.sha256(mot_bytes).hexdigest(),
        },
        "relink_debug_raw": relink_debug,
        "schema": BEHAVIOR_SCHEMA,
    }
    _validate_inventory(inventory)
    return {
        "inventory": inventory,
        "digest": behavior_digest(inventory),
        "mode": "identity" if identity_mode else "production",
        "sequence": sequence,
        "preset": POLICY_PRESET_REL,
        "resolved_fingerprint": resolved,
        "determinism_pinned": bool(identity_mode),
        # Witness only (declaration § 4.1): records which physical binary produced
        # the digest. Never a predicate on the axis.
        "build_witness": build_witness,
    }


def _validate_inventory(inventory: Mapping[str, Any]) -> None:
    missing = [name for name in BEHAVIOR_MEMBERS if name not in inventory]
    if missing:
        raise BehavioralIdentityError(f"inventory is missing members: {missing}")
    if inventory.get("schema") != BEHAVIOR_SCHEMA:
        raise BehavioralIdentityError("inventory schema mismatch")
    if not inventory["active_tid_slot_pairs"] or not inventory["final_track_rows"]:
        raise BehavioralIdentityError(
            "empty inventory: a digest over nothing is not an identity"
        )


def behavior_digest(inventory: Mapping[str, Any]) -> str:
    """Digest exactly the § 4.0 members — never the whole dict.

    Digesting the dict would fold in mode, fixture, and future keys, so an
    identity-mode digest and a production digest of the same behavior would
    differ. The axis is about behavior; provenance of the run is recorded
    alongside it, not inside it.
    """
    _validate_inventory(inventory)
    return digest({name: inventory[name] for name in BEHAVIOR_MEMBERS})


def _compare(results: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    digests = [str(item["digest"]) for item in results]
    return len(set(digests)) == 1, digests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity-mode",
        action="store_true",
        help="pin known nondeterminism sources and use the identity fixture",
    )
    parser.add_argument(
        "--sequences",
        default=None,
        help=f"sequence to run (default {IDENTITY_SEQUENCE} in identity mode)",
    )
    parser.add_argument("--repeats", type=int, default=1, help="run N times")
    parser.add_argument(
        "--require-identical",
        action="store_true",
        help="exit nonzero unless every repeat produces the same digest",
    )
    parser.add_argument("--emit", type=Path, default=None, help="write JSON result")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="evaluator output root (default: a scratch directory)",
    )
    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.require_identical and args.repeats < 2:
        parser.error("--require-identical needs --repeats >= 2")

    sequence = args.sequences or (
        IDENTITY_SEQUENCE if args.identity_mode else IDENTITY_SEQUENCE
    )
    out_root = args.out_dir or (REPO_ROOT / "out" / "h2_behavior")

    results: list[dict[str, Any]] = []
    for index in range(args.repeats):
        run_dir = out_root / f"run_{index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run_behavior_inventory(
                sequence=sequence,
                identity_mode=args.identity_mode,
                output_dir=run_dir,
            )
        except BehavioralIdentityError as exc:
            print(f"behavioral identity failed: {exc}", file=sys.stderr)
            return 1
        results.append(result)
        print(f"run {index}: behavior={result['digest']} mode={result['mode']}")

    identical, digests = _compare(results)
    payload = {
        "build_witness": results[0]["build_witness"],
        "digest": digests[0] if identical else None,
        "digests": digests,
        "identical": identical,
        "mode": results[0]["mode"],
        "preset": results[0]["preset"],
        "repeats": args.repeats,
        "resolved_fingerprint": results[0]["resolved_fingerprint"],
        "schema": "h2_behavior_result_v1",
        "sequence": sequence,
    }
    if args.emit:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_bytes(canonical_json_bytes(payload) + b"\n")
        print(f"wrote {args.emit}")

    if args.require_identical and not identical:
        print(
            "repeats are NOT byte-identical — this run is not reproducible: "
            + ", ".join(digests),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
