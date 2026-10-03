#!/usr/bin/env python3
"""Dump the #465 PR-5 post-detector replay inputs and the Python serial reference.

Issue #465 Phase B PR-5 (U3a; docs/reference/native_runtime_shipping_boundary.md
§6). Developer tooling only (`developer_build_debug`): it runs one unmodified
``scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP`` in the
serial configuration (no ``--double-buffer``) and records, per frame, what the
native replay host (``saccade_replay``) consumes and what it is compared with.
Nothing in the run is replaced; the hooks below only observe a stage's inputs
and outputs (they add host syncs, which order but do not change GPU work):

* ``evaluator._run_detect`` -> ``detector.bin``: the detector output that
  enters ``_run_native_tensor_prep`` (cast to float32/int32 there, as here)
  and ``is_tiled``; ``source_keypoints`` must be None (headline has none);
* ``stages._run_gmc_estimate`` -> ``frames.u8`` + ``gmc.bin``: the GMC input
  frame (float32 CHW, ``torch.div(u8, 255.0)`` of the decoded frame) stored as
  uint8 CHW -- each frame is checked to reproduce the float frame bit for bit,
  so the dump is lossless for the native GMC -- and the warp it returned;
* ``evaluator._run_post_nms_finalize`` -> ``post_nms.bin``: main NMS plus the
  appended private-continuation candidates (stage reference);
* ``evaluator._run_track`` -> ``tracker_in.bin`` / ``tracker_out.bin``: the
  detections and GMC warp the tracker update receives (after the external FP
  rule filter and FP hard filter) and the tracker output rows it returns.

A frame whose detector output is empty has a ``detector.bin`` record only: the
oracle returns before NMS, GMC and the tracker update (evaluator ``_run_frame``).

Format ``saccade.post_detector_replay/v1`` (little-endian, no padding), one
directory per sequence with ``meta.json`` (geometry, record counts, sha256 of
every file) and:

* ``detector.bin``    int32 frame, int32 n, int32 is_tiled, f32 boxes[n*4],
  f32 scores[n], i32 classes[n]
* ``post_nms.bin``    int32 frame, int32 n, f32 boxes[n*4], f32 scores[n],
  i32 classes[n]
* ``gmc.bin``         int32 frame, int32 frame_index, int32 has_warp,
  f32 warp[6]
* ``tracker_in.bin``  int32 frame, int32 n, int32 has_gmc, f32 gmc[6],
  f32 boxes[n*4], f32 scores[n], i32 classes[n]
* ``tracker_out.bin`` int32 frame, int32 count, f32 boxes[count*4],
  f32 scores[count], i32 ids[count], i32 classes[count]
* ``frames.u8``       uint8 CHW frames (3*H*W bytes each), in ``gmc.bin`` order

``manifest.json`` at the top records the run (argv, git head, SACCADE_* env,
the committed resolved config's sha256, the 256-entry ``u8 -> float32``
table torch produced). ``--frames hash`` hashes the frames without storing
them (a second dump for the serial reference's run-to-run check).

Usage (GPU; a formal dump runs under a lease)::

    .venv/bin/python tools/resctl.py run gpu0 -- \\
        .venv/bin/python scripts/eval/diagnostics/dump_post_detector_replay.py \\
        --out results/465_pr5_replay/<label>/dump [--sequences MOT17-09-SDP] \\
        [--max-frames N] [--frames store|hash]

    # integrity of a dump, and bit-equality with a second (hash-only) dump
    .venv/bin/python scripts/eval/diagnostics/dump_post_detector_replay.py \\
        --verify <dump> [--against <dump2>] [--report verify.json]
"""
# status: diagnostic

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import runpy
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO

project_root = Path(__file__).resolve().parents[3]

FORMAT = "saccade.post_detector_replay/v1"
PRESET = "mamba_whole_graph"
RESOLVED_CONFIG = project_root / "configs/shipping/mamba_whole_graph.resolved.json"
STREAMS = ("detector", "post_nms", "gmc", "tracker_in", "tracker_out")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=project_root, capture_output=True, text=True, check=False
    )
    return out.stdout.strip()


class _HashedFile:
    """Append-only file that hashes what it writes (None path = hash only)."""

    def __init__(self, path: Path | None) -> None:
        self._fh: BinaryIO | None = path.open("xb") if path is not None else None
        self._sha = hashlib.sha256()
        self.records = 0
        self.bytes = 0

    def write(self, *parts: bytes) -> None:
        for part in parts:
            self._sha.update(part)
            self.bytes += len(part)
            if self._fh is not None:
                self._fh.write(part)
        self.records += 1

    def close(self) -> dict[str, Any]:
        if self._fh is not None:
            self._fh.close()
        return {
            "records": self.records,
            "bytes": self.bytes,
            "sha256": self._sha.hexdigest(),
        }


class SequenceWriter:
    def __init__(
        self, root: Path, seq: str, width: int, height: int, store_frames: bool
    ):
        self.dir = root / seq
        self.dir.mkdir(parents=True, exist_ok=False)
        self.seq, self.width, self.height = seq, width, height
        self.files = {name: _HashedFile(self.dir / f"{name}.bin") for name in STREAMS}
        self.frames = _HashedFile(self.dir / "frames.u8" if store_frames else None)
        self.store_frames = store_frames
        self.empty_detector_frames: list[int] = []

    def close(self, extra: dict[str, Any]) -> dict[str, Any]:
        files = {f"{name}.bin": f.close() for name, f in self.files.items()}
        frames = self.frames.close()
        meta = {
            "format": FORMAT,
            "sequence": self.seq,
            "width": self.width,
            "height": self.height,
            "files": files,
            "frames_u8": {**frames, "stored": self.store_frames},
            "empty_detector_frames": self.empty_detector_frames,
            **extra,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        return meta


def _i32(*values: int) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def mot17_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "scripts/eval/mot17.py",
        "--preset",
        PRESET,
        "--detector",
        "SDP",
        "--output",
        str(args.out / "eval"),
    ]
    if args.sequences:
        argv += ["--sequences", args.sequences]
    if args.max_frames:
        argv += ["--max-frames", str(args.max_frames)]
    return argv


def run(args: argparse.Namespace) -> int:
    build_path = Path(os.environ.get("SACCADE_BUILD_PATH", project_root / "build"))
    if build_path.exists():
        sys.path.insert(0, str(build_path))
    sys.path.insert(0, str(project_root / "src"))
    import saccade.perception.detector_trt  # noqa: F401  (must precede torchvision)
    import numpy as np
    import torch

    import saccade.perception.eval.evaluator as evaluator_module
    from saccade.perception.eval import stages as stages_module

    store_frames = args.frames == "store"
    writers: dict[str, SequenceWriter] = {}
    seq_meta: dict[str, dict[str, Any]] = {}
    env_seen: dict[str, str] = {}
    dtypes_seen: dict[str, str] = {}

    def to_np(t: torch.Tensor, dtype: Any) -> np.ndarray:
        return np.ascontiguousarray(t.detach().cpu().numpy().astype(dtype, copy=False))

    def writer_for(state: Any) -> SequenceWriter:
        seq = str(state.seq)
        w = writers.get(seq)
        if w is None:
            for prev in list(writers):
                seq_meta[prev] = writers.pop(prev).close({})
            if not env_seen:
                env_seen.update(
                    {k: v for k, v in os.environ.items() if k.startswith("SACCADE_")}
                )
            if state.double_buffer_stream is not None:
                raise RuntimeError("the dump must run the serial configuration")
            w = SequenceWriter(
                args.out, seq, int(state.w_orig), int(state.h_orig), store_frames
            )
            writers[seq] = w
        return w

    def det_rows(boxes: Any, scores: Any, classes: Any) -> tuple[bytes, int]:
        b = to_np(boxes.to(torch.float32), np.float32).reshape(-1, 4)
        s = to_np(scores.to(torch.float32), np.float32).reshape(-1)
        c32 = classes.to(torch.int32)
        if not torch.equal(c32.to(classes.dtype), classes):
            raise RuntimeError("class ids do not survive the int32 cast")
        c = to_np(c32, np.int32).reshape(-1)
        if not (len(b) == len(s) == len(c)):
            raise RuntimeError(f"ragged detections {len(b)}/{len(s)}/{len(c)}")
        return b.tobytes() + s.tobytes() + c.tobytes(), len(s)

    original = {
        "detect": evaluator_module._run_detect,
        "finalize": evaluator_module._run_post_nms_finalize,
        "track": evaluator_module._run_track,
        "gmc": stages_module._run_gmc_estimate,
    }

    def run_detect(state: Any, **kwargs: Any) -> Any:
        out = original["detect"](state, **kwargs)
        boxes, scores, classes, is_tiled, keypoints = out
        if keypoints is not None:
            raise RuntimeError("headline detector produced keypoints; dump has no slot")
        for name, t in (("boxes", boxes), ("scores", scores), ("classes", classes)):
            dtypes_seen.setdefault(name, str(t.dtype))
        payload, n = det_rows(boxes, scores, classes)
        w = writer_for(state)
        frame = int(state.current_frame_id)
        if n == 0:
            w.empty_detector_frames.append(frame)
        w.files["detector"].write(_i32(frame, n, int(bool(is_tiled))), payload)
        return out

    def run_finalize(state: Any, fctx: Any, **kwargs: Any) -> Any:
        out = original["finalize"](state, fctx, **kwargs)
        payload, n = det_rows(out[0], out[1], out[2])
        writer_for(state).files["post_nms"].write(
            _i32(int(state.current_frame_id), n), payload
        )
        return out

    def run_gmc(state: Any, **kwargs: Any) -> Any:
        frame_f = kwargs["_frame_gmc"]
        if (
            frame_f.dtype != torch.float32
            or frame_f.dim() != 3
            or frame_f.shape[0] != 3
        ):
            raise RuntimeError(
                f"unexpected GMC input {frame_f.dtype} {tuple(frame_f.shape)}"
            )
        u8 = torch.round(frame_f * 255.0).clamp_(0, 255).to(torch.uint8)
        if not torch.equal(torch.div(u8, 255.0), frame_f):
            raise RuntimeError(
                f"{state.seq} frame {state.current_frame_id}: the GMC input is not "
                "torch.div(u8, 255.0) of any uint8 frame; the dump cannot carry it"
            )
        w = writer_for(state)
        frame_index = w.frames.records
        w.frames.write(to_np(u8.contiguous(), np.uint8).tobytes())
        warp, uncertain = original["gmc"](state, **kwargs)
        has = warp is not None
        vals = (
            to_np(warp.flatten()[:6].to(torch.float32), np.float32)
            if has
            else np.zeros(6, np.float32)
        )
        w.files["gmc"].write(
            _i32(int(state.current_frame_id), frame_index, int(has)), vals.tobytes()
        )
        return warp, uncertain

    def run_track(state: Any, **kwargs: Any) -> Any:
        gmc = kwargs.get("gmc_warp")
        if kwargs.get("embeddings") is not None:
            raise RuntimeError("headline tracker received embeddings")
        payload, n = det_rows(
            kwargs["fused_boxes"], kwargs["fused_scores"], kwargs["fused_classes"]
        )
        g = (
            to_np(gmc.flatten()[:6].to(torch.float32), np.float32)
            if gmc is not None
            else np.zeros(6, np.float32)
        )
        w = writer_for(state)
        frame = int(state.current_frame_id)
        w.files["tracker_in"].write(
            _i32(frame, n, int(gmc is not None)), g.tobytes(), payload
        )
        result = original["track"](state, **kwargs)
        torch.cuda.synchronize()
        count = int(result["count"].reshape(-1)[0].item())
        boxes = to_np(result["boxes"][:count].to(torch.float32), np.float32)
        scores = to_np(result["scores"][:count].to(torch.float32), np.float32)
        ids = to_np(result["ids"][:count].to(torch.int32), np.int32)
        classes = to_np(result["classes"][:count].to(torch.int32), np.int32)
        w.files["tracker_out"].write(
            _i32(frame, count),
            boxes.tobytes(),
            scores.tobytes(),
            ids.tobytes(),
            classes.tobytes(),
        )
        return result

    lut = torch.div(torch.arange(256, dtype=torch.uint8, device="cuda"), 255.0)
    lut_hex = [struct.pack("<f", float(v)).hex() for v in lut.cpu().tolist()]

    evaluator_module._run_detect = run_detect
    evaluator_module._run_post_nms_finalize = run_finalize
    evaluator_module._run_track = run_track
    stages_module._run_gmc_estimate = run_gmc
    sys.path.insert(0, str(project_root / "scripts" / "eval"))
    argv = mot17_argv(args)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    saved_argv = sys.argv
    sys.argv = list(argv)
    exit_code = 0
    try:
        runpy.run_path(str(project_root / argv[0]), run_name="__main__")
    except SystemExit as exc:
        exit_code = (
            exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        )
    finally:
        sys.argv = saved_argv
        evaluator_module._run_detect = original["detect"]
        evaluator_module._run_post_nms_finalize = original["finalize"]
        evaluator_module._run_track = original["track"]
        stages_module._run_gmc_estimate = original["gmc"]
        for seq in list(writers):
            seq_meta[seq] = writers.pop(seq).close({})

    manifest = {
        "format": FORMAT,
        "tool": "scripts/eval/diagnostics/dump_post_detector_replay.py",
        "argv": argv,
        "exit_code": exit_code,
        "started_utc": started,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "resolved_config": {
            "path": str(RESOLVED_CONFIG.relative_to(project_root)),
            "sha256": _sha256_file(RESOLVED_CONFIG),
        },
        "saccade_env": dict(sorted(env_seen.items())),
        "detector_output_dtypes": dtypes_seen,
        "frames": args.frames,
        "frame_lut_f32_hex": lut_hex,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "sequences": {seq: seq_meta[seq] for seq in sorted(seq_meta)},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return exit_code


def verify(dump: Path, against: Path | None) -> dict[str, Any]:
    """Re-hash ``dump`` against its meta; optionally compare with a second dump.

    Integrity: every stream file (and ``frames.u8`` when stored) must match the
    sha256, byte and record counts its ``meta.json`` records, and each
    ``meta.json`` must equal the manifest's copy. Comparison (``against``, e.g.
    the ``--frames hash`` repeat): every stream's sha256 and the frame hash must
    be equal, i.e. the serial reference reproduced bit for bit.
    """
    manifest = json.loads((dump / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise SystemExit(f"{dump}: not a {FORMAT} dump")
    problems: list[str] = []
    for seq, listed in manifest["sequences"].items():
        meta = json.loads((dump / seq / "meta.json").read_text())
        if meta != listed:
            problems.append(f"{seq}: meta.json differs from the manifest")
        for name, rec in meta["files"].items():
            path = dump / seq / name
            if (
                path.stat().st_size != rec["bytes"]
                or _sha256_file(path) != rec["sha256"]
            ):
                problems.append(f"{seq}/{name}: content differs from meta.json")
        frames = meta["frames_u8"]
        if frames["stored"]:
            path = dump / seq / "frames.u8"
            if (
                path.stat().st_size != frames["bytes"]
                or _sha256_file(path) != frames["sha256"]
            ):
                problems.append(f"{seq}/frames.u8: content differs from meta.json")
    result: dict[str, Any] = {
        "format": "saccade.post_detector_replay_verify/v1",
        "dump": str(dump),
        "git_head": manifest["git_head"],
        "git_dirty": manifest["git_dirty"],
        "sequences": sorted(manifest["sequences"]),
        "integrity_problems": problems,
    }
    if against is not None:
        other = json.loads((against / "manifest.json").read_text())
        diffs: list[str] = []
        if sorted(other["sequences"]) != sorted(manifest["sequences"]):
            diffs.append("sequence sets differ")
        for seq in sorted(set(manifest["sequences"]) & set(other["sequences"])):
            a, b = manifest["sequences"][seq], other["sequences"][seq]
            for name in a["files"]:
                if a["files"][name] != b["files"].get(name):
                    diffs.append(f"{seq}/{name}")
            if a["frames_u8"]["sha256"] != b["frames_u8"]["sha256"]:
                diffs.append(f"{seq}/frames.u8")
            if a["empty_detector_frames"] != b["empty_detector_frames"]:
                diffs.append(f"{seq}/empty_detector_frames")
        result["against"] = {
            "dump": str(against),
            "git_head": other["git_head"],
            "git_dirty": other["git_dirty"],
            "differing_streams": diffs,
        }
    result["verdict"] = (
        "OK"
        if not problems and not result.get("against", {}).get("differing_streams")
        else "FAIL"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, help="new or empty directory (dump mode)")
    ap.add_argument("--sequences", default="", help="comma-separated (default: all)")
    ap.add_argument(
        "--max-frames", type=int, default=0, help="per-sequence cap (smoke)"
    )
    ap.add_argument(
        "--frames",
        choices=("store", "hash"),
        default="store",
        help="store the GMC input frames, or only hash them",
    )
    ap.add_argument(
        "--verify", type=Path, help="verify mode: re-hash this dump against its meta"
    )
    ap.add_argument(
        "--against", type=Path, help="with --verify: a second dump that must match"
    )
    ap.add_argument("--report", type=Path, help="with --verify: write the JSON here")
    args = ap.parse_args(argv)
    if args.verify is not None:
        if args.out is not None:
            ap.error("--verify and --out are exclusive")
        result = verify(args.verify.resolve(), args.against and args.against.resolve())
        text = json.dumps(result, indent=2) + "\n"
        if args.report is not None:
            args.report.write_text(text)
        print(text, end="")
        return 0 if result["verdict"] == "OK" else 1
    if args.out is None:
        ap.error("--out is required (or --verify)")
    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        ap.error(f"{args.out} is not empty")
    # ADR 021 AP-2: claim the dump directory before the first byte. The in-process
    # mot17.py run claims its own ``eval/`` sub-directory.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts.provenance.run_manifest import open_run

    open_run(
        args.out,
        produced_by="diagnostic",
        preset=PRESET,
        detector="SDP",
        dataset="MOT17 train",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
