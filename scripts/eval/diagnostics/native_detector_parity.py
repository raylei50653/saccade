#!/usr/bin/env python3
"""Native detector parity vs the A_L detector path (#465 Phase B PR-8).

Issue #465 Phase B PR-8 (U3b-2; docs/reference/native_runtime_shipping_boundary.md
§6: "detection tensor 對 oracle"; measurement contract:
docs/reference/native_runtime_resolved_config.md §12). Developer tooling only
(``developer_build_debug``). The oracle is ``A_L``: the headline
``mot17.py --preset mamba_whole_graph`` with the PR-1L LibTorch head injected
into the ``_trt_head`` slot exactly as the frozen PR-2L runner
(``native_head_parity_libtorch.py``) injects it -- its own functions are
called here, with the operator library's sha256 taken from the realization
attestation instead of the build the PR-2L packet recorded (PR-1L itself stays
frozen). Subcommands:

``anchor``       re-runs the PR-2L ``A_L`` arm (double-buffer, 7 sequences) with
                 the current build and compares its MOT txt bytes with the
                 PR-2L packet's ``A_L_1`` -- the evidence that this build
                 realizes the accepted ``A_L``; also runs PR-2L's V5 sidecar
                 checks;
``attest``       writes the realization attestation
                 (``configs/shipping/mamba_head_realization.attestation.json``)
                 from an anchor report; refuses unless the anchor is identical;
``oracle-rows``  runs ``A_L`` serially (no ``--double-buffer``, the PR-5 dump's
                 configuration) and records ``evaluator._run_detect``'s output
                 per frame -- the PR-5 ``detector.bin`` boundary (float32 boxes
                 and scores, int32 classes, ``is_tiled``, no keypoints);
``parity``       streams the native ingest + detector of each sequence from
                 ``build/shipping/saccade_detector_probe`` (own process, no
                 Python) and compares, frame by frame and section by section,
                 each native stage with the oracle stage fed the *native* input
                 of that stage, so a difference is reported where it arises and
                 is not carried into the next section:

                 ``decoder``   native decoded uint8 vs torchvision's decode;
                 ``ingest``    native frame buffer vs the oracle ingest op on
                               torchvision's decode (PR-7, re-checked);
                 ``resize``    native [1, 3, 640, 640] vs ``F.interpolate`` of
                               the native frame buffer;
                 ``backbone``  native p3/p4/p5 vs ``TRTYoloBackbone.infer_graph``
                               on the native resized frame;
                 ``head``      native head outputs vs the A_L head (PR-2L's
                               adapter around the PR-1L module) on the native
                               p3/p4/p5;
                 ``s2_raw`` / ``s2_scaled`` / ``s2_rows``  native S2 vs the
                               oracle's compiled ``_postprocess_mamba_fixed`` +
                               ``_whole_graph_fn``'s coordinate scaling on the
                               native head outputs (before scaling, after, and
                               as detector rows);
                 ``detector``  native detector rows (the chained native path)
                               vs the ``oracle-rows`` dump (``--oracle-rows``).

Tensor sections are bit comparisons (``max_abs``, ordered ``max_ulp``,
differing value counts, first differing frames). Row sections report count,
membership (multiset of row bit patterns), order, class ids, score bits, box
bits and all-zero rows separately. There is no tolerance: a section is
``EXACT`` or ``DIFFERS``; the run is ``EXACT`` only when every section is.
``--mutation`` passes a negative control to the probe and reports whether the
expected section caught it. ``--against`` compares per-frame hashes with an
earlier ``parity`` run (run-to-run identity of both sides).

Usage (GPU; formal runs hold the gpu0 lease)::

    cmake --build build --target saccade_detector_probe
    R=results/465_pr8_detector/<label>
    .venv/bin/python tools/resctl.py run gpu0 -- .venv/bin/python \\
        scripts/eval/diagnostics/native_detector_parity.py anchor --out $R/anchor
    .venv/bin/python scripts/eval/diagnostics/native_detector_parity.py \\
        attest --anchor $R/anchor/anchor.json
    .venv/bin/python tools/resctl.py run gpu0 -- .venv/bin/python \\
        scripts/eval/diagnostics/native_detector_parity.py oracle-rows --out $R/oracle_rows
    .venv/bin/python tools/resctl.py run gpu0 -- .venv/bin/python \\
        scripts/eval/diagnostics/native_detector_parity.py parity --out $R/parity \\
        --oracle-rows $R/oracle_rows [--sequences S,..] [--max-frames N] \\
        [--mutation M] [--against $R/parity/report.json]

Exit 0: identical/EXACT (negative control: caught); 1: a difference (not
caught); 2: error.
"""
# status: diagnostic

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import struct
import subprocess
import sys
from pathlib import Path
from typing import IO, Any

project_root = Path(__file__).resolve().parents[3]

SCHEMA = "saccade.native_detector_parity/v1"
ANCHOR_SCHEMA = "saccade.native_detector_anchor/v1"
ROWS_FORMAT = "saccade.native_detector_oracle_rows/v1"
STREAM_FORMAT = "saccade.native_detector_stream/v1"
ATTESTATION_SCHEMA = "saccade.head_realization_attestation/v1"
RESOLVED_CONFIG = "configs/shipping/mamba_whole_graph.resolved.json"
PROBE = "build/shipping/saccade_detector_probe"
LINEAGE = "models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json"
ATTESTATION = "configs/shipping/mamba_head_realization.attestation.json"
OP_LIBRARY = "build/libsaccade_scan_torchop.so"
OP_SOURCES = (
    "src/tracking/mamba_scan_torchop.cpp",
    "src/tracking/mamba_scan.cu",
    "include/tracking/mamba_scan.cuh",
)
PR2L_RUNNER = "scripts/eval/diagnostics/native_head_parity_libtorch.py"
INGEST_HARNESS = "scripts/eval/diagnostics/native_ingest_parity.py"
# The PR-2L formal packet's A_L_1 (owner ACCEPT, #465 issuecomment-5967756337).
PR2L_A_L_REFERENCE = "results/native_head_parity_465_libtorch/20260928T152911Z/l2/A_L_1"
PR2L_A_L_REFERENCE_TXT_SHA256 = {
    "MOT17-02-SDP": "e7e39ecbb0165c2ebdf5833d4ebcf4ae0db3bf2d235e219b62e10fb646ab4334",
    "MOT17-04-SDP": "b3a2a0e62943825bf87aa862373a6b216b071c369ba5a8246dce35a4ca5d7be9",
    "MOT17-05-SDP": "2c2d4f37da63949d456a347fb1a32ae2f16153a88a1442ede5b56a02c79dab69",
    "MOT17-09-SDP": "b46c585ac28511f33acc20007cb7434ff01ab314683e9b1da5805554ad5d7b76",
    "MOT17-10-SDP": "18c84298fd087d9616b27acc731ae701cdcd3b3b57ccceec3e3ee02894b14dc4",
    "MOT17-11-SDP": "73cdc51fc9dd7e947f130d7735fea80a2673589d35b6a3b48da274da658990d9",
    "MOT17-13-SDP": "4a4045e63fa3ba0ff1df4be6a6078944583a8683b0b67e8c870d20b9196725bd",
}
SEQUENCES = tuple(f"MOT17-{n:02d}-SDP" for n in (2, 4, 5, 9, 10, 11, 13))
TOTAL_FRAMES = 5316
DATA_ROOT = "datasets/MOT17"
SPLIT = "train"

TENSOR_SECTIONS = (
    "decoder",
    "ingest",
    "resize",
    "backbone",
    "head",
    "s2_raw",
    "s2_scaled",
)
ROW_SECTIONS = ("s2_rows", "detector")
HEAD_NAMES = ("cls_p3", "cls_p4", "cls_p5", "reg_p3", "reg_p4", "reg_p5")
# Negative controls: the section that must report DIFFERS for each mutation.
NEGCTL_EXPECT = {
    "backbone_ulp": "backbone",
    "head_ulp": "head",
    "s2_threshold": "s2_rows",
    "s2_topk": "s2_rows",
    "s2_order": "s2_rows",
    "box_ulp": "s2_rows",
}
FIRST_DIFFS_KEPT = 20
DET_RECORD = struct.Struct("<3i")  # frame, n, is_tiled


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


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def _load_module(rel: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, project_root / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git_state() -> dict[str, Any]:
    return {
        "head": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


# ── the A_L oracle (the frozen PR-2L runner's own functions) ───────────────────


def load_pr2l(op_library_sha256: str) -> Any:
    """The PR-2L runner module, bound to ``op_library_sha256`` (the realized
    build) instead of the build its packet recorded. Nothing else changes."""
    p2l = _load_module(PR2L_RUNNER, "pr2l_runner")
    p2l.OP_LIBRARY_SHA256 = op_library_sha256
    return p2l


def read_attestation() -> dict[str, Any]:
    """The committed attestation, checked against the frozen lineage file and
    the operator library on disk."""
    att = json.loads((project_root / ATTESTATION).read_text())
    if att.get("schema") != ATTESTATION_SCHEMA:
        raise RuntimeError(f"{ATTESTATION}: schema {att.get('schema')!r}")
    lineage_sha = _sha256_file(project_root / LINEAGE)
    if att["frozen_lineage"]["sha256"] != lineage_sha:
        raise RuntimeError(f"{ATTESTATION} is bound to another lineage file")
    op_sha = _sha256_file(project_root / OP_LIBRARY)
    if att["op_library"]["sha256"] != op_sha:
        raise RuntimeError(
            f"{OP_LIBRARY} sha256 {op_sha} is not the attested build "
            f"{att['op_library']['sha256']}"
        )
    if att["a_l_reproduction"]["identical"] is not True:
        raise RuntimeError(f"{ATTESTATION}: the A_L reproduction is not identical")
    return att


# ── anchor: the A_L reproduction with the current build ────────────────────────


def anchor_child(out_rel: str) -> int:
    op_sha = _sha256_file(project_root / OP_LIBRARY)
    p2l = load_pr2l(op_sha)
    out = project_root / out_rel
    return p2l.run_arm_child(
        "A_L", out / "A_L", out / "A_L.sidecar.json", p2l.SEQUENCES
    )


def run_anchor(args: argparse.Namespace) -> int:
    out: Path = args.out
    op_sha = _sha256_file(project_root / OP_LIBRARY)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().relative_to(project_root)),
        "_anchor-child",
        str(out.relative_to(project_root)),
    ]
    with (out / "A_L.stdout.log").open("w") as log:
        rc = subprocess.run(
            cmd, cwd=project_root, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    p2l = load_pr2l(op_sha)
    sidecar_path = out / "A_L.sidecar.json"
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else None
    ref_sidecar_path = project_root / (PR2L_A_L_REFERENCE + ".sidecar.json")
    ref_sidecar = (
        json.loads(ref_sidecar_path.read_text()) if ref_sidecar_path.exists() else None
    )
    problems = [] if rc == 0 else [f"A_L child exited {rc}"]
    problems += [f"V5: {p}" for p in p2l.sidecar_problems("A_L", sidecar)]
    problems += [
        f"env: {p}"
        for p in p2l.env_override_problems(
            (sidecar or {}).get("resolved_env_overrides"),
            (ref_sidecar or {}).get("resolved_env_overrides"),
        )
    ]
    per_seq = {}
    for seq in SEQUENCES:
        observed = out / "A_L" / f"{seq}.txt"
        reference = project_root / PR2L_A_L_REFERENCE / f"{seq}.txt"
        o = _sha256_file(observed) if observed.exists() else None
        r = _sha256_file(reference) if reference.exists() else None
        if r is not None and r != PR2L_A_L_REFERENCE_TXT_SHA256[seq]:
            problems.append(f"{reference}: sha256 {r} != the recorded reference")
        per_seq[seq] = {"reference": PR2L_A_L_REFERENCE_TXT_SHA256[seq], "observed": o}
    identical = not problems and all(
        v["observed"] == v["reference"] for v in per_seq.values()
    )
    report = {
        "schema": ANCHOR_SCHEMA,
        "generated_utc": _utc(),
        "git": _git_state(),
        "op_library": {"path": OP_LIBRARY, "sha256": op_sha},
        "reference": PR2L_A_L_REFERENCE,
        "mot17_argv": (sidecar or {}).get("mot17_argv"),
        "txt_sha256": per_seq,
        "problems": problems,
        "identical": identical,
    }
    _write_json(out / "anchor.json", report)
    print(f"anchor: {'IDENTICAL' if identical else 'NOT IDENTICAL'} {problems}")
    return 0 if identical else 1


def _needed(path: Path) -> list[str]:
    out = subprocess.run(
        ["readelf", "-d", str(path)], capture_output=True, text=True, check=True
    ).stdout
    return [
        line.split("[", 1)[1].rstrip("]").strip()
        for line in out.splitlines()
        if "(NEEDED)" in line
    ]


def run_attest(args: argparse.Namespace) -> int:
    anchor = json.loads(args.anchor.read_text())
    if anchor.get("schema") != ANCHOR_SCHEMA or anchor.get("identical") is not True:
        print(
            "attest: the anchor report is not an identical A_L reproduction",
            file=sys.stderr,
        )
        return 1
    if anchor["git"]["dirty"]:
        print("attest: the anchor ran on a dirty tree", file=sys.stderr)
        return 1
    lineage_path = project_root / LINEAGE
    lineage = json.loads(lineage_path.read_text())
    op = project_root / OP_LIBRARY
    op_sha = _sha256_file(op)
    if op_sha != anchor["op_library"]["sha256"]:
        print(
            "attest: the operator library changed since the anchor ran", file=sys.stderr
        )
        return 1
    tool_commit = lineage["tool"]["git_commit"]
    sources = {}
    for rel in OP_SOURCES:
        head_blob = _git("rev-parse", f"HEAD:{rel}")
        lineage_blob = _git("rev-parse", f"{tool_commit}:{rel}")
        if head_blob != lineage_blob:
            print(
                f"attest: {rel} changed since the lineage's tool commit",
                file=sys.stderr,
            )
            return 1
        sources[rel] = head_blob
    needed = _needed(op)
    if any(n.startswith(("libpython", "libtorch_python")) for n in needed):
        print(f"attest: {OP_LIBRARY} links Python: {needed}", file=sys.stderr)
        return 1
    comment = subprocess.run(
        ["readelf", "-p", ".comment", str(op)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    att = {
        "schema": ATTESTATION_SCHEMA,
        "issue": "#465 Phase B PR-8 (U3b-2)",
        "reading": (
            "Names the operator-library build that realizes the frozen PR-1L lineage on "
            "this machine. The lineage (and the PR-1L/PR-2L freezes) are unchanged; the "
            "build the lineage names (op_library_sha256 below) no longer exists. This build "
            "compiles the same sources, and the PR-2L A_L arm re-run with it gives MOT txt "
            "byte-identical to the accepted packet's A_L_1 on every sequence."
        ),
        "frozen_lineage": {
            "path": LINEAGE,
            "sha256": _sha256_file(lineage_path),
            "tool_commit": tool_commit,
            "torchscript_sha256": lineage["torchscript"]["sha256"],
            "torchscript_content_sha256": lineage["torchscript"]["content_sha256"],
            "op_library_sha256": lineage["op_library"]["sha256"],
        },
        "op_library": {
            "path": OP_LIBRARY,
            "sha256": op_sha,
            "bytes": op.stat().st_size,
            "needed": needed,
            "compiler_comment": [
                line.split("]", 1)[1].strip()
                for line in comment.splitlines()
                if line.strip().startswith("[")
            ],
            "sources_git_blob": sources,
            "sources_equal_to_lineage_tool_commit": True,
        },
        "a_l_reproduction": {
            "tool": "scripts/eval/diagnostics/native_detector_parity.py anchor",
            "git_head": anchor["git"]["head"],
            "generated_utc": anchor["generated_utc"],
            "reference": anchor["reference"],
            "mot17_argv": anchor["mot17_argv"],
            "txt_sha256": anchor["txt_sha256"],
            "identical": True,
        },
    }
    dest = args.write or (project_root / ATTESTATION)
    _write_json(dest, att)
    print(f"attest: wrote {dest}")
    return 0


# ── oracle-rows: the A_L serial run's detector output ──────────────────────────


def oracle_rows_child(
    out_rel: str, sequences: list[str], max_frames: int | None
) -> int:
    att = read_attestation()
    p2l = load_pr2l(att["op_library"]["sha256"])
    out = project_root / out_rel
    original_argv = p2l.mot17_argv

    def serial_argv(arm, out_arg, seqs, data_root):  # type: ignore[no-untyped-def]
        argv = [
            a
            for a in original_argv(arm, out_arg, seqs, data_root)
            if a != "--double-buffer"
        ]
        if max_frames:
            argv += ["--max-frames", str(max_frames)]
        return argv

    p2l.mot17_argv = serial_argv

    build_path = Path(os.environ.get("SACCADE_BUILD_PATH", project_root / "build"))
    if build_path.exists():
        sys.path.insert(0, str(build_path))
    import saccade.perception.detector_trt  # noqa: F401  (before torchvision)
    import numpy as np
    import torch

    import saccade.perception.eval.evaluator as evaluator_module

    files: dict[str, IO[bytes]] = {}
    meta: dict[str, dict[str, Any]] = {}
    original_detect = evaluator_module._run_detect

    def to_np(t: Any, dtype: Any) -> Any:
        return np.ascontiguousarray(t.detach().cpu().numpy().astype(dtype, copy=False))

    def run_detect(state: Any, **kwargs: Any) -> Any:
        res = original_detect(state, **kwargs)
        boxes, scores, classes, is_tiled, keypoints = res
        if keypoints is not None:
            raise RuntimeError("A_L detector produced keypoints")
        if state.double_buffer_stream is not None:
            raise RuntimeError("oracle-rows must run the serial configuration")
        seq = str(state.seq)
        if seq not in files:
            for prev in list(files):
                files.pop(prev).close()
            (out / "rows" / seq).mkdir(parents=True, exist_ok=True)
            files[seq] = (out / "rows" / seq / "detector.bin").open("wb")
            meta[seq] = {"records": 0, "frames": [], "dtypes": {}, "is_tiled": []}
        m = meta[seq]
        for name, t in (("boxes", boxes), ("scores", scores), ("classes", classes)):
            m["dtypes"].setdefault(name, str(t.dtype))
        c32 = classes.to(torch.int32)
        if not torch.equal(c32.to(classes.dtype), classes):
            raise RuntimeError("class ids do not survive the int32 cast")
        b = to_np(boxes.to(torch.float32), np.float32).reshape(-1, 4)
        s = to_np(scores.to(torch.float32), np.float32).reshape(-1)
        c = to_np(c32, np.int32).reshape(-1)
        if not (len(b) == len(s) == len(c)):
            raise RuntimeError("ragged detector output")
        frame = int(state.current_frame_id)
        files[seq].write(DET_RECORD.pack(frame, len(s), int(bool(is_tiled))))
        files[seq].write(b.tobytes() + s.tobytes() + c.tobytes())
        m["records"] += 1
        m["frames"].append(frame)
        m["is_tiled"].append(bool(is_tiled))
        return res

    evaluator_module._run_detect = run_detect
    rc = p2l.run_arm_child(
        "A_L", out / "mot", out / "A_L.sidecar.json", tuple(sequences)
    )
    for f in files.values():
        f.close()
    for seq, m in meta.items():
        path = out / "rows" / seq / "detector.bin"
        _write_json(
            out / "rows" / seq / "meta.json",
            {
                "format": ROWS_FORMAT,
                "sequence": seq,
                "records": m["records"],
                "frames": m["frames"],
                "any_tiled": any(m["is_tiled"]),
                "dtypes": m["dtypes"],
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            },
        )
    return rc


def run_oracle_rows(args: argparse.Namespace) -> int:
    att = read_attestation()
    out: Path = args.out
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().relative_to(project_root)),
        "_oracle-rows-child",
        str(out.relative_to(project_root)),
        ",".join(args.sequences),
        str(args.max_frames or 0),
    ]
    with (out / "A_L.stdout.log").open("w") as log:
        rc = subprocess.run(
            cmd, cwd=project_root, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    p2l = load_pr2l(att["op_library"]["sha256"])
    sidecar_path = out / "A_L.sidecar.json"
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else None
    problems = [] if rc == 0 else [f"A_L child exited {rc}"]
    problems += [f"V5: {p}" for p in p2l.sidecar_problems("A_L", sidecar)]
    if (sidecar or {}).get("mot17_argv") and "--double-buffer" in sidecar["mot17_argv"]:
        problems.append("the oracle-rows run was not serial")
    ingest = _load_module(INGEST_HARNESS, "native_ingest_parity")
    seqs = {}
    for seq in args.sequences:
        meta_path = out / "rows" / seq / "meta.json"
        if not meta_path.exists():
            problems.append(f"{seq}: no rows recorded")
            continue
        m = json.loads(meta_path.read_text())
        _, _, frame_end = ingest.oracle_sequence_bounds(
            project_root / DATA_ROOT / SPLIT / seq, args.max_frames
        )
        if m["frames"] != list(range(1, frame_end + 1)):
            problems.append(f"{seq}: recorded frames are not 1..{frame_end}")
        if m["any_tiled"]:
            problems.append(f"{seq}: is_tiled was true")
        seqs[seq] = {k: m[k] for k in ("records", "sha256", "bytes", "dtypes")}
    report = {
        "format": ROWS_FORMAT,
        "generated_utc": _utc(),
        "git": _git_state(),
        "attestation_sha256": _sha256_file(project_root / ATTESTATION),
        "mot17_argv": (sidecar or {}).get("mot17_argv"),
        "sequences": seqs,
        "problems": problems,
        "ok": not problems,
    }
    _write_json(out / "oracle_rows.json", report)
    print(f"oracle-rows: {'OK' if not problems else problems}")
    return 0 if not problems else 2


def read_oracle_rows(rows_dir: Path, seq: str) -> dict[int, tuple[Any, Any, Any]]:
    import numpy as np

    meta = json.loads((rows_dir / "rows" / seq / "meta.json").read_text())
    if meta.get("format") != ROWS_FORMAT:
        raise RuntimeError(f"{rows_dir}/{seq}: not an oracle-rows dump")
    path = rows_dir / "rows" / seq / "detector.bin"
    if _sha256_file(path) != meta["sha256"]:
        raise RuntimeError(f"{path}: sha256 differs from its meta.json")
    data = path.read_bytes()
    out: dict[int, tuple[Any, Any, Any]] = {}
    off = 0
    while off < len(data):
        frame, n, tiled = DET_RECORD.unpack_from(data, off)
        off += DET_RECORD.size
        if tiled:
            raise RuntimeError(f"{path}: frame {frame} is tiled")
        b = np.frombuffer(data, np.float32, n * 4, off).reshape(n, 4)
        off += n * 16
        s = np.frombuffer(data, np.float32, n, off)
        off += n * 4
        c = np.frombuffer(data, np.int32, n, off)
        off += n * 4
        out[frame] = (b, s, c)
    return out


# ── comparisons ────────────────────────────────────────────────────────────────


def _ordered(bits: Any) -> Any:
    """float32 bit patterns (int32) -> integers whose difference counts ulps."""
    import torch

    b = bits.to(torch.int64)
    return torch.where(b < 0, -(b & 0x7FFFFFFF), b)


def compare_tensor(native: Any, oracle: Any) -> dict[str, Any]:
    """Two float32 tensors (any device), bit for bit."""
    import torch

    if native.shape != oracle.shape:
        return {
            "equal": False,
            "shape": [list(native.shape), list(oracle.shape)],
            "differing_values": max(native.numel(), oracle.numel()),
        }
    a = native.contiguous().view(torch.int32)
    b = oracle.contiguous().view(torch.int32)
    if torch.equal(a, b):
        return {"equal": True}
    ne = a != b
    d = (native.double() - oracle.double()).abs()
    finite = torch.isfinite(d)
    return {
        "equal": False,
        "differing_values": int(ne.sum()),
        "max_abs": float(d[finite & ne].max()) if bool((finite & ne).any()) else None,
        "max_ulp": int((_ordered(a) - _ordered(b)).abs().max()),
        "nonfinite_differing": int((ne & ~finite).sum()),
    }


def compare_rows(
    native: tuple[Any, Any, Any], oracle: tuple[Any, Any, Any]
) -> dict[str, Any]:
    """Detector rows (boxes [n,4] f32, scores [n] f32, classes [n] i32; numpy)."""
    import numpy as np

    nb, ns, nc = native
    ob, os_, oc = oracle
    n, m = len(ns), len(os_)

    def keys(b: Any, s: Any, c: Any) -> list[bytes]:
        return [
            b[i].tobytes() + s[i : i + 1].tobytes() + c[i : i + 1].tobytes()
            for i in range(len(s))
        ]

    def zero(b: Any, s: Any, c: Any) -> int:
        """Rows whose six values are all +0 (an unfilled ``new_zeros`` row)."""
        return sum(
            1
            for i in range(len(s))
            if not b[i].view(np.int32).any()
            and s[i : i + 1].view(np.int32)[0] == 0
            and c[i] == 0
        )

    nk, ok = keys(nb, ns, nc), keys(ob, os_, oc)
    r: dict[str, Any] = {
        "rows": [n, m],
        "count_equal": n == m,
        "membership_equal": collections.Counter(nk) == collections.Counter(ok),
        "order_equal": nk == ok,
        "zero_rows": [zero(nb, ns, nc), zero(ob, os_, oc)],
    }
    if n == m:
        r["classes_equal"] = bool(np.array_equal(nc, oc))
        r["scores_bits_equal"] = bool(
            np.array_equal(ns.view(np.int32), os_.view(np.int32))
        )
        r["boxes_bits_equal"] = bool(
            np.array_equal(nb.view(np.int32), ob.view(np.int32))
        )
        if not r["order_equal"]:
            r["first_differing_row"] = next(i for i in range(n) if nk[i] != ok[i])
        for name, a, b in (("scores", ns, os_), ("boxes", nb, ob)):
            ai, bi = (
                a.view(np.int32).astype(np.int64),
                b.view(np.int32).astype(np.int64),
            )
            ne = ai != bi
            if ne.any():
                oa = np.where(ai < 0, -(ai & 0x7FFFFFFF), ai)
                obb = np.where(bi < 0, -(bi & 0x7FFFFFFF), bi)
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                r[f"{name}_differing_values"] = int(ne.sum())
                r[f"{name}_max_abs"] = float(np.nanmax(np.where(ne, d, 0.0)))
                r[f"{name}_max_ulp"] = int(np.abs(oa - obb)[ne].max())
        if not r["classes_equal"]:
            r["classes_differing"] = int((nc != oc).sum())
    r["equal"] = r["order_equal"] and r["count_equal"]
    return r


class Section:
    """One section's per-frame results over a sequence."""

    def __init__(self, rows: bool) -> None:
        self.rows = rows
        self.frames = 0
        self.equal_frames = 0
        self.agg: dict[str, Any] = collections.defaultdict(int)
        self.max_abs: float | None = None
        self.max_ulp = 0
        self.first_differing: list[dict[str, Any]] = []

    def add(self, frame: int, r: dict[str, Any]) -> None:
        self.frames += 1
        if r["equal"]:
            self.equal_frames += 1
            return
        if self.rows:
            for k in (
                "count_equal",
                "membership_equal",
                "order_equal",
                "classes_equal",
                "scores_bits_equal",
                "boxes_bits_equal",
            ):
                if r.get(k) is False:
                    self.agg[f"frames_{k.replace('_equal', '')}_differ"] += 1
            for k in ("scores", "boxes"):
                self.agg[f"{k}_differing_values"] += r.get(f"{k}_differing_values", 0)
                if f"{k}_max_abs" in r:
                    self.max_abs = max(self.max_abs or 0.0, r[f"{k}_max_abs"])
                    self.max_ulp = max(self.max_ulp, r[f"{k}_max_ulp"])
            self.agg["classes_differing"] += r.get("classes_differing", 0)
        else:
            self.agg["differing_values"] += r["differing_values"]
            if r.get("max_abs") is not None:
                self.max_abs = max(self.max_abs or 0.0, r["max_abs"])
            self.max_ulp = max(self.max_ulp, r.get("max_ulp", 0))
        if len(self.first_differing) < FIRST_DIFFS_KEPT:
            self.first_differing.append({"frame": frame, **r})

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "verdict": "EXACT" if self.equal_frames == self.frames else "DIFFERS",
            "frames": self.frames,
            "equal_frames": self.equal_frames,
        }
        if self.equal_frames != self.frames:
            out.update(
                dict(self.agg),
                max_abs=self.max_abs,
                max_ulp=self.max_ulp,
                first_differing=self.first_differing,
            )
        return out


def merge_sections(parts: list[dict[str, Any]]) -> dict[str, Any]:
    frames = sum(p["frames"] for p in parts)
    equal = sum(p["equal_frames"] for p in parts)
    out: dict[str, Any] = {
        "verdict": "EXACT" if frames == equal and frames > 0 else "DIFFERS",
        "frames": frames,
        "equal_frames": equal,
    }
    if frames != equal:
        first = next((p for p in parts if p["verdict"] != "EXACT"), None)
        keys = {
            k
            for p in parts
            for k in p
            if k.endswith(("_differ", "_values", "differing")) and isinstance(p[k], int)
        }
        for k in sorted(keys):
            out[k] = sum(p.get(k, 0) for p in parts)
        abs_vals = [p["max_abs"] for p in parts if p.get("max_abs") is not None]
        out["max_abs"] = max(abs_vals) if abs_vals else None
        out["max_ulp"] = max(p.get("max_ulp", 0) for p in parts)
        out["first_differing_sequence"] = first.get("sequence") if first else None
    return out


# ── parity ─────────────────────────────────────────────────────────────────────


class Oracle:
    """The A_L detector's stages, callable on any stage input."""

    def __init__(self, att: dict[str, Any], cfg: dict[str, Any]) -> None:
        build_path = Path(os.environ.get("SACCADE_BUILD_PATH", project_root / "build"))
        if build_path.exists():
            sys.path.insert(0, str(build_path))
        import saccade.perception.detector_trt  # noqa: F401  (before torchvision)
        import torch

        from saccade.perception.temporal_yolo import mamba_gated_detector as mgd

        self.torch = torch
        self.mgd = mgd
        b = cfg["host_params"]["detector"]["build"]
        self.img_size = int(b["img_size"])
        self.conf_thr = float(b["conf_thr"])
        self.max_det = int(b["max_det"])
        self.small_p3 = float(b["small_p3_max_threshold"])
        engine = project_root / b["trt_backbone_engine"]
        lineage = json.loads((project_root / LINEAGE).read_text())
        if _sha256_file(engine) != lineage["companions"]["backbone_engine"]["sha256"]:
            raise RuntimeError(f"{engine}: sha256 differs from the lineage")
        self.p2l = load_pr2l(att["op_library"]["sha256"])
        module, record = self.p2l.load_l(
            torch
        )  # op library, optimize off, artifact checks
        self.artifact_record = record
        self.head = self.p2l.LibTorchHeadAdapter(module, torch)
        bad = self.p2l.runtime_requirement_problems(self.p2l.runtime_readback(torch))
        if bad:
            raise RuntimeError(f"runtime requirements: {bad}")
        self.backbone = mgd.TRTYoloBackbone(str(engine))
        # MambaGatedDetector.__init__ / set_whole_graph_img_dims, as the oracle builds them.
        dev = "cuda"
        self.stride = torch.tensor([8.0, 16.0, 32.0], device=dev)
        s = self.img_size
        ch = lineage["head_load"]["in_channels"]
        feat_shapes = [
            (1, ch[0], s // 8, s // 8),
            (1, ch[1], s // 16, s // 16),
            (1, ch[2], s // 32, s // 32),
        ]
        self.anchors, self.anchor_strides = mgd._precompute_anchor_grid(
            self.stride, feat_shapes
        )
        self.sx = torch.ones(1, device=dev)
        self.sy = torch.ones(1, device=dev)
        self.x_idx = torch.tensor([0, 2], device=dev, dtype=torch.long)
        self.y_idx = torch.tensor([1, 3], device=dev, dtype=torch.long)
        self.nms_pad = 0  # MambaGatedDetector._whole_graph_nms_pad (pinned)

    def set_image_dims(self, h: int, w: int) -> None:
        self.sx.fill_(w / self.img_size)
        self.sy.fill_(h / self.img_size)

    def resize(self, frame_chw: Any) -> Any:
        F = self.torch.nn.functional
        return F.interpolate(
            frame_chw.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

    def backbone_features(self, frame_640: Any) -> list[Any]:
        p3, p4, p5 = self.backbone.infer_graph(frame_640)
        return [p3.clone(), p4.clone(), p5.clone()]

    def head_outputs(self, feats: list[Any]) -> list[Any]:
        cls_preds, reg_preds = self.head.infer_graph(*feats)
        return [*cls_preds, *reg_preds]

    def s2(self, cls_preds: list[Any], reg_preds: list[Any]) -> tuple[Any, Any]:
        return oracle_s2(
            self.mgd,
            cls_preds,
            reg_preds,
            stride=self.stride,
            conf_thr=self.conf_thr,
            max_det=self.max_det,
            nms_pad=self.nms_pad,
            anchors=self.anchors,
            anchor_strides=self.anchor_strides,
            small_p3=self.small_p3,
            sx=self.sx,
            sy=self.sy,
            x_idx=self.x_idx,
            y_idx=self.y_idx,
        )


def oracle_s2(
    mgd: Any,
    cls_preds: list[Any],
    reg_preds: list[Any],
    *,
    stride: Any,
    conf_thr: float,
    max_det: int,
    nms_pad: int,
    anchors: Any,
    anchor_strides: Any,
    small_p3: float,
    sx: Any,
    sy: Any,
    x_idx: Any,
    y_idx: Any,
    compile_: bool | None = None,
) -> tuple[Any, Any]:
    """``_whole_graph_fn``'s tail (pinned by test_native_detector_oracle_pins):
    the S2 call -- compiled unless ``compile_`` is False -- then the coordinate
    scaling. Returns (before scaling, after scaling), both (1, rows, 6)."""
    detections = mgd._postprocess_mamba_fixed(
        cls_preds,
        reg_preds,
        stride,
        conf_thr,
        max(max_det, nms_pad),
        anchors=anchors,
        anchor_strides=anchor_strides,
        small_p3_max_threshold=small_p3,
        box_scale_x=sx,
        box_scale_y=sy,
        _compile=compile_,
    )
    raw = detections.clone()
    detections[:, :, x_idx] *= sx
    detections[:, :, y_idx] *= sy
    return raw, detections


def _np_payload(rec: dict[str, Any], fh: IO[bytes], ingest: Any) -> dict[str, Any]:
    import numpy as np

    dtypes = {"u8": np.uint8, "f32": np.float32, "i32": np.int32}
    out = {}
    for p in rec["payloads"]:
        arr = np.empty(
            p["bytes"] // np.dtype(dtypes[p["dtype"]]).itemsize,
            dtype=dtypes[p["dtype"]],
        )
        ingest._read_exact(fh, memoryview(arr).cast("B"))
        out[p["name"]] = arr.reshape(p["shape"])
    return out


def run_sequence(
    args: argparse.Namespace,
    seq: str,
    oracle: Oracle,
    att: dict[str, Any],
    lineage: dict[str, Any],
    frames_out: IO[str],
) -> dict[str, Any]:
    import numpy as np
    import torch

    from saccade.perception.eval.streaming import TorchvisionGpuStreamer

    ingest = _load_module(INGEST_HARNESS, "native_ingest_parity")
    seq_path = project_root / DATA_ROOT / SPLIT / seq
    w, h, frame_end = ingest.oracle_sequence_bounds(seq_path, args.max_frames)
    streamer = TorchvisionGpuStreamer(seq_path / "img1")
    oracle_files = [Path(f).name for f in streamer.img_files]
    rows_ref = read_oracle_rows(args.oracle_rows, seq) if args.oracle_rows else None

    cmd = [
        str(project_root / PROBE),
        "--config", str(project_root / RESOLVED_CONFIG),
        "--lineage", str(project_root / LINEAGE),
        "--attestation", str(project_root / ATTESTATION),
        "--model-root", str(project_root),
        "--sequence", str(seq_path),
        "--mode", "stages",
        "--mutation", args.mutation,
    ]  # fmt: skip
    if args.max_frames:
        cmd += ["--max-frames", str(args.max_frames)]
    stderr_path = args.out / f"probe_{seq}.stderr"
    sections = {k: Section(rows=False) for k in TENSOR_SECTIONS}
    sections.update(
        {k: Section(rows=True) for k in ROW_SECTIONS if k != "detector" or rows_ref}
    )
    with stderr_path.open("wb") as err:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=err, bufsize=1 << 20
        )
        assert proc.stdout is not None
        try:
            pre = ingest.read_record(proc.stdout)
            problems = preamble_problems(
                pre, att, lineage, w, h, oracle_files[:frame_end], args.mutation
            )
            if problems:
                raise RuntimeError(f"{seq}: {problems}")
            oracle.set_image_dims(h, w)
            oracle_buffer = torch.zeros((3, h, w), dtype=torch.float32, device="cuda")
            oracle_iter = iter(streamer)
            paths: dict[str, int] = {}
            with torch.no_grad():
                for k in range(1, frame_end + 1):
                    rec = ingest.read_record(proc.stdout)
                    if rec.get("frame") != k or rec["file"] != oracle_files[k - 1]:
                        raise RuntimeError(
                            f"{seq}: unexpected probe record {rec.get('frame')} {rec.get('file')}"
                        )
                    nat = _np_payload(rec, proc.stdout, ingest)
                    paths[rec["decode_path"]] = paths.get(rec["decode_path"], 0) + 1
                    g = {n: torch.from_numpy(a).to("cuda") for n, a in nat.items()
                         if n not in ("boxes", "scores", "classes")}  # fmt: skip

                    frame_hwc = next(oracle_iter)
                    oracle_ingest_u8 = frame_hwc.permute(2, 0, 1).contiguous()
                    ingest.oracle_ingest(frame_hwc, oracle_buffer)
                    res: dict[str, dict[str, Any]] = {}
                    dec = ingest.compare_decoded(g["decoded_u8"], oracle_ingest_u8)
                    res["decoder"] = {
                        **dec,
                        "differing_values": dec.get("differing_values", 0),
                    }
                    res["decoder"].pop("abs_hist", None)
                    res["ingest"] = compare_tensor(g["frame"], oracle_buffer)
                    o_resized = oracle.resize(g["frame"])
                    res["resize"] = compare_tensor(g["resized"], o_resized)
                    o_feats = oracle.backbone_features(g["resized"].contiguous())
                    res["backbone"] = _compare_many(
                        [g[n] for n in ("p3", "p4", "p5")], o_feats, ("p3", "p4", "p5")
                    )
                    o_head = oracle.head_outputs([g["p3"], g["p4"], g["p5"]])
                    res["head"] = _compare_many(
                        [g[n] for n in HEAD_NAMES], o_head, HEAD_NAMES
                    )
                    o_raw, o_scaled = oracle.s2(
                        [g[n] for n in HEAD_NAMES[:3]], [g[n] for n in HEAD_NAMES[3:]]
                    )
                    res["s2_raw"] = compare_tensor(g["s2_raw"], o_raw[0])
                    res["s2_scaled"] = compare_tensor(g["s2_scaled"], o_scaled[0])
                    # detect_single_patch_640 -> _run_native_tensor_prep (dump det_rows).
                    o_rows = (
                        np.ascontiguousarray(o_scaled[0, :, :4].float().cpu().numpy()),
                        np.ascontiguousarray(o_scaled[0, :, 4].float().cpu().numpy()),
                        np.ascontiguousarray(
                            o_scaled[0, :, 5].to(torch.int32).cpu().numpy()
                        ),
                    )
                    n_rows = (nat["boxes"], nat["scores"], nat["classes"])
                    res["s2_rows"] = compare_rows(n_rows, o_rows)
                    if rows_ref is not None:
                        if k not in rows_ref:
                            raise RuntimeError(f"{seq}: oracle rows have no frame {k}")
                        res["detector"] = compare_rows(n_rows, rows_ref[k])
                    for name, r in res.items():
                        sections[name].add(k, r)
                    frames_out.write(
                        json.dumps(
                            {
                                "sequence": seq,
                                "frame": k,
                                "file": rec["file"],
                                "decode_path": rec["decode_path"],
                                "equal": {n: r["equal"] for n, r in res.items()},
                                "sha256": _frame_hashes(
                                    nat, o_head, o_scaled, rows_ref, k
                                ),
                            }
                        )
                        + "\n"
                    )
                    if args.progress and k % 100 == 0:
                        print(f"  {seq} {k}/{frame_end}", file=sys.stderr, flush=True)
            end = ingest.read_record(proc.stdout)
            if end != {"end": True, "frames": frame_end}:
                raise RuntimeError(f"{seq}: unexpected end record {end}")
        finally:
            streamer._stop_worker()
            proc.stdout.close()
            rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{seq}: probe exited {rc} (see {stderr_path.name})")
    return {
        "im_width": w,
        "im_height": h,
        "frames": frame_end,
        "decode_paths": paths,
        "load": pre["load"],
        "plan": pre["plan"],
        **{k: s.summary() for k, s in sections.items()},
    }


def _compare_many(
    native: list[Any], oracle: list[Any], names: tuple[str, ...]
) -> dict[str, Any]:
    parts = {n: compare_tensor(a, b) for n, a, b in zip(names, native, oracle)}
    if all(p["equal"] for p in parts.values()):
        return {"equal": True}
    bad = {n: p for n, p in parts.items() if not p["equal"]}
    abs_vals = [p["max_abs"] for p in bad.values() if p.get("max_abs") is not None]
    return {
        "equal": False,
        "differing_values": sum(p["differing_values"] for p in bad.values()),
        "max_abs": max(abs_vals) if abs_vals else None,
        "max_ulp": max(p.get("max_ulp", 0) for p in bad.values()),
        "outputs": bad,
    }


def _frame_hashes(
    nat: dict[str, Any], o_head: list[Any], o_scaled: Any, rows_ref: Any, k: int
) -> dict[str, str]:
    def h(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    out = {
        f"native_{n}": h(nat[n].tobytes())
        for n in ("resized", "p3", "p4", "p5", *HEAD_NAMES, "s2_raw", "s2_scaled", "boxes", "scores", "classes")
    }  # fmt: skip
    out["oracle_head"] = h(b"".join(t.cpu().numpy().tobytes() for t in o_head))
    out["oracle_s2_scaled"] = h(o_scaled.cpu().numpy().tobytes())
    if rows_ref is not None:
        out["oracle_rows"] = h(b"".join(a.tobytes() for a in rows_ref[k]))
    return out


def preamble_problems(
    pre: dict[str, Any],
    att: dict[str, Any],
    lineage: dict[str, Any],
    w: int,
    h: int,
    frames: list[str],
    mutation: str,
) -> list[str]:
    """Validity of one probe run, from its preamble (fail closed)."""
    problems = []
    if pre.get("format") != STREAM_FORMAT or pre.get("mode") != "stages":
        problems.append(f"stream {pre.get('format')!r} mode {pre.get('mode')!r}")
    if (pre.get("im_width"), pre.get("im_height")) != (w, h):
        problems.append("geometry differs from seqinfo.ini")
    if pre.get("frames") != frames:
        problems.append("native frame listing differs from the oracle's")
    if pre.get("mutation") != mutation:
        problems.append(f"probe mutation {pre.get('mutation')!r} != {mutation!r}")
    if pre.get("python_libraries_mapped") != []:
        problems.append(
            f"Python mapped in the probe: {pre.get('python_libraries_mapped')}"
        )
    load = pre.get("load", {})
    plan = pre.get("plan", {})
    if load.get("op_library_sha256") != att["op_library"]["sha256"]:
        problems.append("probe loaded another operator library")
    if not plan.get("op_library", {}).get("from_attestation"):
        problems.append(
            "probe did not bind the operator library through the attestation"
        )
    if load.get("head_artifact_sha256") != lineage["torchscript"]["sha256"]:
        problems.append("probe loaded another head artifact")
    if (
        load.get("backbone_engine_sha256")
        != lineage["companions"]["backbone_engine"]["sha256"]
    ):
        problems.append("probe loaded another backbone engine")
    if load.get("runtime_readback") != lineage["runtime_requirements"]:
        problems.append("probe runtime requirements differ from the lineage's")
    if load.get("native_scan_calls") != lineage["torchscript"]["native_scan_calls"]:
        problems.append("probe head graph scan-call count differs from the lineage's")
    if load.get("param_devices") != ["cuda:0"] or any(
        d != "cpu" for d in load.get("constant_devices", ["?"])
    ):
        problems.append("probe head placement is not parameters cuda:0 / constants cpu")
    return problems


def compare_against(frames_path: Path, prior: Path) -> dict[str, Any]:
    def load(p: Path) -> dict[tuple[str, int], dict[str, str]]:
        return {
            (r["sequence"], r["frame"]): r["sha256"]
            for r in map(json.loads, p.read_text().splitlines())
        }

    a, b = load(frames_path), load(prior.parent / "frames.jsonl")
    if a.keys() != b.keys():
        return {"identical": False, "reason": "different frame sets"}
    diffs = [
        {
            "sequence": s,
            "frame": f,
            "keys": sorted(k for k in a[(s, f)] if a[(s, f)][k] != b[(s, f)].get(k)),
        }
        for (s, f) in sorted(a)
        if a[(s, f)] != b[(s, f)]
    ]
    return {
        "identical": not diffs,
        "frames": len(a),
        "differing_frames": len(diffs),
        "first": diffs[:FIRST_DIFFS_KEPT],
    }


def run_parity(args: argparse.Namespace) -> int:
    import torch

    att = read_attestation()
    lineage = json.loads((project_root / LINEAGE).read_text())
    cfg = json.loads((project_root / RESOLVED_CONFIG).read_text())
    oracle = Oracle(att, cfg)
    rows_report = None
    if args.oracle_rows is not None:
        rows_report = json.loads((args.oracle_rows / "oracle_rows.json").read_text())
        if not rows_report.get("ok"):
            raise RuntimeError(f"{args.oracle_rows}: oracle-rows run is not ok")
    per_seq = {}
    with (args.out / "frames.jsonl").open("w") as frames_out:
        for seq in args.sequences:
            print(f"{seq} ...", file=sys.stderr, flush=True)
            per_seq[seq] = run_sequence(args, seq, oracle, att, lineage, frames_out)
    names = [*TENSOR_SECTIONS, "s2_rows"] + (["detector"] if args.oracle_rows else [])
    merged = {
        n: merge_sections([{**per_seq[s][n], "sequence": s} for s in args.sequences])
        for n in names
    }
    verdict = (
        "EXACT"
        if all(m["verdict"] == "EXACT" for m in merged.values())
        else "NOT_EXACT"
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": _utc(),
        "git": _git_state(),
        "host": platform.node(),
        "torch": torch.__version__,
        "attestation_sha256": _sha256_file(project_root / ATTESTATION),
        "lineage_sha256": _sha256_file(project_root / LINEAGE),
        "oracle_artifact": oracle.artifact_record,
        "oracle_rows": str(args.oracle_rows) if args.oracle_rows else None,
        "oracle_rows_report": rows_report,
        "sequences": args.sequences,
        "max_frames": args.max_frames,
        "mutation": args.mutation,
        "frames": sum(per_seq[s]["frames"] for s in args.sequences),
        "sections": merged,
        "verdict": verdict,
        "per_sequence": per_seq,
    }
    if args.mutation != "none":
        expected = NEGCTL_EXPECT[args.mutation]
        report["negative_control"] = {
            "mutation": args.mutation,
            "expected_section": expected,
            "caught": merged[expected]["verdict"] == "DIFFERS",
            "sections_differ": [
                n for n, m in merged.items() if m["verdict"] != "EXACT"
            ],
        }
    if args.against is not None:
        report["against"] = {
            "prior": str(args.against),
            **compare_against(args.out / "frames.jsonl", args.against),
        }
    _write_json(args.out / "report.json", report)
    line = " ".join(f"{n}={m['verdict']}" for n, m in merged.items())
    print(f"parity: {verdict} ({report['frames']} frames) {line}")
    if args.mutation != "none":
        nc = report["negative_control"]
        print(
            f"negative control {args.mutation}: {'CAUGHT' if nc['caught'] else 'NOT CAUGHT'}"
        )
        return 0 if nc["caught"] else 1
    ok = verdict == "EXACT" and (args.against is None or report["against"]["identical"])
    return 0 if ok else 1


# ── main ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    for p in (project_root, project_root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if argv[:1] == ["_anchor-child"]:
        return anchor_child(argv[1])
    if argv[:1] == ["_oracle-rows-child"]:
        return oracle_rows_child(
            argv[1], [s for s in argv[2].split(",") if s], int(argv[3]) or None
        )

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    seqs = lambda s: [x for x in s.split(",") if x]  # noqa: E731
    a = sub.add_parser("anchor", help="re-run the PR-2L A_L arm, compare with A_L_1")
    a.add_argument("--out", type=Path, required=True)
    t = sub.add_parser(
        "attest", help="write the realization attestation from an anchor report"
    )
    t.add_argument("--anchor", type=Path, required=True)
    t.add_argument("--write", type=Path, default=None)
    o = sub.add_parser("oracle-rows", help="record the A_L serial run's detector rows")
    o.add_argument("--out", type=Path, required=True)
    o.add_argument("--sequences", type=seqs, default=list(SEQUENCES))
    o.add_argument("--max-frames", type=int, default=None)
    q = sub.add_parser("parity", help="stage-separated native vs oracle comparison")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--oracle-rows", type=Path, default=None)
    q.add_argument("--sequences", type=seqs, default=list(SEQUENCES))
    q.add_argument("--max-frames", type=int, default=None)
    q.add_argument("--mutation", choices=["none", *NEGCTL_EXPECT], default="none")
    q.add_argument("--against", type=Path, default=None)
    q.add_argument("--progress", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "attest":
        return run_attest(args)

    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        ap.error(f"{args.out} is not empty")
    if not args.out.is_relative_to(project_root):
        ap.error(
            "--out must be inside the repository (the A_L child runs mot17.py from it)"
        )
    for attr in ("oracle_rows", "against"):
        if getattr(args, attr, None) is not None:
            setattr(args, attr, getattr(args, attr).resolve())
    if args.cmd == "parity" and not (project_root / PROBE).exists():
        ap.error(
            f"{PROBE} not built (cmake --build build --target saccade_detector_probe)"
        )
    from scripts.provenance.run_manifest import open_run

    # ADR 021 AP-2: claim the run directory before the first byte.
    open_run(
        args.out,
        produced_by="diagnostic",
        preset="mamba_whole_graph",
        detector="SDP",
        dataset=f"MOT17 {SPLIT}",
    )
    try:
        if args.cmd == "anchor":
            return run_anchor(args)
        if args.cmd == "oracle-rows":
            return run_oracle_rows(args)
        return run_parity(args)
    except Exception as exc:  # noqa: BLE001 -- reported, exit 2
        print(f"native_detector_parity: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
