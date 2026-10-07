#!/usr/bin/env python3
"""Native end-to-end track parity vs the A_L runs (#465 Phase B PR-9, PR-10).

Issue #465 Phase B PR-9 (U3b-3, serial; boundary §6 "7-seq MOT txt 對 Python
serial 組態") and PR-10 (U5, double buffer + CUDA graphs; "7-seq MOT txt 對 §2
oracle（double-buffer）"); measurement contracts:
docs/reference/native_runtime_resolved_config.md §13 and §14. Developer tooling
only. The oracle is ``A_L`` (PR-1L head injected by the frozen PR-2L runner's
functions, operator library bound by the realization attestation):

* ``--schedule serial`` (PR-9): the serial run recorded by
  ``native_detector_parity.py oracle-rows``: its ``mot/<seq>.txt``,
  ``mot/_global_id_map.txt`` and ``evaluator._run_detect`` rows
  (``rows/<seq>/detector.bin``);
* ``--schedule double_buffer`` (PR-10, the default): the MOT txt of the
  double-buffer run ``native_detector_parity.py anchor`` writes
  (``--oracle-txt``: ``A_L/<seq>.txt``, ``A_L/_global_id_map.txt``,
  ``A_L.stdout.log``); the detector rows are still the serial
  ``oracle-rows`` (the oracle's detector is frame-independent: the same
  ``_run_detect`` on the side stream).

The native side is ``build/shipping/saccade_track`` (own process, no Python):
every requested sequence in one process, in the oracle run's order, with
``--trace`` (each frame's detector rows) and ``--report``. #465 Phase C PR-C2
(docs §18): the shipping entrypoint has no developer option, so the harness
gives it only its interface (config, lineage, attestation, model root, out,
report, trace, sequences). Negative controls (``--mutation``), the serial
override (``--schedule serial``) and ``--max-frames`` need ``--entrypoint
measurement``: the developer build ``build/shipping/saccade_track_measurement``
(the measurement variant of the runtime libraries; not installed). The report
must name the entrypoint that was asked for, and a shipping report must carry
no ``measurement`` record. Sections:

``detector``        native detector rows at the end-to-end wiring vs the
                    oracle's ``_run_detect`` rows, per frame, bit for bit;
``mot_txt``         native ``<seq>.txt`` vs the oracle's, byte for byte after
                    relabeling (the oracle numbers ids run-globally; its ids
                    for a sequence are the per-sequence ids plus the ids
                    earlier sequences used, read from its
                    ``_global_id_map.txt``, which must list the sequence's ids
                    as one contiguous block), and the same number of track ids;
``graph_captures``  (double buffer only) per sequence, the native whole-detect,
                    main NMS and GMC graph captures vs the captures the oracle
                    logs in that sequence, and the native replay counts vs the
                    frames (the tracker graph's capture is not logged by the
                    oracle; its native count must be 1).

No tolerance: ``EXACT`` only when every section is, on every sequence.
``--no-trace`` runs ``saccade_track`` without ``--trace`` (the shipping
configuration: the trace reads each frame's rows back with an extra sync, which
could hide a race) and reports ``detector`` as ``NOT_RUN``.
``--mutation`` runs a ``saccade_track_measurement`` negative control of the
schedule's runtime; ``--ref-edit`` changes one character of the first
sequence's oracle txt in memory (comparator check); ``--against`` compares the
native txt and trace hashes, and (double buffer) the native graph counts, with
an earlier report. ``--track-binary`` runs another build of the entrypoint
(#465 PR-11: the ``-DSACCADE_WITH_OPENCV=OFF`` build).
#465 PR-12 (the installed shipping tree, docs §16): ``--model-root`` runs it on
the tree's model root (its config, lineage, attestation and the files they
bind) and ``--track-library-path`` gives that process alone an
``LD_LIBRARY_PATH`` (the tree's RUNPATH is ``$ORIGIN``-relative only);
``--native-from DIR`` runs nothing and judges a saccade_track run made
elsewhere with the same arguments (the clean container:
``scripts/native/run_shipping_container.sh``), from ``DIR/native``,
``DIR/trace`` and ``DIR/track_report.json``.

Usage (GPU, gpu0 lease; R=results/465_pr10_track/<label>)::

    .venv/bin/python scripts/eval/diagnostics/native_detector_parity.py \\
        anchor --out $R/anchor
    .venv/bin/python scripts/eval/diagnostics/native_detector_parity.py \\
        oracle-rows --out $R/oracle_rows
    .venv/bin/python scripts/eval/diagnostics/native_track_parity.py \\
        parity --out $R/parity --oracle-rows $R/oracle_rows --oracle-txt $R/anchor \\
        [--ref-edit] [--against $R/parity/report.json]
    .venv/bin/python scripts/eval/diagnostics/native_track_parity.py \\
        parity --entrypoint measurement --out $R/negctl --oracle-rows $R/oracle_rows \\
        --oracle-txt $R/anchor --mutation M
    .venv/bin/python scripts/eval/diagnostics/native_track_parity.py \\
        parity --entrypoint measurement --schedule serial --out $R/serial \\
        --oracle-rows $R/oracle_rows [--sequences S,..] [--max-frames N] [--mutation M]

Exit 0: EXACT / negative control caught; 1: a difference; 2: error.
"""
# status: diagnostic

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[3]

SCHEMA = "saccade.native_track_parity/v1"
TRACK_REPORT_FORMAT = "saccade.native_track_report/v2"
TRACK = "build/shipping/saccade_track"
MEASUREMENT_TRACK = "build/shipping/saccade_track_measurement"
# --entrypoint -> the name the report must carry, and the default binary.
ENTRYPOINTS = {
    "shipping": ("saccade_track", TRACK),
    "measurement": ("saccade_track_measurement", MEASUREMENT_TRACK),
}
DETECTOR_HARNESS = "scripts/eval/diagnostics/native_detector_parity.py"
INGEST_HARNESS = "scripts/eval/diagnostics/native_ingest_parity.py"
DET_RECORD = struct.Struct("<3i")  # frame, n, is_tiled
FIRST_DIFFS_KEPT = 20
SCHEDULES = ("double_buffer", "serial")
# Negative controls: the section that must report DIFFERS. The serial
# runtime's wiring controls (PR-9) and the double-buffer runtime's schedule /
# graph controls (PR-10); ref_edit is the comparator check of either.
NEGCTL_EXPECT = {
    "shared_post_host": "mot_txt",
    "stale_image_dims": "detector",
    "gmc_previous_frame": "mot_txt",
    "stale_detector_input": "detector",
    "stale_gmc_input": "mot_txt",
    "swapped_detection_parity": "detector",
    "ref_edit": "mot_txt",
}
NEGCTL_SCHEDULE = {
    "shared_post_host": "serial",
    "stale_image_dims": "serial",
    "gmc_previous_frame": "serial",
    "stale_detector_input": "double_buffer",
    "stale_gmc_input": "double_buffer",
    "swapped_detection_parity": "double_buffer",
}
# What the oracle logs when it captures a graph (pipeline.py / stages.py /
# mamba_gated_detector.py). The "[TrackerGraph] Captured" line is printed when
# GraphedTrackerUpdate is constructed, before its lazy capture, so it only
# marks where a sequence's log begins.
_SEQ_START = re.compile(r"\[TrackerGraph\] Captured tracker update for seq (\S+)")
_ORACLE_CAPTURES = {
    "detector": re.compile(r"\[WholeDetectGraph\] Capturing graphed callable"),
    "nms": re.compile(r"\[MainNMSGraphNoCopyback\] Captured main NMS nocopyback graph"),
    "gmc": re.compile(r"\[GMCGraph\] Captured C\+\+ cuFFT GMC graph for seq (\S+)"),
}


def _load_module(rel: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, project_root / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


det = _load_module(DETECTOR_HARNESS, "native_detector_parity")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ── detector rows ──────────────────────────────────────────────────────────────


def read_records(path: Path) -> list[tuple[int, int, bytes]]:
    """``(frame, n, record bytes)`` of a PR-5 ``detector.bin``."""
    data = path.read_bytes()
    out, off = [], 0
    while off < len(data):
        if len(data) - off < DET_RECORD.size:
            raise RuntimeError(f"{path}: truncated record header")
        frame, n, _tiled = DET_RECORD.unpack_from(data, off)
        end = off + DET_RECORD.size + n * 24  # boxes 16 + score 4 + class 4
        if end > len(data):
            raise RuntimeError(f"{path}: truncated record at frame {frame}")
        out.append((frame, n, data[off:end]))
        off = end
    return out


def compare_detector(native: Path, oracle: Path) -> dict[str, Any]:
    a, b = read_records(native), read_records(oracle)
    frames_a, frames_b = [r[0] for r in a], [r[0] for r in b]
    if frames_a != frames_b:
        return {
            "frames": len(a),
            "equal_frames": 0,
            "frame_sets_equal": False,
            "verdict": "DIFFERS",
            "first_differing": [],
        }
    diffs = []
    count_mismatch = 0
    for (frame, n_a, rec_a), (_, n_b, rec_b) in zip(a, b):
        if rec_a != rec_b:
            count_mismatch += int(n_a != n_b)
            diffs.append({"frame": frame, "rows": [n_a, n_b]})
    return {
        "frames": len(a),
        "equal_frames": len(a) - len(diffs),
        "frame_sets_equal": True,
        "count_mismatch_frames": count_mismatch,
        "first_differing": diffs[:FIRST_DIFFS_KEPT],
        "verdict": "EXACT" if not diffs else "DIFFERS",
    }


# ── MOT txt ────────────────────────────────────────────────────────────────────


def id_blocks(map_path: Path) -> dict[str, tuple[int, int]]:
    """``seq -> (offset, ids)`` from the oracle's ``_global_id_map.txt``
    (``<seq>\\tlocal_id=<L>\\tglobal_id=<G>``); each sequence's global ids must
    be one contiguous block."""
    ids: dict[str, list[int]] = {}
    for line in map_path.read_text().splitlines():
        seq, _local, glob = line.split("\t")
        if not glob.startswith("global_id="):
            raise RuntimeError(f"{map_path}: malformed line {line!r}")
        ids.setdefault(seq, []).append(int(glob[len("global_id=") :]))
    out = {}
    for seq, g in ids.items():
        g.sort()
        if g != list(range(g[0], g[0] + len(g))):
            raise RuntimeError(
                f"{seq}: its run-global ids are not one contiguous block"
            )
        out[seq] = (g[0] - 1, len(g))
    return out


def relabel(text: str, offset: int) -> str:
    """The oracle's lines with the id field (2nd) shifted back by ``offset``."""
    if not text:
        return text
    out = []
    for line in text.split("\n"):
        f = line.split(",", 2)
        if len(f) < 3:
            raise RuntimeError(f"reference MOT line without an id field: {line!r}")
        out.append(f"{f[0]},{int(f[1]) - offset},{f[2]}")
    return "\n".join(out)


def compare_mot(native: str, oracle_relabeled: str) -> dict[str, Any]:
    a, b = native.split("\n"), oracle_relabeled.split("\n")
    first = None
    if native != oracle_relabeled:
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        first = {
            "line": i,
            "native": a[i] if i < len(a) else None,
            "oracle": b[i] if i < len(b) else None,
        }
    return {
        "lines": len(a) if native else 0,
        "reference_lines": len(b) if oracle_relabeled else 0,
        "native_sha256": _sha256_bytes(native.encode()),
        "byte_identical_after_relabel": native == oracle_relabeled,
        "first_difference": first,
    }


# ── graph captures (double buffer) ─────────────────────────────────────────────


def oracle_graph_captures(log_text: str) -> dict[str, dict[str, int]]:
    """``seq -> {detector, nms, gmc}`` capture counts from the oracle's log,
    attributed to the sequence whose log block they fall in."""
    out: dict[str, dict[str, int]] = {}
    seq = None
    for line in log_text.splitlines():
        m = _SEQ_START.search(line)
        if m:
            seq = m.group(1)
            out.setdefault(seq, {k: 0 for k in _ORACLE_CAPTURES})
            continue
        for name, pat in _ORACLE_CAPTURES.items():
            g = pat.search(line)
            if not g:
                continue
            if seq is None:
                raise RuntimeError(f"oracle log: a {name} capture before any sequence")
            if name == "gmc" and g.group(1) != seq:
                raise RuntimeError(
                    f"oracle log: GMC capture for {g.group(1)} inside {seq}"
                )
            out[seq][name] += 1
    return out


def compare_graphs(
    native: dict[str, Any], oracle: dict[str, int] | None
) -> dict[str, Any]:
    """One sequence's native graph counts (track report) vs the oracle log."""
    g = native["graphs"]
    frames, updates = native["frames"], native["tracker_updates"]
    problems = []
    if oracle is None:
        problems.append("no oracle log block for the sequence")
    else:
        for key, ours in (
            ("detector", g["detector_captures"]),
            ("nms", g["nms_captures"]),
            ("gmc", g["gmc_captures"]),
        ):
            if ours != oracle[key]:
                problems.append(f"{key} captures {ours} != oracle {oracle[key]}")
    if g["tracker_captures"] != (1 if updates > 0 else 0):
        problems.append(f"tracker captures {g['tracker_captures']}")
    expected = {
        "detector_replays": frames,
        "nms_replays": updates,
        "tracker_replays": updates,
        "gmc_replays": max(updates - 1, 0),
    }
    for key, want in expected.items():
        if g[key] != want:
            problems.append(f"{key} {g[key]} != {want}")
    return {
        "native": g,
        "oracle": oracle,
        "problems": problems,
        "verdict": "EXACT" if not problems else "DIFFERS",
    }


# ── validity ───────────────────────────────────────────────────────────────────


def expected_measurement(
    mutation: str, max_frames: int | None, schedule: str
) -> dict[str, Any]:
    """The ``measurement`` record saccade_track_measurement writes for the
    options run_track gives it."""
    return {
        "mutation": mutation,
        "schedule_override": "serial" if schedule == "serial" else None,
        "max_frames": max_frames or 0,
    }


def report_problems(
    rep: dict[str, Any],
    att: dict[str, Any],
    lineage: dict[str, Any],
    sequences: list[str],
    entrypoint: str,
    mutation: str,
    max_frames: int | None,
    schedule: str = "serial",
) -> list[str]:
    """Validity of the saccade_track run, from its report (fail closed)."""
    problems = []
    if rep.get("format") != TRACK_REPORT_FORMAT or rep.get("schedule") != schedule:
        problems.append(
            f"report {rep.get('format')!r} schedule {rep.get('schedule')!r}"
        )
    name = ENTRYPOINTS[entrypoint][0]
    if rep.get("entrypoint") != name:
        problems.append(f"report entrypoint {rep.get('entrypoint')!r} != {name!r}")
    if entrypoint == "shipping":
        if "measurement" in rep:
            problems.append("a shipping saccade_track report with a measurement record")
    elif rep.get("measurement") != expected_measurement(mutation, max_frames, schedule):
        problems.append(
            f"saccade_track_measurement measurement {rep.get('measurement')!r} "
            "differs from the request"
        )
    if rep.get("python_libraries_mapped") != []:
        problems.append(
            f"Python mapped in saccade_track: {rep.get('python_libraries_mapped')}"
        )
    if rep.get("sequence_order") != sequences:
        problems.append("saccade_track ran the sequences in another order")
    plan = rep.get("detector", {}).get("plan", {})
    load = rep.get("detector", {}).get("load", {})
    if load.get("op_library_sha256") != att["op_library"]["sha256"]:
        problems.append("saccade_track loaded another operator library")
    if not plan.get("op_library", {}).get("from_attestation"):
        problems.append(
            "saccade_track did not bind the operator library through the attestation"
        )
    if load.get("head_artifact_sha256") != lineage["torchscript"]["sha256"]:
        problems.append("saccade_track loaded another head artifact")
    if (
        load.get("backbone_engine_sha256")
        != lineage["companions"]["backbone_engine"]["sha256"]
    ):
        problems.append("saccade_track loaded another backbone engine")
    if load.get("runtime_readback") != lineage["runtime_requirements"]:
        problems.append("saccade_track runtime requirements differ from the lineage's")
    if load.get("native_scan_calls") != lineage["torchscript"]["native_scan_calls"]:
        problems.append(
            "saccade_track head graph scan-call count differs from the lineage's"
        )
    if load.get("param_devices") != ["cuda:0"] or any(
        d != "cpu" for d in load.get("constant_devices", ["?"])
    ):
        problems.append(
            "saccade_track head placement is not parameters cuda:0 / constants cpu"
        )
    return problems


# ── parity ─────────────────────────────────────────────────────────────────────


def run_track(args: argparse.Namespace, out: Path) -> tuple[int, Path]:
    report = out / "track_report.json"
    root = args.model_root
    # The shipping interface (saccade_track has nothing else, PR-C2).
    cmd = [
        str(project_root / args.track_binary),
        "--config", str(root / det.RESOLVED_CONFIG),
        "--lineage", str(root / det.LINEAGE),
        "--attestation", str(root / det.ATTESTATION),
        "--model-root", str(root),
        "--out", str(out / "native"),
        "--report", str(report),
    ]  # fmt: skip
    if not args.no_trace:
        cmd += ["--trace", str(out / "trace")]
    if args.entrypoint == "measurement":
        cmd += ["--measurement-mutation", args.mutation]
        if args.schedule == "serial":
            cmd += ["--schedule", "serial"]
        if args.max_frames:
            cmd += ["--max-frames", str(args.max_frames)]
    cmd += [str(Path(det.DATA_ROOT) / det.SPLIT / s) for s in args.sequences]
    env = None
    if args.track_library_path is not None:
        env = {**os.environ, "LD_LIBRARY_PATH": str(args.track_library_path)}
    with (out / "saccade_track.log").open("w") as log:
        rc = subprocess.run(
            cmd, cwd=project_root, stdout=log, stderr=subprocess.STDOUT, env=env
        ).returncode
    return rc, report


def compare_against(report: dict[str, Any], prior_path: Path) -> dict[str, Any]:
    prior = json.loads(prior_path.read_text())
    diffs = []
    for seq, cur in report["per_sequence"].items():
        old = prior["per_sequence"].get(seq)
        for key in ("native_txt_sha256", "native_trace_sha256"):
            if cur.get(key) is None:  # --no-trace
                continue
            if old is None or old.get(key) != cur[key]:
                diffs.append({"sequence": seq, "key": key})
        # The native graph capture / replay counts (double buffer, PR-C2).
        if "graph_captures" in cur:
            ours = cur["graph_captures"]["native"]
            theirs = (old or {}).get("graph_captures", {}).get("native")
            if ours != theirs:
                diffs.append({"sequence": seq, "key": "graph_captures.native"})
    same_set = set(prior["per_sequence"]) == set(report["per_sequence"])
    return {
        "identical": same_set and not diffs,
        "same_sequences": same_set,
        "differing": diffs,
    }


def oracle_txt_problems(
    anchor_dir: Path, anchor: dict[str, Any], sequences: list[str], head: str
) -> list[str]:
    """Validity of the double-buffer oracle (an ``anchor`` run)."""
    problems = []
    if anchor.get("schema") != det.ANCHOR_SCHEMA:
        problems.append(f"{anchor_dir}: not an anchor report")
    if anchor.get("problems"):
        problems.append(f"{anchor_dir}: anchor problems {anchor['problems']}")
    if anchor.get("identical") is not True:
        problems.append(f"{anchor_dir}: the A_L re-run is not identical to PR-2L A_L_1")
    argv = anchor.get("mot17_argv") or []
    if "--double-buffer" not in argv:
        problems.append("the oracle-txt run was not double-buffer")
    order = (
        argv[argv.index("--sequences") + 1].split(",") if "--sequences" in argv else []
    )
    if order != sequences:
        problems.append(f"oracle-txt sequence order {order} != {sequences}")
    if anchor.get("git", {}).get("dirty") or anchor.get("git", {}).get("head") != head:
        problems.append("the oracle-txt run is not at this clean commit")
    return problems


def run_parity(args: argparse.Namespace) -> int:
    att = det.read_attestation()
    lineage = json.loads((project_root / det.LINEAGE).read_text())
    rows_dir: Path = args.oracle_rows
    rows_report = json.loads((rows_dir / "oracle_rows.json").read_text())
    double_buffer = args.schedule == "double_buffer"
    problems = []
    if not rows_report.get("ok"):
        problems.append(f"{rows_dir}: oracle-rows run is not ok")
    argv = rows_report.get("mot17_argv") or []
    if "--double-buffer" in argv:
        problems.append("the oracle run was not serial")
    oracle_order = (
        argv[argv.index("--sequences") + 1].split(",") if "--sequences" in argv else []
    )
    if oracle_order != args.sequences:
        problems.append(f"oracle run sequence order {oracle_order} != {args.sequences}")
    oracle_max = (
        int(argv[argv.index("--max-frames") + 1]) if "--max-frames" in argv else None
    )
    if oracle_max != args.max_frames:
        problems.append(f"oracle --max-frames {oracle_max} != {args.max_frames}")
    txt_dir, map_path = rows_dir / "mot", rows_dir / "mot" / "_global_id_map.txt"
    anchor: dict[str, Any] = {}
    oracle_graphs: dict[str, dict[str, int]] = {}
    if double_buffer:
        anchor = json.loads((args.oracle_txt / "anchor.json").read_text())
        problems += oracle_txt_problems(
            args.oracle_txt, anchor, args.sequences, det._git_state()["head"]
        )
        txt_dir = args.oracle_txt / "A_L"
        map_path = txt_dir / "_global_id_map.txt"
        oracle_graphs = oracle_graph_captures(
            (args.oracle_txt / "A_L.stdout.log").read_text(errors="replace")
        )

    out: Path = args.out
    if args.native_from is not None:
        # A run made elsewhere (the clean container): judge its files.
        native = args.native_from
        log = (native / "saccade_track.log").read_text(errors="replace")
        rc = 0 if log.rstrip().endswith("exit=0") else 1
        track_report_path = native / "track_report.json"
    else:
        native = out
        rc, track_report_path = run_track(args, out)
    if rc != 0:
        problems.append(f"saccade_track exited {rc}")
    rep = (
        json.loads(track_report_path.read_text()) if track_report_path.exists() else {}
    )
    if rep:
        problems += report_problems(
            rep,
            att,
            lineage,
            args.sequences,
            args.entrypoint,
            args.mutation,
            args.max_frames,
            args.schedule,
        )

    ingest = _load_module(INGEST_HARNESS, "native_ingest_parity")
    blocks = id_blocks(map_path)
    per_seq: dict[str, Any] = {}
    for i, seq in enumerate(args.sequences):
        if seq not in rep.get("sequences", {}):
            problems.append(f"{seq}: saccade_track wrote nothing")
            continue
        st = rep["sequences"][seq]
        w, h, frame_end = ingest.oracle_sequence_bounds(
            project_root / det.DATA_ROOT / det.SPLIT / seq, args.max_frames
        )
        if (st["im_width"], st["im_height"], st["frames"]) != (w, h, frame_end):
            problems.append(
                f"{seq}: native geometry / frame count differs from seqinfo.ini"
            )
        meta = json.loads((rows_dir / "rows" / seq / "meta.json").read_text())
        oracle_bin = rows_dir / "rows" / seq / "detector.bin"
        if det._sha256_file(oracle_bin) != meta["sha256"]:
            problems.append(f"{oracle_bin}: sha256 differs from its meta.json")
        native_bin = native / "trace" / seq / "detector.bin"
        native_txt_path = native / "native" / f"{seq}.txt"
        native_txt = native_txt_path.read_text()
        oracle_txt = (txt_dir / f"{seq}.txt").read_text()
        if args.ref_edit and i == 0:
            # Comparator check: one character of the oracle's first line.
            j = oracle_txt.index(",", oracle_txt.index(",") + 1) + 1
            oracle_txt = (
                oracle_txt[:j]
                + ("1" if oracle_txt[j] != "1" else "2")
                + oracle_txt[j + 1 :]
            )
        offset, oracle_ids = blocks.get(seq, (0, 0))
        mot = compare_mot(native_txt, relabel(oracle_txt, offset))
        mot.update(
            {
                "track_ids": st["track_ids"],
                "reference_track_ids": oracle_ids,
                "reference_id_offset": offset,
                "interpolation": st["interpolation"],
            }
        )
        mot["verdict"] = (
            "EXACT"
            if mot["byte_identical_after_relabel"] and st["track_ids"] == oracle_ids
            else "DIFFERS"
        )
        detector = (
            {"verdict": "NOT_RUN", "frames": 0, "equal_frames": 0}
            if args.no_trace
            else compare_detector(native_bin, oracle_bin)
        )
        loop = st.get("loop_seconds") or 0.0
        per_seq[seq] = {
            "frames": st["frames"],
            "tracker_updates": st["tracker_updates"],
            "skipped_empty_frames": st["skipped_empty_frames"],
            "pre_roll_updates": st["pre_roll_updates"],
            "decodes": {
                "hardware": st["hardware_decodes"],
                "decoupled": st["decoupled_decodes"],
            },
            "detector": detector,
            "mot_txt": mot,
            "native_txt_sha256": det._sha256_file(native_txt_path),
            "native_trace_sha256": None
            if args.no_trace
            else det._sha256_file(native_bin),
            "oracle_txt_sha256": det._sha256_file(txt_dir / f"{seq}.txt"),
            "oracle_serial_txt_sha256": det._sha256_file(
                rows_dir / "mot" / f"{seq}.txt"
            ),
            "native_loop_seconds": loop,
            "native_loop_fps": st["frames"] / loop if loop > 0 else None,
        }
        if double_buffer:
            per_seq[seq]["graph_captures"] = compare_graphs(st, oracle_graphs.get(seq))

    sections: dict[str, Any] = {}
    names = ("detector", "mot_txt") + (("graph_captures",) if double_buffer else ())
    for name in names:
        if name == "detector" and args.no_trace:
            sections[name] = {"verdict": "NOT_RUN"}
            continue
        differ = [s for s in per_seq if per_seq[s][name]["verdict"] != "EXACT"]
        sections[name] = {
            "sequences": len(per_seq),
            "sequences_exact": len(per_seq) - len(differ),
            "sequences_differ": differ,
            "verdict": "EXACT"
            if not differ and len(per_seq) == len(args.sequences)
            else "DIFFERS",
        }
    if not args.no_trace:
        sections["detector"]["frames"] = sum(
            p["detector"]["frames"] for p in per_seq.values()
        )
        sections["detector"]["equal_frames"] = sum(
            p["detector"]["equal_frames"] for p in per_seq.values()
        )
    if problems:
        verdict = "UNRESOLVED"
    elif all(s["verdict"] in ("EXACT", "NOT_RUN") for s in sections.values()):
        verdict = "EXACT"
    else:
        verdict = "NOT_EXACT"
    # Observations, not gates: the serial oracle's txt vs PR-2L's A_L_1
    # (double-buffer) reference, and (double buffer) the double-buffer oracle's
    # txt vs the serial oracle's.
    a_l_1 = {
        seq: per_seq[seq]["oracle_serial_txt_sha256"]
        == det.PR2L_A_L_REFERENCE_TXT_SHA256.get(seq)
        for seq in per_seq
    }
    db_vs_serial = (
        {
            seq: per_seq[seq]["oracle_txt_sha256"]
            == per_seq[seq]["oracle_serial_txt_sha256"]
            for seq in per_seq
        }
        if double_buffer
        else None
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": det._utc(),
        "git": det._git_state(),
        "attestation_sha256": det._sha256_file(project_root / det.ATTESTATION),
        "lineage_sha256": det._sha256_file(project_root / det.LINEAGE),
        "entrypoint": args.entrypoint,
        "track_binary": str(args.track_binary),
        "track_binary_sha256": det._sha256_file(project_root / args.track_binary),
        "model_root": str(args.model_root),
        "track_library_path": args.track_library_path,
        "native_from": str(args.native_from) if args.native_from else None,
        "schedule": args.schedule,
        "oracle_rows": str(rows_dir),
        "oracle_rows_report": rows_report,
        "oracle_txt": str(args.oracle_txt) if double_buffer else None,
        "oracle_txt_report": anchor or None,
        "sequences": args.sequences,
        "max_frames": args.max_frames,
        "mutation": args.mutation,
        "ref_edit": args.ref_edit,
        "trace": not args.no_trace,
        "track_report": rep,
        "problems": problems,
        "sections": sections,
        "verdict": verdict,
        "observation_oracle_txt_equals_pr2l_a_l_1": a_l_1
        if not args.max_frames
        else None,
        "observation_oracle_double_buffer_txt_equals_serial": db_vs_serial,
        "per_sequence": per_seq,
    }
    negctl = "ref_edit" if args.ref_edit else args.mutation
    if negctl != "none":
        expected = NEGCTL_EXPECT[negctl]
        report["negative_control"] = {
            "name": negctl,
            "expected_section": expected,
            "caught": not problems and sections[expected]["verdict"] == "DIFFERS",
            "sections_differ": [
                n for n, s in sections.items() if s["verdict"] == "DIFFERS"
            ],
        }
    if args.against is not None:
        report["against"] = {
            "prior": str(args.against),
            **compare_against(report, args.against),
        }
    det._write_json(out / "report.json", report)
    line = " ".join(f"{n}={s['verdict']}" for n, s in sections.items())
    print(f"track parity: {verdict} {line} {problems if problems else ''}")
    if negctl != "none":
        nc = report["negative_control"]
        print(
            f"negative control {negctl}: {'CAUGHT' if nc['caught'] else 'NOT CAUGHT'}"
        )
        return 0 if nc["caught"] else 1
    if problems:
        return 2
    ok = verdict == "EXACT" and (args.against is None or report["against"]["identical"])
    return 0 if ok else 1


# ── main ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    for p in (project_root, project_root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    seqs = lambda s: [x for x in s.split(",") if x]  # noqa: E731
    q = sub.add_parser("parity", help="end-to-end native vs A_L comparison")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--schedule", choices=SCHEDULES, default="double_buffer")
    q.add_argument("--oracle-rows", type=Path, required=True)
    q.add_argument(
        "--oracle-txt",
        type=Path,
        default=None,
        help="an anchor run (double buffer): its A_L/<seq>.txt is the MOT oracle",
    )
    q.add_argument("--sequences", type=seqs, default=list(det.SEQUENCES))
    q.add_argument("--max-frames", type=int, default=None)
    neg = q.add_mutually_exclusive_group()
    neg.add_argument(
        "--mutation",
        choices=["none", *(k for k in NEGCTL_EXPECT if k != "ref_edit")],
        default="none",
    )
    neg.add_argument("--ref-edit", action="store_true")
    q.add_argument("--against", type=Path, default=None)
    q.add_argument("--no-trace", action="store_true")
    q.add_argument(
        "--entrypoint",
        choices=sorted(ENTRYPOINTS),
        default="shipping",
        help="shipping: saccade_track, its interface only; measurement: the "
        "developer build saccade_track_measurement (needed for --mutation, "
        "--schedule serial and --max-frames)",
    )
    q.add_argument(
        "--track-binary",
        type=Path,
        default=None,
        help="the binary to run (relative to the repository root; default: the "
        "entrypoint's build/shipping/ target)",
    )
    q.add_argument(
        "--model-root",
        type=Path,
        default=Path("."),
        help="the model root saccade_track reads (config, lineage, attestation and "
        "the files they bind at their repository-relative paths)",
    )
    q.add_argument(
        "--track-library-path",
        default=None,
        help="LD_LIBRARY_PATH for the saccade_track process only",
    )
    q.add_argument(
        "--native-from",
        type=Path,
        default=None,
        help="judge this saccade_track run (DIR/native, DIR/trace, "
        "DIR/track_report.json, DIR/saccade_track.log ending exit=0) instead of "
        "running one; --track-binary names the binary that ran it",
    )
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if args.entrypoint == "shipping" and (
        args.mutation != "none" or args.schedule == "serial" or args.max_frames
    ):
        ap.error(
            "--mutation, --schedule serial and --max-frames are not options of the "
            "shipping saccade_track: use --entrypoint measurement"
        )
    if args.track_binary is None:
        args.track_binary = Path(ENTRYPOINTS[args.entrypoint][1])

    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        ap.error(f"{args.out} is not empty")
    args.oracle_rows = args.oracle_rows.resolve()
    if args.schedule == "double_buffer":
        if args.oracle_txt is None:
            ap.error("--schedule double_buffer needs --oracle-txt (an anchor run)")
        if args.max_frames:
            ap.error("the double-buffer oracle (anchor) runs whole sequences")
        args.oracle_txt = args.oracle_txt.resolve()
    elif args.oracle_txt is not None:
        ap.error("--oracle-txt is the double-buffer oracle")
    if args.no_trace and (args.mutation != "none" or args.ref_edit):
        ap.error("--no-trace is the shipping configuration: no negative control")
    if args.mutation != "none" and NEGCTL_SCHEDULE[args.mutation] != args.schedule:
        ap.error(
            f"--mutation {args.mutation} is a {NEGCTL_SCHEDULE[args.mutation]} control"
        )
    if args.against is not None:
        args.against = args.against.resolve()
    if args.native_from is not None:
        if args.entrypoint != "shipping" or args.no_trace:
            ap.error("--native-from judges a plain traced shipping run")
        args.native_from = args.native_from.resolve()
    elif not (project_root / args.track_binary).exists():
        ap.error(
            f"{args.track_binary} not built (cmake --build build --target "
            f"{ENTRYPOINTS[args.entrypoint][0]})"
        )
    from scripts.provenance.run_manifest import open_run

    # ADR 021 AP-2: claim the run directory before the first byte.
    open_run(
        args.out,
        produced_by="diagnostic",
        preset="mamba_whole_graph",
        detector="SDP",
        dataset=f"MOT17 {det.SPLIT}",
    )
    try:
        return run_parity(args)
    except Exception as exc:  # noqa: BLE001 -- reported, exit 2
        print(f"native_track_parity: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
