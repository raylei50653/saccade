#!/usr/bin/env python3
"""Native end-to-end serial track parity vs the A_L serial run (#465 Phase B PR-9).

Issue #465 Phase B PR-9 (U3b-3; boundary §6 "7-seq MOT txt 對 Python serial
組態"; measurement contract: docs/reference/native_runtime_resolved_config.md
§13). Developer tooling only. The oracle is ``A_L`` in the serial
configuration, recorded by ``native_detector_parity.py oracle-rows`` (one
``mot17.py`` run without ``--double-buffer``, PR-1L head injected by the frozen
PR-2L runner's functions, operator library bound by the realization
attestation): its ``mot/<seq>.txt`` and ``mot/_global_id_map.txt``, and its
``evaluator._run_detect`` rows (``rows/<seq>/detector.bin``).

The native side is ``build/shipping/saccade_track`` (own process, no Python):
every requested sequence in one process, in the oracle run's order, with
``--trace`` (each frame's detector rows) and ``--report``. Sections:

``detector``  native detector rows at the end-to-end wiring vs the oracle
              run's ``_run_detect`` rows, per frame, bit for bit;
``mot_txt``   native ``<seq>.txt`` vs the oracle's, byte for byte after
              relabeling (the oracle numbers ids run-globally; its ids for a
              sequence are the per-sequence ids plus the ids earlier sequences
              used, read from its ``_global_id_map.txt``, which must list the
              sequence's ids as one contiguous block), and the same number of
              track ids.

No tolerance: ``EXACT`` only when both sections are, on every sequence.
``--mutation`` runs a ``saccade_track`` wiring negative control;
``--ref-edit`` changes one character of the first sequence's oracle txt in
memory (comparator check); ``--against`` compares the native txt and trace
hashes with an earlier report.

Usage (GPU, gpu0 lease; R=results/465_pr9_track/<label>)::

    .venv/bin/python scripts/eval/diagnostics/native_detector_parity.py \\
        oracle-rows --out $R/oracle_rows
    .venv/bin/python scripts/eval/diagnostics/native_track_parity.py \\
        parity --out $R/parity --oracle-rows $R/oracle_rows \\
        [--sequences S,..] [--max-frames N] [--mutation M | --ref-edit] \\
        [--against $R/parity/report.json]

Exit 0: EXACT / negative control caught; 1: a difference; 2: error.
"""
# status: diagnostic

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[3]

SCHEMA = "saccade.native_track_parity/v1"
TRACK_REPORT_FORMAT = "saccade.native_track_report/v1"
TRACK = "build/shipping/saccade_track"
DETECTOR_HARNESS = "scripts/eval/diagnostics/native_detector_parity.py"
INGEST_HARNESS = "scripts/eval/diagnostics/native_ingest_parity.py"
DET_RECORD = struct.Struct("<3i")  # frame, n, is_tiled
FIRST_DIFFS_KEPT = 20
# Wiring negative controls: the section that must report DIFFERS.
NEGCTL_EXPECT = {
    "shared_post_host": "mot_txt",
    "stale_image_dims": "detector",
    "gmc_previous_frame": "mot_txt",
    "ref_edit": "mot_txt",
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


# ── validity ───────────────────────────────────────────────────────────────────


def report_problems(
    rep: dict[str, Any],
    att: dict[str, Any],
    lineage: dict[str, Any],
    sequences: list[str],
    mutation: str,
    max_frames: int | None,
) -> list[str]:
    """Validity of the saccade_track run, from its report (fail closed)."""
    problems = []
    if rep.get("format") != TRACK_REPORT_FORMAT or rep.get("schedule") != "serial":
        problems.append(
            f"report {rep.get('format')!r} schedule {rep.get('schedule')!r}"
        )
    if rep.get("mutation") != mutation:
        problems.append(
            f"saccade_track mutation {rep.get('mutation')!r} != {mutation!r}"
        )
    if rep.get("max_frames") != (max_frames or 0):
        problems.append("saccade_track max_frames differs from the request")
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
    cmd = [
        str(project_root / TRACK),
        "--config", det.RESOLVED_CONFIG,
        "--lineage", det.LINEAGE,
        "--attestation", det.ATTESTATION,
        "--model-root", ".",
        "--out", str(out / "native"),
        "--report", str(report),
        "--trace", str(out / "trace"),
        "--measurement-mutation", args.mutation,
    ]  # fmt: skip
    if args.max_frames:
        cmd += ["--max-frames", str(args.max_frames)]
    cmd += [str(Path(det.DATA_ROOT) / det.SPLIT / s) for s in args.sequences]
    with (out / "saccade_track.log").open("w") as log:
        rc = subprocess.run(
            cmd, cwd=project_root, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    return rc, report


def compare_against(report: dict[str, Any], prior_path: Path) -> dict[str, Any]:
    prior = json.loads(prior_path.read_text())
    diffs = []
    for seq, cur in report["per_sequence"].items():
        old = prior["per_sequence"].get(seq)
        for key in ("native_txt_sha256", "native_trace_sha256"):
            if old is None or old.get(key) != cur[key]:
                diffs.append({"sequence": seq, "key": key})
    same_set = set(prior["per_sequence"]) == set(report["per_sequence"])
    return {
        "identical": same_set and not diffs,
        "same_sequences": same_set,
        "differing": diffs,
    }


def run_parity(args: argparse.Namespace) -> int:
    att = det.read_attestation()
    lineage = json.loads((project_root / det.LINEAGE).read_text())
    rows_dir: Path = args.oracle_rows
    rows_report = json.loads((rows_dir / "oracle_rows.json").read_text())
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

    out: Path = args.out
    rc, track_report_path = run_track(args, out)
    if rc != 0:
        problems.append(f"saccade_track exited {rc}")
    rep = (
        json.loads(track_report_path.read_text()) if track_report_path.exists() else {}
    )
    if rep:
        problems += report_problems(
            rep, att, lineage, args.sequences, args.mutation, args.max_frames
        )

    ingest = _load_module(INGEST_HARNESS, "native_ingest_parity")
    blocks = id_blocks(rows_dir / "mot" / "_global_id_map.txt")
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
        native_bin = out / "trace" / seq / "detector.bin"
        native_txt_path = out / "native" / f"{seq}.txt"
        native_txt = native_txt_path.read_text()
        oracle_txt = (rows_dir / "mot" / f"{seq}.txt").read_text()
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
        detector = compare_detector(native_bin, oracle_bin)
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
            "native_trace_sha256": det._sha256_file(native_bin),
            "oracle_txt_sha256": det._sha256_file(rows_dir / "mot" / f"{seq}.txt"),
        }

    sections = {}
    for name in ("detector", "mot_txt"):
        differ = [s for s in per_seq if per_seq[s][name]["verdict"] != "EXACT"]
        sections[name] = {
            "sequences": len(per_seq),
            "sequences_exact": len(per_seq) - len(differ),
            "sequences_differ": differ,
            "verdict": "EXACT"
            if not differ and len(per_seq) == len(args.sequences)
            else "DIFFERS",
        }
    sections["detector"]["frames"] = sum(
        p["detector"]["frames"] for p in per_seq.values()
    )
    sections["detector"]["equal_frames"] = sum(
        p["detector"]["equal_frames"] for p in per_seq.values()
    )
    if problems:
        verdict = "UNRESOLVED"
    elif all(s["verdict"] == "EXACT" for s in sections.values()):
        verdict = "EXACT"
    else:
        verdict = "NOT_EXACT"
    # Observation, not a gate: the serial oracle's txt vs PR-2L's A_L_1
    # (double-buffer) reference.
    a_l_1 = {
        seq: per_seq[seq]["oracle_txt_sha256"]
        == det.PR2L_A_L_REFERENCE_TXT_SHA256.get(seq)
        for seq in per_seq
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": det._utc(),
        "git": det._git_state(),
        "attestation_sha256": det._sha256_file(project_root / det.ATTESTATION),
        "lineage_sha256": det._sha256_file(project_root / det.LINEAGE),
        "track_binary_sha256": det._sha256_file(project_root / TRACK),
        "oracle_rows": str(rows_dir),
        "oracle_rows_report": rows_report,
        "sequences": args.sequences,
        "max_frames": args.max_frames,
        "mutation": args.mutation,
        "ref_edit": args.ref_edit,
        "track_report": rep,
        "problems": problems,
        "sections": sections,
        "verdict": verdict,
        "observation_oracle_txt_equals_pr2l_a_l_1": a_l_1
        if not args.max_frames
        else None,
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
                n for n, s in sections.items() if s["verdict"] != "EXACT"
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
    q = sub.add_parser("parity", help="end-to-end native vs A_L serial comparison")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--oracle-rows", type=Path, required=True)
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
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        ap.error(f"{args.out} is not empty")
    args.oracle_rows = args.oracle_rows.resolve()
    if args.against is not None:
        args.against = args.against.resolve()
    if not (project_root / TRACK).exists():
        ap.error(f"{TRACK} not built (cmake --build build --target saccade_track)")
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
