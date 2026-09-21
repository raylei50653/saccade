"""Full-stack leave-one-out subtraction harness for the production preset (#418)."""

# status: diagnostic
#
# Runs `scripts/eval/mot17.py` for the production configuration (FULL) and for a
# pre-declared set of subtractions (FULL-A), pair subtractions (FULL-A-B),
# historical reference points and runtime-only toggles, then reduces the run
# records into one summary (JSON + markdown) read under rules that are fixed in
# this file before the first run.  It does not own benchmark numbers: FULL is
# checked against the #421 production formal run by MOT-file md5, and every
# other row is an observed difference against FULL, never an effect claim.
#
# Rows are read at print precision against the larger of the two sides'
# same-session repeat ranges (n runs, k distinct outputs).  Materiality and
# role rules (`RULES`) are pre-declared; they classify observed differences,
# they do not test hypotheses.  A subtraction whose remaining stack was tuned
# for the removed piece (e.g. `--no-multiplicative-cost` under multiplicative
# knob values) carries its confound on the row instead of a retune.
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[3]
for _extra in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts" / "tools"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from scripts.provenance.training_eval_contract import (  # noqa: E402
    parse_overall_metrics,
    parse_overall_throughput,
)
from eval_repeat_identity import list_mot_files, mot_md5  # noqa: E402

SCHEMA = "saccade-module-interactions-subtraction-v1"
MOT17_ENTRY = REPO_ROOT / "scripts" / "eval" / "mot17.py"
PRESET = "mamba_whole_graph"
PRESET_PATH = REPO_ROOT / "configs" / "presets" / f"{PRESET}.yaml"
SEQUENCES = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]
# The #421 production formal run this FULL row must reproduce (same commit for
# every decision-relevant source; see report_data/training_eval_campaign.json).
REFERENCE_RUN = (
    "results/training_eval_contract/wg_mamba_whole_graph_s.t3t1_phase_b/"
    "formal-20260919T092203Z-r01"
)
REFERENCE_RECIPE = "wg:mamba_whole_graph:s.t3t1_phase_b"

QUALITY_METRICS = [
    "HOTA",
    "IDF1",
    "AssA",
    "DetA",
    "MOTA",
    "IDs",
    "FP",
    "FN",
    "Rcll",
    "Prcn",
]
PRINT_PRECISION = {"IDs": 1.0, "FP": 1.0, "FN": 1.0}
DEFAULT_PRECISION = 0.1

# --------------------------------------------------------------------------- pre-declared rules
RULES: dict[str, Any] = {
    "delta_direction": "variant minus FULL (a negative quality delta means the removal hurt)",
    "repeat_range": (
        "each side's same-session observed range (max-min over its runs); a delta "
        "is 'resolved' when |delta| exceeds the larger of the two sides' ranges "
        "and is above print precision; an observed range of 0.0 bounds nothing"
    ),
    "materiality": {
        "quality_axes": ["IDF1", "HOTA", "AssA", "MOTA"],
        "quality_threshold": 0.2,
        "ids_threshold": 10,
        "note": (
            "|delta| > 0.2 on IDF1/HOTA/AssA/MOTA or |delta IDs| > 10, and resolved "
            "against the repeat ranges, counts as material; the 0.2 figure is the "
            "evidence ledger's decision-knob noise guidance, not a significance test"
        ),
    },
    "runtime": {
        "identity": "all seven MOT md5s equal to FULL's (bit-identical output)",
        "fps_threshold_pct": 2.0,
        "note": (
            "a runtime row is 'runtime-only' when output is identical and the serial "
            "fps delta exceeds 2% of FULL and the two sides' fps ranges do not overlap; "
            "fps here is the serial whole-graph profile, not the double-buffer headline; "
            "when a late same-session FULL control exists the row must also clear its range"
        ),
    },
    "single_roles": {
        "essential": "material negative on IDF1 or HOTA and no material positive on IDF1/HOTA/MOTA",
        "trade_off": "material negative on one quality axis and material positive on another",
        "conflict": "material positive on IDF1 or HOTA and no material negative on IDF1/HOTA/MOTA",
        "no_material_quality_role": "no material delta on any axis (role then rests on pairs, runtime, safety or dependency)",
    },
    "pair_reading": {
        "definition": "I(A,B) = M(FULL) - M(FULL-A) - M(FULL-B) + M(FULL-A-B) on IDF1 and HOTA",
        "threshold": 0.2,
        "substitute_backup": "I < -0.2: each covers for the other; removing both loses more than the sum",
        "complement_dependency": "I > +0.2: A's contribution needs B (or vice versa); removing both loses less than the sum",
        "additive": "|I| <= 0.2: independent contributions at this resolution",
    },
    "reference_rows": "not subtractions: retuned or historical values quoted beside a subtraction to name its confound",
}

# --------------------------------------------------------------------------- pre-declared variants
# kind: full | quality | pair | reference | runtime
# argv: extra CLI flags on top of the preset; env: process env overrides;
# config: YAML keys the CLI cannot reach (the run uses a derived --config file
# equal to the preset with these keys replaced; mot17.py applies it at the same
# priority position the preset occupies).
VARIANTS: list[dict[str, Any]] = [
    {
        "id": "FULL",
        "kind": "full",
        "family": "reference",
        "argv": [],
        "env": {},
        "removes": [],
        "note": "production preset, serial (identity of the #421 production row)",
    },
    # --- single subtractions -------------------------------------------------
    {
        "id": "FULL-GMC",
        "kind": "quality",
        "family": "motion",
        "argv": ["--no-gmc"],
        "env": {},
        "removes": ["GMC"],
        "note": "global motion compensation off (tracker sees raw camera motion)",
    },
    {
        "id": "FULL-KalmanR",
        "kind": "quality",
        "family": "motion",
        "argv": ["--kalman-r-scale", "1.0"],
        "env": {},
        "removes": ["KalmanR"],
        "note": "measurement-noise scale 2.8 -> 1.0 (unscaled R); raw parser default is 0.75",
    },
    {
        "id": "FULL-OccState",
        "kind": "quality",
        "family": "occlusion",
        "argv": ["--no-occ-state-enabled"],
        "env": {},
        "removes": ["OccState"],
        "note": "same-height front-latch occlusion gate off",
    },
    {
        "id": "FULL-OAO",
        "kind": "quality",
        "family": "occlusion",
        "argv": ["--oao-tau", "0"],
        "env": {},
        "removes": ["OAO"],
        "note": "OAO penalty off entirely (ramp has nothing to scale)",
    },
    {
        "id": "FULL-OAORamp",
        "kind": "quality",
        "family": "occlusion",
        "argv": ["--oao-ramp-frames", "0"],
        "env": {},
        "removes": ["OAORamp"],
        "note": "duration ramp off at the ramped tau 0.50 (confound: plain OAO shipped at tau 0.25)",
    },
    {
        "id": "FULL-Bridge",
        "kind": "quality",
        "family": "recovery",
        "argv": ["--no-relink-bridge-enabled"],
        "env": {},
        "removes": ["Bridge"],
        "note": "tracker-core bidirectional foot-bridge relink off",
    },
    {
        "id": "FULL-BridgeMargin",
        "kind": "quality",
        "family": "recovery",
        "argv": ["--relink-bridge-margin", "0"],
        "env": {},
        "removes": ["BridgeMargin"],
        "note": "bridge reciprocal-margin guard off (ambiguous bridges accepted)",
    },
    {
        "id": "FULL-BridgeDirBonus",
        "kind": "quality",
        "family": "recovery",
        "argv": ["--relink-bridge-dir-bonus", "0"],
        "env": {},
        "removes": ["BridgeDirBonus"],
        "note": "bridge directional-consistency bonus off (s preset value 0.8; m ships 0.0)",
    },
    {
        "id": "FULL-Interp",
        "kind": "quality",
        "family": "recovery",
        "argv": ["--no-interpolate-tracklets"],
        "env": {},
        "removes": ["Interp"],
        "note": "output-layer gap interpolation off (max_gap 35)",
    },
    {
        "id": "FULL-PrivCont",
        "kind": "quality",
        "family": "recovery",
        "argv": ["--no-private-continuation"],
        "env": {},
        "removes": ["PrivCont"],
        "note": "track-gated wider-NMS continuation candidates off",
    },
    {
        "id": "FULL-StabCost",
        "kind": "quality",
        "family": "association",
        "argv": ["--stability-cost-w", "0"],
        "env": {},
        "removes": ["StabCost"],
        "note": "height-stability reward in the multiplicative cost off",
    },
    {
        "id": "FULL-StabBid",
        "kind": "quality",
        "family": "association",
        "argv": [],
        "env": {"SACCADE_STABILITY_W": "0"},
        "removes": ["StabBid"],
        "note": "auction height-stability bid off (env default 0.1)",
    },
    {
        "id": "FULL-DDA",
        "kind": "quality",
        "family": "association",
        "argv": [],
        "env": {"SACCADE_ENABLE_DDA": "0"},
        "removes": ["DDA"],
        "note": "S0 unambiguous pre-stage (cost <= 0.12, high-conf, confirmed) off",
    },
    {
        "id": "FULL-MultCost",
        "kind": "quality",
        "family": "association",
        "argv": ["--no-multiplicative-cost"],
        "env": {},
        "removes": ["MultCost"],
        "note": "additive clamp chain instead of log-linear cost (confound: knobs tuned for the multiplicative form; stability cost has no additive path)",
    },
    # --- reference points (not subtractions) --------------------------------
    {
        "id": "REF-OAOPlain025",
        "kind": "reference",
        "family": "occlusion",
        "argv": ["--oao-ramp-frames", "0", "--oao-tau", "0.25"],
        "env": {},
        "removes": ["OAORamp"],
        "note": "historical plain-OAO operating point (tau 0.25, no ramp): names the FULL-OAORamp tau confound",
    },
    # --- pairs (filled by --pairs; defaults below are the a-priori candidates) --
    {
        "id": "FULL-Bridge-Interp",
        "kind": "pair",
        "family": "recovery",
        "argv": ["--no-relink-bridge-enabled", "--no-interpolate-tracklets"],
        "env": {},
        "removes": ["Bridge", "Interp"],
        "note": "both gap-recovery layers off: suspected backup pair",
    },
    {
        "id": "FULL-OccState-OAO",
        "kind": "pair",
        "family": "occlusion",
        "argv": ["--no-occ-state-enabled", "--oao-tau", "0"],
        "env": {},
        "removes": ["OccState", "OAO"],
        "note": "both occlusion cost terms off: suspected overlap",
    },
    {
        "id": "FULL-StabCost-StabBid",
        "kind": "pair",
        "family": "association",
        "argv": ["--stability-cost-w", "0"],
        "env": {"SACCADE_STABILITY_W": "0"},
        "removes": ["StabCost", "StabBid"],
        "note": "dual height stability both off (re-reads the 2026-07-09 4-way on the current identity)",
    },
    {
        "id": "FULL-GMC-KalmanR",
        "kind": "pair",
        "family": "motion",
        "argv": ["--no-gmc", "--kalman-r-scale", "1.0"],
        "env": {},
        "removes": ["GMC", "KalmanR"],
        "note": "motion model both off: does R scaling absorb residual camera motion?",
    },
    {
        "id": "FULL-Bridge-PrivCont",
        "kind": "pair",
        "family": "recovery",
        "argv": ["--no-relink-bridge-enabled", "--no-private-continuation"],
        "env": {},
        "removes": ["Bridge", "PrivCont"],
        "note": "identity recovery vs continuation recall: suspected orthogonal",
    },
    # --- runtime-only toggles (output must stay identical) ------------------
    {
        "id": "RT-NoWholeGraph",
        "kind": "runtime",
        "family": "runtime",
        "argv": [],
        "env": {},
        "config": {"use_whole_graph": False},
        "removes": ["WholeGraph"],
        "note": "head-only CUDA graph instead of whole-detect graph (checkpoint temporal path left as the builder resolves it)",
    },
    {
        "id": "RT-NoWholeGraph-T1",
        "kind": "runtime",
        "family": "runtime",
        "argv": ["--no-temporal"],
        "env": {},
        "config": {"use_whole_graph": False},
        "removes": ["WholeGraph"],
        "note": "head-only CUDA graph with the temporal path pinned to T=1 (the whole-graph forward bypasses the checkpoint's temporal blocks; this is the pure scheduling subtraction)",
    },
    {
        "id": "RT-NoDetectGraph-T1",
        "kind": "runtime",
        "family": "runtime",
        "argv": ["--no-temporal"],
        "env": {},
        "config": {"use_whole_graph": False, "use_cuda_graph": False},
        "removes": ["WholeGraph", "HeadGraph"],
        "note": "eager detect path with the temporal path pinned to T=1",
    },
    {
        "id": "RT-NoDDA",
        "kind": "runtime",
        "family": "runtime",
        "argv": [],
        "env": {"SACCADE_ENABLE_DDA": "0"},
        "removes": ["DDA"],
        "note": "S0 unambiguous pre-stage off, measured as a runtime row because its output is bit-identical to FULL",
    },
    {
        "id": "CTRL-FULL-late",
        "kind": "runtime",
        "family": "reference",
        "argv": [],
        "env": {},
        "removes": [],
        "note": "same-session FULL control run at the end of the campaign; reads the serial-fps drift over the session",
    },
    {
        "id": "RT-NoDetectGraph",
        "kind": "runtime",
        "family": "runtime",
        "argv": [],
        "env": {},
        "config": {"use_whole_graph": False, "use_cuda_graph": False},
        "removes": ["WholeGraph", "HeadGraph"],
        "note": "eager detect path (no CUDA graph on detect)",
    },
    {
        "id": "RT-NoTrackerGraph",
        "kind": "runtime",
        "family": "runtime",
        "argv": [],
        "env": {},
        "config": {"use_tracker_graph": False},
        "removes": ["TrackerGraph"],
        "note": "tracker update not captured into a CUDA graph",
    },
    {
        "id": "RT-NoNMSGraph",
        "kind": "runtime",
        "family": "runtime",
        "argv": ["--no-main-nms-graphed"],
        "env": {},
        "removes": ["NMSGraph"],
        "note": "eager main NMS instead of the graphed nocopyback variant",
    },
    {
        "id": "RT-DoubleBuffer",
        "kind": "runtime",
        "family": "runtime",
        "argv": ["--double-buffer"],
        "env": {},
        "removes": [],
        "adds": ["DoubleBuffer"],
        "note": "detect(N+1) || tracker(N) overlap (the headline scheduling)",
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def variant_by_id() -> dict[str, dict[str, Any]]:
    return {v["id"]: v for v in VARIANTS}


def derived_config(variant: Mapping[str, Any], root: Path) -> Path | None:
    """Write preset + overrides for keys the CLI cannot set; return its path."""
    overrides = variant.get("config")
    if not overrides:
        return None
    base = yaml.safe_load(PRESET_PATH.read_text()) or {}
    base.update(overrides)
    path = root / "_configs" / f"{variant['id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# derived from configs/presets/{PRESET}.yaml with {json.dumps(overrides)}\n"
        + yaml.safe_dump(base, sort_keys=False)
    )
    return path


def build_cmd(variant: Mapping[str, Any], run_dir: Path, root: Path) -> list[str]:
    cfg = derived_config(variant, root)
    head = ["--config", str(cfg)] if cfg else ["--preset", PRESET]
    return [
        sys.executable,
        str(MOT17_ENTRY),
        *head,
        "--detector",
        "SDP",
        "--sequences",
        ",".join(SEQUENCES),
        "--output",
        str(run_dir),
        *variant["argv"],
    ]


def execute(variant: Mapping[str, Any], run_dir: Path, root: Path) -> dict[str, Any]:
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(variant, run_dir, root)
    env = dict(os.environ)
    env.update(variant.get("env", {}))
    started = _now()
    t0 = time.perf_counter()
    with (
        (run_dir.parent / f"{run_dir.name}.stdout.log").open("w") as out,
        (run_dir.parent / f"{run_dir.name}.stderr.log").open("w") as err,
    ):
        code = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, stdout=out, stderr=err
        ).returncode
    wall = time.perf_counter() - t0
    stdout_path = run_dir.parent / f"{run_dir.name}.stdout.log"
    # keep the raw stdout beside the MOT files as well
    if run_dir.is_dir():
        shutil.copy(stdout_path, run_dir / "stdout.log")
        shutil.copy(
            run_dir.parent / f"{run_dir.name}.stderr.log", run_dir / "stderr.log"
        )
    stdout = stdout_path.read_text(errors="replace")
    metrics = parse_overall_metrics(stdout)
    mot = (
        {name: mot_md5(p) for name, p in sorted(list_mot_files(run_dir).items())}
        if run_dir.is_dir()
        else {}
    )
    latency = {}
    lp = run_dir / "_latency_profile.json"
    if lp.is_file():
        payload = json.loads(lp.read_text())
        payload.pop("samples_ms", None)
        latency = payload
    record = {
        "schema": SCHEMA + "-run",
        "variant": variant["id"],
        "cmd": cmd,
        "env_overrides": variant.get("env", {}),
        "config_overrides": variant.get("config", {}),
        "started_at": started,
        "finished_at": _now(),
        "wall_seconds": round(wall, 3),
        "exit_code": code,
        "complete": code == 0 and set(SEQUENCES) <= set(mot) and bool(metrics),
        "metrics": (metrics or {}).get("numeric", {}),
        "throughput": parse_overall_throughput(stdout),
        "latency_overall": latency,
        "mot_md5": mot,
    }
    (run_dir.parent / f"{run_dir.name}.record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    by_id = variant_by_id()
    ids = args.variants or [v["id"] for v in VARIANTS if v["kind"] in set(args.kinds)]
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    manifest_path = root / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.is_file()
        else {
            "schema": SCHEMA + "-manifest",
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "preset": PRESET,
            "preset_sha256": __import__("hashlib")
            .sha256(PRESET_PATH.read_bytes())
            .hexdigest(),
            "sequences": SEQUENCES,
            "reference_run": REFERENCE_RUN,
            "reference_recipe": REFERENCE_RECIPE,
            "rules": RULES,
            "variants": {},
            "runs": [],
        }
    )
    for vid in ids:
        variant = by_id[vid]
        reps = (
            args.reps_full
            if vid == "FULL"
            else (args.reps_runtime if variant["kind"] == "runtime" else args.reps)
        )
        manifest["variants"][vid] = {k: v for k, v in variant.items()}
        existing = [
            r for r in manifest["runs"] if r["variant"] == vid and r["complete"]
        ]
        for i in range(len(existing), reps):
            run_dir = root / vid / f"r{i + 1:02d}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            print(f"[418] {vid} r{i + 1:02d}", flush=True)
            rec = execute(variant, run_dir, root)
            rec["run_dir"] = str(run_dir.relative_to(REPO_ROOT))
            manifest["runs"].append(rec)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            print(
                f"[418]   exit={rec['exit_code']} complete={rec['complete']} "
                f"IDF1={rec['metrics'].get('IDF1')} HOTA={rec['metrics'].get('HOTA')} "
                f"IDs={rec['metrics'].get('IDs')} fps={(rec['throughput'] or {}).get('fps')}",
                flush=True,
            )
    return 0


# --------------------------------------------------------------------------- report


def _precision(key: str) -> float:
    return PRINT_PRECISION.get(key, DEFAULT_PRECISION)


def _side(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    keys = sorted({k for r in runs for k in r["metrics"]})
    for k in keys:
        vals = [r["metrics"][k] for r in runs if k in r["metrics"]]
        metrics[k] = {
            "mean": round(statistics.fmean(vals), 4),
            "min": min(vals),
            "max": max(vals),
            "range": round(max(vals) - min(vals), 4),
            "n": len(vals),
        }
    fps = [r["throughput"]["fps"] for r in runs if r.get("throughput")]
    lat = [r["throughput"]["mean_latency_ms"] for r in runs if r.get("throughput")]
    p99 = [r["latency_overall"].get("p99_ms") for r in runs if r.get("latency_overall")]
    md5_sets = {json.dumps(r["mot_md5"], sort_keys=True) for r in runs}
    return {
        "n": len(runs),
        "k_distinct_outputs": len(md5_sets),
        "metrics": metrics,
        "fps": {
            "mean": round(statistics.fmean(fps), 2),
            "min": min(fps),
            "max": max(fps),
            "range": round(max(fps) - min(fps), 2),
        }
        if fps
        else None,
        "mean_latency_ms": {
            "mean": round(statistics.fmean(lat), 3),
            "min": min(lat),
            "max": max(lat),
        }
        if lat
        else None,
        "p99_ms": {
            "mean": round(statistics.fmean([p for p in p99 if p is not None]), 3)
        }
        if any(p is not None for p in p99)
        else None,
        "mot_md5": runs[0]["mot_md5"],
        "run_dirs": [r["run_dir"] for r in runs],
    }


def _delta(
    full: Mapping[str, Any], side: Mapping[str, Any], key: str
) -> dict[str, Any] | None:
    a = full["metrics"].get(key)
    b = side["metrics"].get(key)
    if not a or not b:
        return None
    d = b["mean"] - a["mean"]
    prec = _precision(key)
    rng = max(a["range"], b["range"])
    if abs(d) < prec / 2:
        reading = "="
    elif abs(d) <= rng:
        reading = "<=range"
    else:
        reading = ">range"
    return {
        "delta": round(d, 3),
        "reading": reading,
        "range_full": a["range"],
        "range_side": b["range"],
    }


def _material(deltas: Mapping[str, Any]) -> dict[str, str]:
    """Per-axis materiality sign: '+', '-' or ''."""
    m = RULES["materiality"]
    out: dict[str, str] = {}
    for ax in m["quality_axes"] + ["IDs"]:
        d = deltas.get(ax)
        if not d or d["reading"] != ">range":
            out[ax] = ""
            continue
        thr = m["ids_threshold"] if ax == "IDs" else m["quality_threshold"]
        if abs(d["delta"]) <= thr:
            out[ax] = ""
        else:
            # IDs: fewer is better, so sign is inverted to 'quality' direction
            sign = d["delta"] > 0
            if ax == "IDs":
                sign = not sign
            out[ax] = "+" if sign else "-"
    return out


def _single_role(mat: Mapping[str, str]) -> str:
    neg_primary = "-" in (mat["IDF1"], mat["HOTA"])
    pos_primary = "+" in (mat["IDF1"], mat["HOTA"])
    neg_any = "-" in mat.values()
    pos_any = "+" in mat.values()
    if neg_primary and "+" not in (mat["IDF1"], mat["HOTA"], mat["MOTA"]):
        return "essential"
    if pos_primary and "-" not in (mat["IDF1"], mat["HOTA"], mat["MOTA"]):
        return "conflict"
    if neg_any and pos_any:
        return "trade_off"
    if neg_any:
        return "essential"
    if pos_any:
        return "conflict"
    return "no_material_quality_role"


def cmd_report(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    runs_by: dict[str, list[dict[str, Any]]] = {}
    for r in manifest["runs"]:
        if r["complete"]:
            runs_by.setdefault(r["variant"], []).append(r)
    if "FULL" not in runs_by:
        raise SystemExit("no complete FULL run")
    sides = {vid: _side(rs) for vid, rs in runs_by.items()}
    full = sides["FULL"]

    # reference identity check against the #421 production formal run
    ref_path = REPO_ROOT / REFERENCE_RUN / "contract_run.json"
    ref = json.loads(ref_path.read_text()) if ref_path.is_file() else None
    identity = {
        "reference_run": REFERENCE_RUN,
        "reference_recipe": REFERENCE_RECIPE,
        "reference_commit": None,
        "reference_metrics": ref["metrics"]["numeric"] if ref else None,
        "mot_md5_equal": (ref["mot_md5"] == full["mot_md5"]) if ref else None,
        "full_k_distinct": full["k_distinct_outputs"],
        "full_n": full["n"],
    }
    rm_path = REPO_ROOT / REFERENCE_RUN / "run_manifest.json"
    if rm_path.is_file():
        identity["reference_commit"] = json.loads(rm_path.read_text()).get("commit")

    rows: list[dict[str, Any]] = []
    for vid, side in sides.items():
        if vid == "FULL":
            continue
        variant = manifest["variants"][vid]
        deltas = {k: _delta(full, side, k) for k in QUALITY_METRICS}
        deltas = {k: v for k, v in deltas.items() if v}
        identical = side["mot_md5"] == full["mot_md5"]
        metrics_identical = all(
            side["metrics"].get(k, {}).get("mean")
            == full["metrics"].get(k, {}).get("mean")
            for k in QUALITY_METRICS
        )
        diff_lines = None if identical else _diff_lines(root, full, side)
        row: dict[str, Any] = {
            "id": vid,
            "kind": variant["kind"],
            "family": variant["family"],
            "removes": variant["removes"],
            "note": variant["note"],
            "n": side["n"],
            "k_distinct_outputs": side["k_distinct_outputs"],
            "output_identical_to_full": identical,
            "metrics_identical_to_full": metrics_identical,
            "mot_diff_lines": diff_lines,
            "deltas": deltas,
            "fps": side["fps"],
            "fps_delta_pct": round(
                (side["fps"]["mean"] / full["fps"]["mean"] - 1) * 100, 2
            )
            if side["fps"] and full["fps"]
            else None,
            "mean_latency_ms": side["mean_latency_ms"],
            "p99_ms": side["p99_ms"],
        }
        if variant["kind"] in ("quality", "pair", "reference"):
            mat = _material(deltas)
            row["material"] = mat
            row["single_role"] = (
                _single_role(mat) if variant["kind"] == "quality" else None
            )
        if variant["kind"] == "runtime":
            fr = RULES["runtime"]["fps_threshold_pct"]
            # a late same-session FULL control (if present) guards against
            # serial-fps drift: the row must clear both FULL and the control
            controls = [full] + (
                [sides["CTRL-FULL-late"]]
                if "CTRL-FULL-late" in sides and vid != "CTRL-FULL-late"
                else []
            )
            no_overlap = bool(side["fps"]) and all(
                c["fps"]
                and (
                    side["fps"]["min"] > c["fps"]["max"]
                    or side["fps"]["max"] < c["fps"]["min"]
                )
                for c in controls
            )
            late = sides.get("CTRL-FULL-late")
            row["fps_delta_pct_vs_late_control"] = (
                round((side["fps"]["mean"] / late["fps"]["mean"] - 1) * 100, 2)
                if late and side["fps"] and late["fps"]
                else None
            )
            row["runtime_reading"] = (
                "runtime-only"
                if identical
                and row["fps_delta_pct"] is not None
                and abs(row["fps_delta_pct"]) > fr
                and no_overlap
                else (
                    "output-identical, fps unresolved"
                    if identical
                    else (
                        "runtime-only (metric-identical; md5 differs by sub-precision score rounding)"
                        if metrics_identical
                        and row["fps_delta_pct"] is not None
                        and abs(row["fps_delta_pct"]) > fr
                        and no_overlap
                        else "OUTPUT DIFFERS (not a pure runtime toggle)"
                    )
                )
            )
        rows.append(row)

    # pair interactions
    pairs: list[dict[str, Any]] = []
    for vid, side in sides.items():
        variant = manifest["variants"][vid]
        if variant["kind"] != "pair":
            continue
        a, b = variant["removes"]
        sa, sb = sides.get(f"FULL-{a}"), sides.get(f"FULL-{b}")
        if not sa or not sb:
            continue
        inter: dict[str, Any] = {}
        for k in ("IDF1", "HOTA", "AssA", "MOTA", "IDs"):
            mF, mA, mB, mAB = (s["metrics"][k]["mean"] for s in (full, sa, sb, side))
            inter[k] = {
                "d_A": round(mF - mA, 3),
                "d_B": round(mF - mB, 3),
                "d_AB": round(mF - mAB, 3),
                "d_A_without_B": round(mB - mAB, 3),
                "d_B_without_A": round(mA - mAB, 3),
                "I": round(mF - mA - mB + mAB, 3),
            }
        thr = RULES["pair_reading"]["threshold"]
        readings = {}
        for k in ("IDF1", "HOTA"):
            i = inter[k]["I"]
            readings[k] = (
                "substitute_backup"
                if i < -thr
                else ("complement_dependency" if i > thr else "additive")
            )
        pairs.append(
            {
                "id": vid,
                "A": a,
                "B": b,
                "note": variant["note"],
                "interaction": inter,
                "reading": readings,
            }
        )

    summary = {
        "schema": SCHEMA,
        "generated_at": _now(),
        "commit": manifest["commit"],
        "dirty": manifest["dirty"],
        "preset": manifest["preset"],
        "preset_sha256": manifest["preset_sha256"],
        "sequences": manifest["sequences"],
        "host": _host(),
        "rules": manifest["rules"],
        "identity": identity,
        "full": {k: v for k, v in full.items() if k != "run_dirs"}
        | {"run_dirs": full["run_dirs"]},
        "rows": rows,
        "pairs": pairs,
        "raw_root": str(root.relative_to(REPO_ROOT))
        if root.is_relative_to(REPO_ROOT)
        else str(root),
    }
    out = Path(args.json_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.markdown_output:
        Path(args.markdown_output).write_text(render_markdown(summary))
    print(
        f"[418] wrote {out}"
        + (f" and {args.markdown_output}" if args.markdown_output else "")
    )
    return 0


def _diff_lines(
    root: Path, full: Mapping[str, Any], side: Mapping[str, Any]
) -> dict[str, Any]:
    """Line-level MOT diff of the first run of each side (count + first differing pair)."""
    a = REPO_ROOT / full["run_dirs"][0]
    b = REPO_ROOT / side["run_dirs"][0]
    out: dict[str, Any] = {}
    for seq in SEQUENCES:
        la = (a / f"{seq}.txt").read_text().splitlines()
        lb = (b / f"{seq}.txt").read_text().splitlines()
        pairs = [(x, y) for x, y in zip(la, lb) if x != y]
        out[seq] = {
            "lines": len(la),
            "lines_other": len(lb),
            "differing": len(pairs) + abs(len(la) - len(lb)),
            "first": list(pairs[0]) if pairs else None,
        }
    return out


def _host() -> dict[str, Any]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    import platform
    import socket

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu": gpu,
    }


def _fmt_delta(d: Mapping[str, Any] | None, key: str) -> str:
    if not d:
        return "—"
    prec = 0 if key in PRINT_PRECISION else 1
    sign = "+" if d["delta"] >= 0 else ""
    mark = {"=": "=", "<=range": "≤range", ">range": ">range"}[d["reading"]]
    return f"{sign}{d['delta']:.{prec}f} ({mark})"


def render_markdown(s: Mapping[str, Any]) -> str:
    full = s["full"]
    fm = full["metrics"]
    lines = [
        f"<!-- generated by scripts/benchmarks/module_interactions/subtraction.py report at {s['generated_at']}; regenerate rather than edit -->",
        "",
        f"FULL = `{s['preset']}` SDP 7-seq serial at commit `{s['commit'][:12]}`"
        f"{' (dirty)' if s['dirty'] else ''}, host `{s['host']['hostname']}`, GPU `{s['host']['gpu']}`; "
        f"n={full['n']} runs, k={full['k_distinct_outputs']} distinct output(s).",
        "",
        f"FULL readings: HOTA {fm['HOTA']['mean']:.1f} · IDF1 {fm['IDF1']['mean']:.1f} · AssA {fm['AssA']['mean']:.1f} · "
        f"DetA {fm['DetA']['mean']:.1f} · MOTA {fm['MOTA']['mean']:.1f} · IDs {fm['IDs']['mean']:.0f} · "
        f"FP {fm['FP']['mean']:.0f} · FN {fm['FN']['mean']:.0f} · fps {full['fps']['mean']:.1f} "
        f"(range {full['fps']['range']:.2f}) · mean latency {full['mean_latency_ms']['mean']:.2f} ms.",
        "",
    ]
    idn = s["identity"]
    lines.append(
        f"Identity vs #421 production formal run `{idn['reference_run']}` (recipe `{idn['reference_recipe']}`, "
        f"commit `{(idn['reference_commit'] or '?')[:12]}`): MOT md5 equal = **{idn['mot_md5_equal']}**."
    )
    lines.append("")
    lines.append("### Single subtractions (variant − FULL)")
    lines.append("")
    lines.append(
        "| variant | removes | n/k | HOTA | IDF1 | AssA | DetA | MOTA | IDs | FP | FN | fps Δ% | reading |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for kind in ("quality", "reference"):
        for r in s["rows"]:
            if r["kind"] != kind:
                continue
            d = r["deltas"]
            role = r.get("single_role") or "reference"
            lines.append(
                f"| `{r['id']}` | {', '.join(r['removes'])} | {r['n']}/{r['k_distinct_outputs']} | "
                f"{_fmt_delta(d.get('HOTA'), 'HOTA')} | {_fmt_delta(d.get('IDF1'), 'IDF1')} | {_fmt_delta(d.get('AssA'), 'AssA')} | "
                f"{_fmt_delta(d.get('DetA'), 'DetA')} | {_fmt_delta(d.get('MOTA'), 'MOTA')} | {_fmt_delta(d.get('IDs'), 'IDs')} | "
                f"{_fmt_delta(d.get('FP'), 'FP')} | {_fmt_delta(d.get('FN'), 'FN')} | {r['fps_delta_pct']:+.1f} | {role} |"
            )
    lines.append("")
    lines.append("### Pair subtractions")
    lines.append("")
    lines.append(
        "| variant | n/k | HOTA | IDF1 | AssA | MOTA | IDs | I(IDF1) | I(HOTA) | reading |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    pair_by = {p["id"]: p for p in s["pairs"]}
    for r in s["rows"]:
        if r["kind"] != "pair":
            continue
        d = r["deltas"]
        p = pair_by.get(r["id"])
        i_idf1 = f"{p['interaction']['IDF1']['I']:+.1f}" if p else "—"
        i_hota = f"{p['interaction']['HOTA']['I']:+.1f}" if p else "—"
        reading = (
            f"IDF1 {p['reading']['IDF1']} · HOTA {p['reading']['HOTA']}" if p else "—"
        )
        lines.append(
            f"| `{r['id']}` | {r['n']}/{r['k_distinct_outputs']} | {_fmt_delta(d.get('HOTA'), 'HOTA')} | {_fmt_delta(d.get('IDF1'), 'IDF1')} | "
            f"{_fmt_delta(d.get('AssA'), 'AssA')} | {_fmt_delta(d.get('MOTA'), 'MOTA')} | {_fmt_delta(d.get('IDs'), 'IDs')} | {i_idf1} | {i_hota} | {reading} |"
        )
    lines.append("")
    lines.append("### Runtime toggles (serial profile unless noted)")
    lines.append("")
    lines.append(
        "| variant | n/k | output identical | fps | fps Δ% vs FULL | fps Δ% vs late control | mean latency ms | p99 ms | reading |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in s["rows"]:
        if r["kind"] != "runtime":
            continue
        lines.append(
            f"| `{r['id']}` | {r['n']}/{r['k_distinct_outputs']} | {r['output_identical_to_full']} | "
            f"{r['fps']['mean']:.1f} (range {r['fps']['range']:.2f}) | {r['fps_delta_pct']:+.1f} | "
            f"{'—' if r.get('fps_delta_pct_vs_late_control') is None else format(r['fps_delta_pct_vs_late_control'], '+.1f')} | "
            f"{r['mean_latency_ms']['mean']:.2f} | {r['p99_ms']['mean'] if r['p99_ms'] else '—'} | {r['runtime_reading']} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="execute the pre-declared variants into --root")
    r.add_argument("--root", required=True, help="artifact root (results/...)")
    r.add_argument(
        "--variants", nargs="*", help="subset of variant ids (default: all of --kinds)"
    )
    r.add_argument(
        "--kinds",
        nargs="*",
        default=["full", "quality", "reference", "pair", "runtime"],
    )
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--reps-full", type=int, default=3)
    r.add_argument("--reps-runtime", type=int, default=3)
    r.set_defaults(fn=cmd_run)
    q = sub.add_parser("report", help="reduce a run root into summary JSON/markdown")
    q.add_argument("--root", required=True)
    q.add_argument("--json-output", required=True)
    q.add_argument("--markdown-output")
    q.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
