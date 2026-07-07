#!/usr/bin/env python
"""Audit whether two MOT17 detector-suffix runs contain distinct evidence.

Some eval paths use ``--detector`` only to select MOT17 sequence suffixes while
the actual detections still come from the same learned detector. This tool
normalizes sequence suffixes and provenance paths, then compares MOT result
files, Cheb-GR handover logs, labeled decision CSVs, and registries/summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

DETECTOR_SUFFIX_RE = re.compile(r"-(SDP|DPM|FRCNN)\b")
RESULTS_PATH_RE = re.compile(r"results/[A-Za-z0-9_./-]+")


def _normalize_text(text: str, *, path_a: Path, path_b: Path) -> str:
    out = DETECTOR_SUFFIX_RE.sub("-DET", text)
    out = out.replace(str(path_a), "RUN")
    out = out.replace(str(path_b), "RUN")
    out = out.replace(path_a.name, "RUN")
    out = out.replace(path_b.name, "RUN")
    out = RESULTS_PATH_RE.sub("RESULTS_PATH", out)
    return out


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _mot_key(path: Path) -> str:
    return DETECTOR_SUFFIX_RE.sub("-DET", path.name)


def _read_mot_normalized(path: Path) -> str:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        rows.append(",".join(parts[:10]))
    return "\n".join(rows) + ("\n" if rows else "")


def _read_csv_normalized(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys()) if rows else []
    for row in rows:
        if "seq" in row:
            row["seq"] = DETECTOR_SUFFIX_RE.sub("-DET", row["seq"])
    rows.sort(key=lambda r: tuple(r.get(k, "") for k in fields))
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(row.get(k, "") for k in fields))
    return "\n".join(lines) + "\n"


def _compare_blob(name: str, a: str, b: str) -> tuple[bool, str]:
    same = a == b
    return same, f"{name}: {'same' if same else 'DIFF'} sha={_sha(a)}/{_sha(b)}"


def _compare_mot_dirs(dir_a: Path, dir_b: Path) -> list[str]:
    files_a = {_mot_key(p): p for p in dir_a.glob("MOT17-*.txt")}
    files_b = {_mot_key(p): p for p in dir_b.glob("MOT17-*.txt")}
    lines: list[str] = []
    keys = sorted(set(files_a) | set(files_b))
    missing = [k for k in keys if k not in files_a or k not in files_b]
    if missing:
        lines.append(f"mot_files: DIFF missing={missing}")
        return lines
    all_same = True
    for key in keys:
        a = _read_mot_normalized(files_a[key])
        b = _read_mot_normalized(files_b[key])
        same, line = _compare_blob(f"mot:{key}", a, b)
        all_same = all_same and same
        if not same:
            lines.append(line)
    lines.insert(0, f"mot_files: {'same' if all_same else 'DIFF'} count={len(keys)}")
    return lines


def run(dir_a: Path, dir_b: Path) -> list[str]:
    lines = [f"run_a: {dir_a}", f"run_b: {dir_b}"]
    lines.extend(_compare_mot_dirs(dir_a, dir_b))
    csv_names = [
        "_cheb_gr_offline_handover.csv",
        "_cheb_gr_offline_handover_labeled.csv",
    ]
    for name in csv_names:
        a = _read_csv_normalized(dir_a / name)
        b = _read_csv_normalized(dir_b / name)
        same, line = _compare_blob(name, a, b)
        if not (dir_a / name).exists() or not (dir_b / name).exists():
            line += " (missing one or both files)"
        lines.append(line)
    for name in ("parameter_registry.md", "parameter_summary.json"):
        path_a = dir_a / name
        path_b = dir_b / name
        a_text = _normalize_text(
            path_a.read_text() if path_a.exists() else "",
            path_a=dir_a,
            path_b=dir_b,
        )
        b_text = _normalize_text(
            path_b.read_text() if path_b.exists() else "",
            path_a=dir_a,
            path_b=dir_b,
        )
        same, line = _compare_blob(name, a_text, b_text)
        if not path_a.exists() or not path_b.exists():
            line += " (missing one or both files)"
        lines.append(line)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    args = ap.parse_args()
    for line in run(args.run_a, args.run_b):
        print(line)


if __name__ == "__main__":
    main()
