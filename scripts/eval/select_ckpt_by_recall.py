#!/usr/bin/env python3
"""Post-training checkpoint selection by MOT17 tracking recall.

Train with `--best-by none --save-every 1` so a stage keeps every epoch
candidate, then run this to pick the deploy checkpoint by the metric we
actually care about (Rcll) instead of training loss.

Recall over fine-tune epochs is treated as unimodal (rise -> plateau ->
mild overfit decline), so we *ternary-search* the epoch axis rather than
evaluating every candidate: ~7 evals instead of ~15 for a 15-epoch stage.
Every probe is memoised and the full probed curve is printed, so if the
curve looks non-unimodal you can see it and widen.

The winner is copied to <run-dir>/best_recall.ckpt; the train-loss
best.ckpt is left untouched for comparison.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "bin" / "python")
RCLL_RE = re.compile(r"\bRcll\b\s*[:=]?\s*([0-9.]+)")
EXTRA_RES = {
    "IDF1": re.compile(r"\bIDF1\b\s*[:=]?\s*([0-9.]+)"),
    "MOTA": re.compile(r"\bMOTA\b\s*[:=]?\s*([0-9.]+)"),
    "IDs": re.compile(r"\bIDs\b\s*[:=]?\s*([0-9.]+)"),
}


def list_candidates(run_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in run_dir.glob("epoch_*.ckpt"):
        m = re.search(r"epoch_(\d+)\.ckpt", p.name)
        if m:
            out[int(m.group(1))] = p
    if not out:
        sys.exit(f"no epoch_*.ckpt candidates in {run_dir}")
    return out


def eval_recall(ckpt: Path, args, scratch: Path) -> tuple[float, dict[str, float]]:
    out_dir = scratch / ckpt.stem
    cmd = [
        PY,
        str(ROOT / "scripts" / "eval" / "mot17.py"),
        "--preset",
        args.preset,
        "--detector",
        args.detector,
        "--split",
        args.split,
        "--mamba-ckpt",
        str(ckpt),
        "--mamba-teacher-ckpt",
        args.teacher_ckpt,
        "--output",
        str(out_dir),
    ]
    if args.sequences:
        cmd += ["--sequences", args.sequences]
    if args.extra_eval_args:
        cmd += args.extra_eval_args.split()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-2000:])
        sys.exit(f"eval failed for {ckpt}")
    text = proc.stdout
    m = RCLL_RE.search(text)
    if not m:
        sys.stderr.write(text[-2000:])
        sys.exit(f"could not parse Rcll for {ckpt}")
    rcll = float(m.group(1))
    extra = {}
    for k, rgx in EXTRA_RES.items():
        em = rgx.search(text)
        if em:
            extra[k] = float(em.group(1))
    return rcll, extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--preset", default="mamba_whole_graph")
    ap.add_argument("--detector", default="SDP")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sequences", default="")
    ap.add_argument(
        "--extra-eval-args",
        default="",
        help="Extra flags appended verbatim to each mot17.py eval (e.g. "
        "'--no-temporal --no-compile' for a PyTorch-backbone PP22 deploy ckpt).",
    )
    ap.add_argument(
        "--brute",
        action="store_true",
        help="evaluate every candidate instead of ternary search",
    )
    args = ap.parse_args()

    cands = list_candidates(args.run_dir)
    epochs = sorted(cands)
    cache: dict[int, tuple[float, dict[str, float]]] = {}
    scratch = Path(tempfile.mkdtemp(prefix="ckpt_sel_"))

    def probe(ep: int) -> float:
        # snap to nearest available epoch
        ep = min(epochs, key=lambda e: abs(e - ep))
        if ep not in cache:
            r, extra = eval_recall(cands[ep], args, scratch)
            cache[ep] = (r, extra)
            tail = "  ".join(f"{k} {v}" for k, v in extra.items())
            print(f"  epoch {ep:>3}  Rcll {r:6.2f}   {tail}", flush=True)
        return cache[ep][0]

    print(
        f"[select] {len(epochs)} candidates in {args.run_dir.name}: "
        f"epochs {epochs[0]}..{epochs[-1]}",
        flush=True,
    )

    if args.brute:
        for ep in epochs:
            probe(ep)
    else:
        lo, hi = epochs[0], epochs[-1]
        while hi - lo > 2:
            m1 = lo + (hi - lo) // 3
            m2 = hi - (hi - lo) // 3
            if probe(m1) < probe(m2):
                lo = m1 + 1
            else:
                hi = m2 - 1
        for ep in range(lo, hi + 1):
            probe(ep)

    best_ep = max(cache, key=lambda e: cache[e][0])
    best_r, best_extra = cache[best_ep]
    dst = args.run_dir / "best_recall.ckpt"
    shutil.copy2(cands[best_ep], dst)
    tail = "  ".join(f"{k} {v}" for k, v in best_extra.items())
    print(f"\n[select] WINNER epoch {best_ep}  Rcll {best_r:.2f}   {tail}")
    print(f"[select] evaluated {len(cache)}/{len(epochs)} candidates")
    print(f"[select] copied -> {dst}")
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
