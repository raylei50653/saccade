#!/usr/bin/env python3
"""
Sweep: 局部軌跡密度自適應門控 (Density-Gating) 超參數搜索
=============================================================
掃描 --semantic-exp-density-k 與 --semantic-exp-density-eta 的最佳組合，
在 MOT17-05-SDP 上比較 IDF1 / HOTA / AssA / IDs。

Usage:
    cd <repo root>
    uv run python scripts/eval/sweep_density_gating.py

輸出:
    results/sweep_density_gating.csv   — 完整結果表
    results/sweep_density_gating.md    — Markdown 摘要（最佳 Top-5）
"""
# status: diagnostic

from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

# ─────────────────────────────────────────────
# 掃描格點
# ─────────────────────────────────────────────
K_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]  # 密度搜尋半徑（× 行人高度）
ETA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.30]  # 指數衰減強度

# 基礎評估指令（與之前驗證時相同的組態）
BASE_FLAGS = (
    "--detector SDP "
    "--sequences MOT17-05-SDP "
    "--reid-mode semantic "
    "--semantic-mahalanobis-threshold 9.4877 "
    "--semantic-exp-density-gating "  # 永遠啟用（我們在掃 k/eta）
)

EVAL_SCRIPT = "scripts/eval/mot17.py"
CALC_SCRIPT = "scripts/eval/calculate_mota.py"
RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "sweep_density_gating.csv"
MD_PATH = RESULTS_DIR / "sweep_density_gating.md"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _run(cmd: str, cwd: Path = PROJECT_ROOT) -> tuple[str, str]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(cwd))
    return r.stdout, r.stderr


def _parse_overall_metrics(stdout: str) -> dict[str, str]:
    """
    Parse the '=== OVERALL METRICS ===' block that mot17.py prints when
    evaluating a single sequence. Example output:

        === OVERALL METRICS ===
          IDF1: 52.8%
          HOTA: 39.9%
          AssA: 41.7%
          DetA: 38.3%
          IDs: 66
          MOTA: 46.2%
          ...
    Also parse any key: value% lines outside that block (fallback).
    """
    metrics: dict[str, str] = {}
    in_block = False
    for line in stdout.splitlines():
        if "=== OVERALL METRICS ==" in line:
            in_block = True
            continue
        if in_block:
            # end of block on empty line or next === section
            if line.strip() == "" or line.strip().startswith("==="):
                in_block = False
                continue
            # parse "  KEY: VALUE"
            m = re.match(r"\s+(\w+):\s*([\d.]+)%?", line)
            if m:
                metrics[m.group(1)] = m.group(2)
    # fallback: grep for KEY: XX.X% anywhere
    if not metrics:
        for m in re.finditer(r"(HOTA|AssA|DetA|IDF1|MOTA|IDs):\s*([\d.]+)%?", stdout):
            metrics[m.group(1)] = m.group(2)
    return metrics


def run_one(k: float, eta: float) -> dict[str, str | float]:
    label = f"k={k:.2f}_eta={eta:.2f}"
    print(f"\n[{time.strftime('%H:%M:%S')}] ▶ {label}", flush=True)

    cmd = (
        f"uv run python {EVAL_SCRIPT} "
        f"{BASE_FLAGS}"
        f"--semantic-exp-density-k {k} "
        f"--semantic-exp-density-eta {eta}"
    )
    stdout, stderr = _run(cmd)
    combined = stdout + stderr

    metrics = _parse_overall_metrics(combined)

    row: dict[str, str | float] = {"label": label, "k": k, "eta": eta}
    row.update(metrics)

    idf1_str = metrics.get("IDF1", "?")
    hota_str = metrics.get("HOTA", "?")
    assa_str = metrics.get("AssA", "?")
    ids_str = metrics.get("IDs", "?")
    print(
        f"   IDF1={idf1_str}%  HOTA={hota_str}%  AssA={assa_str}%  IDs={ids_str}",
        flush=True,
    )
    return row


def write_csv(results: list[dict]) -> None:
    if not results:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ CSV → {CSV_PATH}", flush=True)


def write_md(results: list[dict]) -> None:
    if not results:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _f(row: dict, key: str, pct: bool = False) -> str:
        v = row.get(key, "—")
        if v == "—" or v == "?":
            return "—"
        suffix = "%" if pct else ""
        return f"{v}{suffix}"

    # Sort by IDF1 desc (fallback HOTA)
    def sort_key(r: dict):
        try:
            return float(r.get("IDF1", 0))
        except (ValueError, TypeError):
            return 0.0

    ranked = sorted(results, key=sort_key, reverse=True)
    top5 = ranked[:5]

    lines = [
        "# Density-Gating 超參數 Sweep 結果",
        "",
        "> 序列: MOT17-05-SDP | 啟用 `--semantic-exp-density-gating` | 固定 `--semantic-mahalanobis-threshold 9.4877`",
        "",
        "## Top-5 最佳組合（按 IDF1 排序）",
        "",
        "| Rank | k | eta | IDF1 | HOTA | AssA | IDs | MOTA |",
        "|------|---|-----|------|------|------|-----|------|",
    ]
    for i, row in enumerate(top5, 1):
        lines.append(
            f"| {i} | {row.get('k', '?')} | {row.get('eta', '?')} "
            f"| {_f(row, 'IDF1', True)} "
            f"| {_f(row, 'HOTA', True)} "
            f"| {_f(row, 'AssA', True)} "
            f"| {_f(row, 'IDs')} "
            f"| {_f(row, 'MOTA', True)} |"
        )

    lines += [
        "",
        "## 完整結果",
        "",
        "| label | k | eta | IDF1 | HOTA | AssA | IDs | MOTA |",
        "|-------|---|-----|------|------|------|-----|------|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['label']} | {row['k']} | {row['eta']} "
            f"| {_f(row, 'IDF1', True)} "
            f"| {_f(row, 'HOTA', True)} "
            f"| {_f(row, 'AssA', True)} "
            f"| {_f(row, 'IDs')} "
            f"| {_f(row, 'MOTA', True)} |"
        )

    lines += ["", "---", f"_生成時間: {time.strftime('%Y-%m-%d %H:%M:%S')}_", ""]
    MD_PATH.write_text("\n".join(lines))
    print(f"✅ MD  → {MD_PATH}", flush=True)


def main() -> None:
    grid = list(product(K_VALUES, ETA_VALUES))
    total = len(grid)
    print(f"🔍 Density-Gating Sweep: {total} 組合 (k={K_VALUES}, eta={ETA_VALUES})")
    print(f"   基礎指令: {BASE_FLAGS.strip()}\n")

    results: list[dict] = []

    for idx, (k, eta) in enumerate(grid, 1):
        print(f"[{idx}/{total}]", end=" ", flush=True)
        try:
            row = run_one(k, eta)
            results.append(row)
            # 每次寫入，避免中途失敗丟失資料
            write_csv(results)
            write_md(results)
        except KeyboardInterrupt:
            print("\n⚠️  使用者中斷，儲存目前結果...")
            write_csv(results)
            write_md(results)
            sys.exit(0)
        except Exception as e:
            print(f"   ❌ 錯誤: {e}", flush=True)
            results.append(
                {"label": f"k={k}_eta={eta}", "k": k, "eta": eta, "error": str(e)}
            )

    print(f"\n{'=' * 60}")
    print(f"✅ Sweep 完成！共 {len(results)} 組結果")
    write_csv(results)
    write_md(results)


if __name__ == "__main__":
    main()
