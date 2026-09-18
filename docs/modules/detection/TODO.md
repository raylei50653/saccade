# Detection — 模組 TODO

> **WIP register only.** 任務與排序在 GitHub issue（milestone [`Now`](https://github.com/raylei50653/saccade/milestone/1)）。開發路由：[DEVELOPMENT.md](../../../DEVELOPMENT.md#agent-action-cards)。

## Sole active

⏸️ 無 active — VGT-Mamba 自 2026-07-06 無 run／commit，sole-active 已釋放；resume-or-retire 由 [#440](https://github.com/raylei50653/saccade/issues/440)（`Later`）決定。

- Canonical: [README 設計入口](README.md) · training protocols under module docs

## Parked

- VGT-Mamba temporal head + contingent follow-ups（Hybrid Mamba-ViT、annotation reinforcement）→ [#440](https://github.com/raylei50653/saccade/issues/440)
- Ultralytics runtime decoupling — analysis in [ADR 023](../../decisions/023-ultralytics-runtime-decouple.md) / #391; follow-up 4（AGPL-free training loss/assigner）未開 issue，需要時另開

## Done / closed

- ADR 023 follow-ups 1–3 — #393 decode、#395 TRT construction skip、`ultralytics` → `yolo` extra（#414）
- mamba_head CUDA graph eval bug → [research/mamba-cuda-graph-bug.md](research/mamba-cuda-graph-bug.md)
- ST-Mamba → superseded by VGT
- Option F / whole-graph lineage → [option-f-mamba-head.md](option-f-mamba-head.md) · [TODO_history](../../TODO_history.md)
