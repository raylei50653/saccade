# ReID — 模組 TODO

> **WIP register only.** Association / handover policy 文檔家 = [semantic](../semantic/README.md)。

## Sole active

⏸️ **暫緩** — 無 active（卡在特徵能力上限）

## Parked

- ReID identity recovery — blocked on MOT/crowd-domain feature quality  
  Unlock: pass `scripts/eval/reid_id_benchmark.py` gates, then retest relink (`--relink-enabled` default off)  
  → [appearance_ceiling_mot17](../../research/reid/appearance_ceiling_mot17.md)

## Done / closed

See [README](README.md) · [no_go_registry](../../reference/no_go_registry.md)（sync online ReID #57 等）.
