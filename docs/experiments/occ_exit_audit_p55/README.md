<!-- doc-status: active -->
<!-- doc-promotion: navigation-only; not evidence -->
<!-- workspace: occ_exit_audit_p55 -->
<!-- module: semantic -->
<!-- pipeline-node: association-recovery / relink -->
<!-- disposition-owner: docs/research/threads/occ_exit_audit_20260709.md -->

# occ-exit audit (#55) — 實驗工作區（prototype）

> **導航入口。** 只給概況與連結;**verdict、數字、統計理由不在這裡**——fact-owner 是 `wp3_promotion_decision.md`。**work-disposition（做不做/parked/reopen）不在這裡**——owner 是上層 thread。**object-rung / terminal 不在這裡**——owner 是 registry。此檔不複寫任一方的狀態(no second truth);狀態一律去 owner 讀。

## 概況

在已落地的 Cheb-GR log-only probe substrate（frozen `mamba_whole_graph_m`,no-ReID）上,稽核「occ-exit 乾淨 crop 遲延再關聯」能否成為安全的序列條件化門控。三個 work package:範圍 → per-seq 適用性 → 升格決策。terminal 與數字見 WP3;工作線的做不做見 thread。

## 成員

| 檔 | 角色 |
|---|---|
| [`scope.md`](scope.md) | WP1 範圍與 intent |
| [`wp2_seq_conditioning.md`](wp2_seq_conditioning.md) | WP2 per-seq / scene-type 適用性 map |
| [`wp3_promotion_decision.md`](wp3_promotion_decision.md) | **WP3 升格決策(terminal fact-owner)** |

## 狀態去哪讀（owner，非本檔）

- **work-disposition（做不做/parked/reopen）:** [thread `occ_exit_audit_20260709`](../../research/threads/occ_exit_audit_20260709.md)
- **object-rung / terminal:** `docs/research/contracts/claim_state_registry.md`
- **no-go 登錄:** [`../../reference/no_go_registry.md`](../../reference/no_go_registry.md)
- memory:`project_occ_exit_audit_nogo`
