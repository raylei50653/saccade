# Motion Module (運動關聯模組)

## 📐 模組職責
負責物體運動一致性校驗、軌跡速度與加速度 EMA 平滑，並在 ReID 外觀失效時提供運動補償匹配 Fallback。

## 🟢 目前現況
* 提供 EMA 平滑係數 `vel_alpha=0.3` 與 `acc_alpha=0.15`。
* ⚠️ **運動關聯去重已被設為 NO-GO (2026-05-17)**：實驗表明 89% 候選匹配被 age gate 阻斷，且純運動增益與隨機噪聲相近，因此代碼保留但預設關閉。
* **OA-SORT OAO duration-ramp** 已在 `mamba_whole_graph` 中復活為 tracker cost 的時間軸遮擋懲罰（`oao_tau=0.50`, `oao_ramp_frames=25`）；這不是 2026-05 的 plain OAO cost。

## 🔗 I/O & Dataflow

| | |
|---|---|
| **Pipeline stage** | `track` association 內（見 [pipeline_flow.md](../../reference/pipeline_flow.md)） |
| **輸入** | track 狀態（速度 / 加速度 EMA）+ detections |
| **輸出** | motion consistency cost（z-score）/ motion-only fallback 匹配 |
| **上游 → 下游** | `tracker association：appearance 失效 → motion EMA 預測 → IoU fallback` |

## ⚖️ GO / NO-GO 決策

> 完整脈絡見 [TODO_history.md](../../TODO_history.md)。default：EMA only（`vel_alpha=0.3`、`acc_alpha=0.15`），motion-only relink off。

| 日期 | 項目 | 結論 |
|------|------|------|
| 2026-05-17 | Motion-based relinking + better association cost | ❌ NO-GO（89% 候選被 age gate 攔截，增益≈雜訊）；code 保留 default=off |
| 2026-05-20 / 2026-06-17 | OA-SORT OAO | plain OAO 曾為 NO-GO；duration-ramp 版本已復活並進 current preset（見 [no_go_registry #7](../../reference/no_go_registry.md)） |

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
