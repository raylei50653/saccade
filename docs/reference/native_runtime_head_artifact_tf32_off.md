# Native runtime head artifact — TF32 off（#465 Phase B PR-1R／U1a redesign）

> 狀態：PR-1R artifact 端完成；**沒有量 parity**（那是 PR-2R，需要新的預宣告）。不改 eval harness、preset、threshold、weights、benchmark claim。
> 來由：PR-1 的形式（TRT FP32、TF32 允許、head-only）已被 PR-2 判為 `HEAD_PARITY_OUT_OF_TOLERANCE`（[結果](native_runtime_head_parity_result.md)），該結果永久保留，描述的是 PR-1 的 artifact。owner 決定的 redesign 路線記錄在 #465：一次只改一個設計軸。
> 工具：`scripts/model/export_headline_mamba_head.py --precision fp32-no-tf32`（`developer_build_debug`）。

---

## 1. 唯一的設計變更

| 軸 | PR-1（已否決） | PR-1R |
|:--|:--|:--|
| TensorRT TF32 builder flag | 預設（開） | **清除**（`config.clear_flag(TF32)`，build 後讀回 `tf32=false`） |
| ONNX | `6e919dad…` | **同一份**；工具重新匯出後若 sha256 不同就停止（fail-closed） |
| checkpoint／scan plugin／batch／head-only 範圍／FP16 | — | 全部與 PR-1 相同（FP16 關、batch 靜態 1、plugin `9f4d6dac…`） |

沒有改的：FP16、LibTorch、含 S2 的 artifact、其他 builder flag。FP16 被刻意排除：FP32 已經因 identity-sensitive 差異出界，降低精度沒有理由。

PR-1R 的產物用自己的 stem（`*_notf32`），PR-1 被否決的 engine 原地保留，只作 reference evidence，不參與任何門檻設定。

## 2. 這台機器上的紀錄

| 項目 | 值 |
|:--|:--|
| ONNX | `models/yolo/mamba_head_s_v14replica_t3_t1_fp32_notf32.onnx`，sha256 `6e919dad14af81083a25679225930a3473a8cdd6ebf9828ea07b414a9316b58b`（== PR-1） |
| engine | `models/yolo/mamba_head_s_v14replica_t3_t1_fp32_notf32.engine`，sha256 `3b7a98ab122e2ab79d94a6de3c5848af0c07d95f9445d0d6d800405284e5d3ce`（只識別這次 build，理由同 PR-1 artifact doc §3） |
| builder flag 讀回 | `fp16=false`、`tf32=false` |
| lineage manifest | `…_notf32.lineage.json`，`issue` = PR-1R，`tool.git_commit` = `7d31c590`，`git_dirty=false` |
| `--check` | PR-1R 與 PR-1 兩個 manifest 各自回 `OK`；PR-1 engine sha256 仍是 `b0502f84…`（PR-2 正式 run 記錄的那一份，未被改動） |
| 環境 | TensorRT 10.16.1.11；RTX 5070 Ti Laptop（SM 12.0） |

## 3. PR-2R 之前已經看過的東西

- 只有結構檢查：以 `TRTMambaHead` 載入 engine，餵全零的 p3/p4/p5，六個輸出的 shape 與 PR-1 相同、全部 finite。**沒有**在任何真實 frame 上比較 tensor，也沒有跑任何 MOT。
- PR-2 的全部結果（PR-1 形式的 L1／L2 數字）在 PR-2R 宣告時屬於「已看過的資料」，只能當 archived reference，不能拿來設定新門檻。

## 4. 交給 PR-2R

- PR-2R 必須是**新的**預宣告，在第一個量測 run 之前凍結；不是 PR-2 宣告的 amendment。可以沿用 PR-2 的 V1／V2／V3、L1／L2 結構與容差政策，但凍結輸入要換成本文的 engine／manifest。
- 若 TF32-off 仍失敗，依 owner 路線停止調整 TRT head-only 的設定，改做預宣告的 failure-localization study，再決定 S2 artifact 或 LibTorch。

## 5. 重現

```bash
.venv/bin/python tools/resctl.py run gpu0 -- \
    .venv/bin/python scripts/model/export_headline_mamba_head.py --precision fp32-no-tf32   # 已存在時需 --overwrite
.venv/bin/python tools/resctl.py run gpu0 -- \
    .venv/bin/python scripts/model/export_headline_mamba_head.py --precision fp32-no-tf32 --check
```
