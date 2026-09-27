# Native runtime head artifact（#465 Phase B PR-1／U1a）

> 狀態：PR-1 artifact 端完成；**不改** eval harness、preset、threshold、weights、benchmark claim，也沒有量 parity（那是 PR-2）。
> 邊界依據：[native_runtime_shipping_boundary.md](native_runtime_shipping_boundary.md) §5 B1／B5、§6 PR-1。本文沿用該文的編號（S1–S11、B1–B5、PR-1…PR-12），不重述它的論證。
> 工具：`scripts/model/export_headline_mamba_head.py`（`developer_build_debug`）。

---

## 1. PR-1 做的三個選擇

| 軸 | 選擇 | 理由 |
|:--|:--|:--|
| head 形式 | **TensorRT engine＋`libsaccade_scan_plugin.so`**（ONNX 帶 `saccade::SelectiveScan` custom op） | shipping 執行檔可以完全不依賴 libtorch（audit §6）；PR-8 直接沿用既有 native `TRTEngine`；eval harness 已有 `--mamba-head-engine` → `TRTMambaHead`，會把這個 artifact 放進 `_whole_graph_fn` 的同一個位置，PR-2 不需要動 harness 就能對 oracle 比 |
| artifact 範圍 | **只有 head**：`p3/p4/p5 → cls_p3..p5, reg_p3..p5` | S2（640 stretch resize、anchor decode、sigmoid／class max、top-k、座標縮放）留給 U3b 以 native kernel 實作，`conf_thr`／`max_det` 只有 B2 resolved config 這一個來源，不會被凍進 engine |
| 精度與形狀 | **FP32**（TensorRT 預設，TF32 允許；FP16 關）；batch 靜態 1 | oracle 的 head 是 FP32 eager（cuDNN TF32 預設開、matmul TF32 關）；FP16 是另一個偏差來源，不在 PR-1 引入。shipping 與 oracle 的 whole-detect 都是 batch 1 |

head 由**與 oracle 相同的建構路徑**產生：`build_mamba_gated_detector(..., trt_backbone_engine=<preset>, use_whole_graph=True)`，匯出的是 `_whole_graph_fn` 實際呼叫的那個 `MambaDetectionHead`（單幀、temporal blocks bypass、`return_embeddings=False`）。

## 2. Lineage 與 fail-closed 輸入

工具在匯出前檢查，任何一項不成立就停：

1. preset `configs/presets/mamba_whole_graph.yaml` 的 `use_whole_graph` 為 true；
2. preset 的 `mamba_ckpt` 路徑 == [training lineage inventory](../research/training/training_lineage_inventory.md) 節點 `s.t3t1_phase_b` 的路徑；
3. 該 checkpoint 的 sha256 == inventory 記錄的 sha256；
4. head 載入乾淨：用 head **自己的** `load_state_dict(sd, strict=False)` 在 head 的副本上重載一次（probe），回傳的 `missing_keys`、`unexpected_keys` 都必須為空。reduction migration、PixelShuffle 相容、合法的 A_log shared↔per-channel 轉換都以真正 loader 的語義為準；其餘 shape mismatch 會被 loader 丟掉，因此以 missing key 的形式出現並被拒絕。工具不另寫一套相容規則。

第 2 項的名稱綁定與第 4 項的 probe 規則由 `tests/unit/test_headline_head_export_binding.py` 在 CI 檢查（probe 用小型 CPU head 做 mutation：非法 A_log shape、其他 shape mismatch、多餘 key 都必須失敗，合法 A_log broadcast 必須通過）；hash 需要 gitignored 的 `runs/`，只能在有 checkpoint 的機器上跑。

B5 相關的觀察：這個 checkpoint 的 `mamba_args` **沒有**記錄 `base_yolo_sha256`／`teacher_checkpoint_sha256`，所以 oracle 的 SHA gate（`MambaGatedDetector.__init__`）對它是 no-op。lineage 因此改由 inventory 的 ckpt sha256 釘住，這正是 B5「lineage 判定移到匯出時」的形狀。

輸出（預設在 gitignored 的 `models/yolo/`）：

| 檔案 | 身分 |
|:--|:--|
| `mamba_head_s_v14replica_t3_t1_fp32.onnx` | **可攜身分**。同一組輸入重跑匯出時 bit-identical |
| `mamba_head_s_v14replica_t3_t1_fp32.engine` | 這台機器的 build（綁 SM＋TensorRT 版本；tactic timing 讓 bytes 不可重現），sha256 只識別這次 build |
| `mamba_head_s_v14replica_t3_t1_fp32.lineage.json` | `saccade.head_artifact_lineage/v1`：preset hash、inventory hash 與比對結果、ckpt sha256／epoch／selection、完整 `mamba_args`、head 載入描述、artifact 範圍、ONNX／engine／plugin hash、TensorRT 版本、GPU／SM、工具 commit、環境版本、backbone engine hash（只供 B5 驗證用，不背書其 provenance） |

## 3. 這台機器上的紀錄

| 項目 | 值 |
|:--|:--|
| checkpoint | `runs/mamba_gt_v14replica_t3_t1/best.ckpt`，sha256 `c161c88e50b894d8b51cc614c46c3700370373decf05a15825bdf00ccf0e0876`（== inventory `s.t3t1_phase_b`），epoch 15 |
| head 載入 | loader probe：missing 0、unexpected 0；`upsample_loaded=true`；in_channels `(128, 256, 512)`；temporal blocks 存在但 bypass |
| ONNX | sha256 `6e919dad14af81083a25679225930a3473a8cdd6ebf9828ea07b414a9316b58b`，opset 17 |
| scan plugin | `build/libsaccade_scan_plugin.so`，sha256 `9f4d6dac28b95af822efc0a99b6e641310e6152da55a49c78caca4e6ec1163fc` |
| 環境 | TensorRT 10.16.1.11；RTX 5070 Ti Laptop（SM 12.0）；torch 2.11.0+cu130；onnx 1.21.0 |
| 可重建 | `--check`：從相同輸入重新匯出到暫存目錄，ONNX sha256 與紀錄相同（bit-identical），且 on-disk ONNX／engine 未被改動 → `OK` |
| engine | 同一台機器、同一份 ONNX 建兩次，engine sha256 不同（`b324da3d…`、`e8de5d15…`），證實 engine bytes 不能當身分；manifest 由 clean tree（`git_dirty=false`）產生 |
| backbone engine | `models/yolo/yolo26s_backbone_640_best.engine` sha256 == inventory `s.backbone_engine` |

## 4. PR-2 之前已經看過的東西（不是 parity 證據）

為了確認 TRT 形式在結構上可行，PR-1 做了一次**冒煙**比對，後續 PR-2 預宣告容差時應把它當成「已經看過的資料」，不是證據：

- 3 張 MOT17 frame（04 #1、04 #300、13 #100），同一份 TRT backbone 特徵餵給 eager head 與這個 FP32 engine；
- 輸出**不是 bit-exact**：logits 最大差 ~1.4e-2、reg 最大差 ~7.8e-3，逐 anchor 最大 sigmoid score 差 ~1e-3，score > 0.25 的 anchor 數三張都相同；
- 同一張 frame 上 `torch.compile`（head＋block）對 eager 也不是 bit-exact（cls 最大差 ~4.9e-3）。

這只說明「engine 算的是同一個函數，且沒有大錯」，不說明 detection、tracking 或 MOT 輸出的差異是否可接受，也不能引用為精度或效能數字。

## 5. 已知限制（交給後續 PR）

- ONNX parser 對 plugin node 報 `Attribute {B,L,D,N,has_D,a_per_channel} not found`，`build_mamba_head_trt.py` 的 `get_plugin_creator` 查詢也回傳 None（creator 仍列在 registry、engine 仍可建、輸出與 eager 接近）。這表示 plugin 從輸入 shape 推導這些值；PR-2 的 tensor parity 是它的實際檢查點。
- backbone engine 的 provenance 仍是 inventory 記錄的狀態（engine bytes 不可歸屬、sibling ONNX 指向不同 teacher）。PR-1 只記錄它的 hash；它屬於 PR-8／B5 的 backbone 端，不在本 PR 範圍。
- engine 是 per-machine 產物；跨機器分發與 bundling 是 Phase C／owner 決策。

## 6. 重現

```bash
.venv/bin/python tools/resctl.py run gpu0 -- \
    .venv/bin/python scripts/model/export_headline_mamba_head.py            # 產生（已存在時需 --overwrite）
.venv/bin/python tools/resctl.py run gpu0 -- \
    .venv/bin/python scripts/model/export_headline_mamba_head.py --check    # 重新匯出並比對紀錄
```

PR-2 用這個 artifact 對 oracle 比對時，harness 端只需要 `--mamba-head-engine models/yolo/mamba_head_s_v14replica_t3_t1_fp32.engine`（會讓 `_whole_graph_fn` 走 `TRTMambaHead.infer_graph`）。
