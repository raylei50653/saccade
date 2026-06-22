# `frozen_v2` 累積消融與延遲 benchmark（2026-06-21）

> **權威來源**：本文件是 [PROJECT_SHOWCASE](../../PROJECT_SHOWCASE.md) 附錄「單機可重現的累積消融」
> 與 [ADR 018](../../decisions/018-project-main-line-direction.md) headline 的可追溯佐證。
> 全部數字出自 2026-06-21 同一批 run（`mamba_whole_graph` preset、MOT17 train / SDP 七序列、
> RTX 5070 Ti Laptop GPU 12 GB）。品質指標 run-to-run bit-exact（`reid_mode: off` + GMC graph path），
> FPS / latency 為 timing 量測，有 ±1 FPS 系統噪聲。
>
> 原始 eval 輸出（逐幀 MOT result）為可由下列指令重生的中間產物，未入庫；本文件保留摘要。

---

## 1. 現行 headline（`frozen_v2`）

| 指標 | 7-seq (GT-w) | ex-04 (6-seq GT-w) |
|---|---:|---:|
| IDF1 | **78.2** | 67.8 |
| HOTA | **70.2** | 58.4 |
| AssA | **69.7** | 55.8 |
| MOTA | **78.4** | — |
| DetA | 70.9 | — |
| IDs | 413 | — |
| Rcll / Prcn | 81.0 / 97.2 | — |

`78.2` 與 `ex-04 67.8` 的 ~10 分落差全來自 MOT17-04 一條（占 ~42% GT 權重）；雙軌並列是誠信要求，04 是 showcase 非 headline。

> ⚠️ **leakage：這些是 in-sample（training-set）絕對值。** detection head + teacher + cache 全在這 7 序列上訓練 → 高估泛化。本文件的**累積消融 delta（§2-§3）leakage 輕**（tracker 無在評測序列上訓練的權重），是更可主張的部分；leakage-free 的 detector 泛化數字見 [held-out plan](../../modules/detection/research/holdout_generalization_plan.md)。

### Per-sequence（`frozen_v2`）

| Seq | HOTA | AssA | IDF1 |
|---|---:|---:|---:|
| 02 | 49.1 | 47.0 | 58.8 |
| 04 | 84.2 | 83.6 | 91.2 |
| 05 | 61.1 | 61.7 | 73.8 |
| 09 | 57.6 | 49.7 | 66.8 |
| 10 | 55.8 | 49.8 | 63.4 |
| 11 | 71.2 | 67.3 | 79.7 |
| 13 | 61.3 | 60.5 | 72.4 |
| **7-seq (GT-w)** | **70.2** | **69.7** | **78.2** |
| **ex-04 (GT-w)** | **58.4** | **55.8** | **67.8** |

---

## 2. 累積消融（bare → full headline）

從全關 bare tracker 開始，依 `+GMC → +bridge relink → +occ-gate + OAO`(等) 逐步開啟：

| 配置 | IDF1 | MOTA | HOTA | AssA | IDs | Rcll | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| bare（GMC / bridge / occ-gate / OAO 全關） | 71.4 | 75.3 | 65.0 | 62.4 | 888 | 79.3 | 3581 |
| + GMC | 75.9 | 77.7 | 67.9 | 66.4 | 445 | 81.3 | 3616 |
| + GMC + bridge relink | 76.7 | 78.2 | 68.5 | 67.2 | 406 | 81.5 | 3284 |
| **full headline `frozen_v2`**（+ occ-gate + OAO 等全開） | **78.2** | **78.4** | **70.2** | **69.7** | **413** | **81.0** | **2589** |

bare → full：IDF1 **+6.8** · HOTA **+5.2** · AssA **+7.3** · IDs 888→413（**−53%**） · FP 3581→2589。

---

## 3. 模組非獨立性（為何 delta 不可加）

各模組**單獨**加到不同基礎上的邊際效果（同 preset、7-seq、IDF1）：

| 加在 | + GMC | + bridge | + occ-gate | + OAO ramp |
|---|---:|---:|---:|---:|
| **bare**（71.4） | +4.5 → 75.9 | +1.6 → 73.0 | **−1.2 → 70.2** | **−1.3 → 70.1** |
| **GMC**（75.9） | — | +0.8 → 76.7 | **−1.7 → 74.2** | **−1.5 → 74.4** |
| **GMC + bridge**（76.7） | — | — | 併同 OAO 共 +1.5 → **78.2** | — |

**關鍵觀察**：same-height occ-gate 與 OAO duration-ramp 不只在 bare 上為負，**單獨疊在 GMC 上仍為負**
（74.2 / 74.4 < 75.9）。它們只有在 **GMC + bridge relink 同時在場**、幀間（前後）一致性建立後才轉為增益
（76.7 → 78.2 的 +1.5 由 occ-gate + OAO 共同貢獻）。bridge 的增益也由單獨 +1.6 在 GMC 之上縮為 +0.8（次可加）。

> 機制：occ-gate 比較同高足點幾何、OAO 累積連續重疊幀數，兩者都假設框在相鄰 frame 位置一致；
> 相機運動未經 GMC 補償時，足點幾何與重疊持續時間訊號被污染，gate 誤觸發。
> **因此這張表只能讀成累積因果鏈（每列在前一列之上的邊際效果），不能與各模組自有對照實驗的 delta 互相加減。**

---

## 4. 延遲：兩個操作點（throughput vs single-frame latency）

| 模式 | Eval FPS | mean ms/frame | p99 ms | 用途 |
|---|---:|---:|---:|---|
| `--double-buffer`（throughput-optimized） | 269–270 | 7.42 | ~12.5 | 部署吞吐量 |
| single-frame（latency-optimized） | 144 | 6.34 | ~7.8 | 低尾延遲 |

⚠️ **FPS 是 throughput、mean_ms 是單幀 latency，兩者不可互推。** double-buffer 把 `detect(N) ‖ tracker(N−1)`
排到相鄰 frame → throughput 近 2×（143.8 → 270.4 FPS, **+88%**），但單幀 latency 反而較高（6.34 → 7.42 ms，重疊代價）。

**double-buffer 對品質 bit-exact**：full preset 開 / 關 double-buffer，raw MOT result md5 完全相同
（IDF1 / HOTA / MOTA / IDs 一字不差，皆 78.2 / 70.2 / 78.4 / 413）；它只改排程不改任何追蹤決策。
故累積表 `+GMC+bridge`(76.7) → `full`(78.2) 的 +1.5 全部來自 occ-gate + OAO，double-buffer 不貢獻品質。

> `docs/reference/benchmarks/latency_log.md` 另記 40–71 FPS，是不同量測脈絡（stream 數 / warm-up /
> 是否含 decode+track / GPU 不同）；報告須各自註明協定，不可與 269 / 7.42 並列而不解釋。

---

## 5. 重現

```bash
# headline（full frozen_v2）
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
  --double-buffer --output out/frozen_v2
.venv/bin/python scripts/eval/_perseq_extract.py out/frozen_v2

# 累積消融（bare → full）：bare 用全關旗標，再逐項改回 preset 預設
#   --no-gmc --no-relink-bridge-enabled --no-occ-state-enabled --oao-tau 0
# double-buffer 品質 bit-exact 對照：同 preset 開 / 關 --double-buffer，比對 raw MOT result md5
```

2026-06-21 的逐項 run 由 `scratch/run_ab.sh` / `scratch/run_ab2.sh` 驅動（個人 driver，非 release artifact）。
