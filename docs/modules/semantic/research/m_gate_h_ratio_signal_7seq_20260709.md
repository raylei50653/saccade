# Gate 1 — height-ratio signal (m production scale gate)

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**One gate only.** Next gate = separate note / study.  
**Ledger row:** `m.h_ratio.scale` → [signal_analysis_ledger](../../../research/eval/signal_analysis_ledger.md)  
**Study master:** [`out/signal_study/m_gate_h_ratio_7seq_20260709T122056Z/signal_gate_h_ratio.json`](../../../../out/signal_study/m_gate_h_ratio_7seq_20260709T122056Z/signal_gate_h_ratio.json)  
**Pairs:** 7-seq B1 substrate (`m_b1_smoke_*`, bridge/interp off)  
**Layer:** L0 support gate + L1 1D term readout on the same physical quantity  
**Preset knobs:** `relink_bridge_h_lo=0.6` · `relink_bridge_h_hi=1.7`  
**Proxy:** `h_ratio = h_lost_raw / h_cand_raw`（非 live `ema_h`）

---

## 1. Gate 定義

```text
reject  iff  h_ratio ∉ [0.6, 1.7]
score   =  |log(h_ratio)|     # 連續量；愈小愈像真對
AUC     on  −|log h|          # higher = more pos-like
```

問題只問兩件事：

1. **訊號：** `|log h|` 能不能分 pos / neg？  
2. **L0 覆蓋：** production 帶寬砍多少 FP、傷多少 GT？

---

## 2. 訊號（連續量）

| | pos (n=340) | neg (n=21449) |
|:--|--:|--:|
| `|log h|` median | **0.099** | **0.579** |
| `|log h|` p90 | 0.336 | 1.594 |
| `|log h|` max | 0.910 | （長尾） |

| Pool | AUC (−\|log h\|) |
|:--|--:|
| **full** | **0.863** |
| **hard** (`bridge_dist≤1`) | **0.784** |

**讀法：** scale 是**強全池 ranker**（砍極端尺度跳變）；難池仍 >0.75，比「隨機」好，但不是閉合 identity 通道。  
真對集中在 ratio≈1（median |log|≈0.10 → ratio ~0.91–1.10）。

---

## 3. Production 帶寬 L0 覆蓋

`reject = h_ratio ∉ [0.6, 1.7]`

| Pool | GT_hurt | FP_removed | kept pos recall |
|:--|--:|--:|--:|
| **full** | **11 / 340 (3.24%)** | **11608 / 21449 (54.1%)** | 96.8% |
| **hard** | 4 / 212 (1.89%) | 517 / 1583 (32.7%) | — |

- 閘後仍留 **9841 FP（45.9% of FP）** → 尺度過了的假對，要交給 **px / score / assignment**。  
- 傷到的 11 個 GT：gap median **116**（kept GT gap median **42**）→ 誤傷偏**長 gap**。

### Gap

| gap | GT_hurt | FP_rm% | AUC |
|:--|--:|--:|--:|
| 1–10 | 1/53 (1.9%) | 47.9% | 0.87 |
| 11–30 | **0/76 (0%)** | 50.7% | 0.85 |
| 31–60 | 2/63 (3.2%) | 51.7% | 0.86 |
| 61–150 | 4/100 (4.0%) | 52.4% | 0.85 |
| 151–300 | 4/48 (8.3%) | 57.4% | 0.81 |

gap↑ → pos 的 `|log h|` 中位略升、AUC 略降；**長 gap 尺度漂移**是 GT_hurt 主來源。

### Seq

| seq | hurt | FP_rm% | AUC |
|:--|--:|--:|--:|
| 02 | 0/72 | 48% | 0.84 |
| 04 | 0/12 | **9.5%** | 0.82 |
| 05 | 1/42 | **66%** | **0.93** |
| 09 | 1/14 | 54% | 0.83 |
| 10 | **8/157** | 53% | 0.86 |
| 11 | 1/20 | 68% | 0.87 |
| 13 | 0/23 | 49% | 0.83 |

- **GT 稅幾乎全在 10**（8/11）。  
- **04** FP 幾乎砍不到（尺度本來就接近）→ 此 gate 在 04 **覆蓋面薄**。  
- **05** 訊號最強（AUC 0.93）且 FP_rm 高。

---

## 4. 帶寬掃（同一訊號，不同 gate 寬）

| band | GT_hurt | FP_rm |
|:--|--:|--:|
| [0.5, 2.0] | 1.2% | 42% |
| [0.55, 1.8] | 2.1% | 49% |
| **[0.6, 1.7] m** | **3.2%** | **54%** |
| [0.65, 1.55] | 4.4% | 61% |
| **[0.75, 1.33] s** | **12.7%** | **72%** |
| [0.8, 1.25] | 17.7% | 78% |

m 相對 s：用 ~9pp 的 GT_hurt 換回大量真對 coverage（s 傷 43 vs m 傷 11）。  
**ε=0 1D ceiling：** 要 GT_hurt=0 → thr `|log|≈0.91` ≈ ratio 超出 **~[0.40, 2.49]**，只砍 **31% FP**（比 production 54% 保守一截）。

---

## 5. 這一個 gate 的結論（只講 h-ratio）

1. **有訊號：** full AUC 0.86 / hard 0.78；pos/neg `|log h|` 中位差約 6×。  
2. **m production 帶寬合理：** 3% GT 稅換 54% FP 減負；再緊到 s 形會明顯吃 GT。  
3. **不是 identity 解：** 過閘後仍剩 ~46% FP；難池只砍 33% FP。  
4. **條件化：** 長 gap + seq-10 承擔幾乎全部 GT_hurt；04 上此 gate 幾乎無負池可砍。  
5. **ε=0 可解釋規則：** 比 production 更鬆的 1D ceiling 才 ε=0；production 帶寬本身是 **risky 但低稅** 的 support gate，不是 safe_reject 庫候選敘事。

**下一步（下一個 note 再開）：** Gate 2 = `score_m_bridge` / `relink_bridge_px=0.4` 單訊號同規格分析。  
本檔不混 px、不改 preset。

---

## Reproduce

```bash
# numbers: out/signal_study/m_gate_h_ratio_7seq_*/signal_gate_h_ratio.json
# pairs from existing 7-seq B1 substrate
```
