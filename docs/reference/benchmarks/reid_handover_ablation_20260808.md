# ReID 與 handover 消融（2026-08-08）

> **本文回答兩個會被反覆問起的問題**:為什麼 headline 把 ReID 關掉,以及 online/streaming
> handover 值不值得實作。
>
> 全部數字出自 2026-08-08 同一批 run:`mamba_whole_graph_m` preset、MOT17 **train** / SDP 七序列、
> RTX 5070 Ti Laptop GPU 12 GB、main `f1dfc616`。除 §3 標註處外,一律 `--double-buffer --no-gpu-decode`。
>
> ⚠️ **leakage:這些是 in-sample(training-set)絕對值**,與
> [frozen_v2_ablation](frozen_v2_ablation.md) 同一個限制 —— 絕對值高估泛化,
> **本文可主張的是 delta 不是絕對值**。
>
> 原始逐幀 MOT 輸出為可由 §5 指令重生的中間產物,未入庫。

---

## 0. 先決條件:`--no-gpu-decode` 下 eval 完全確定性

三個 arm × 三次重複,**每個品質指標的 stdev 都是 `±0.00`**。

這修正了先前「N≥6 才穩」的經驗法則 —— 那是 **GPU decode 開著**時的觀察。
關掉之後品質指標 run-to-run bit-exact,**A/B 用 N=1 即可**,不必跑多輪取平均。
FPS 仍是 timing 量測,有系統噪聲。

---

## 1. ReID 在當前 tracker 上買到 0.0 IDF1

| base | IDF1 | MOTA | HOTA | DetA | AssA | IDs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`reid_mode: off`**(出貨組態) | **80.4** | 81.6 | 74.4 | 75.4 | 73.5 | 344 | **245.1** |
| `--reid-mode tracker`(mnv4 mainline) | **80.4** | 81.6 | 74.4 | 75.4 | 73.5 | 345 | **160.9** |

**每一個品質指標小數點都相同;差異只有 1 個 ID switch 與 7 個 FN(噪聲級)。**
ReID 買到 **0.0 IDF1**,付掉 **84 FPS(−34%)**。

這比先前的歷史對照(reid-off 79.5 / reid-tracker 79.7)更乾淨 —— 當時還有 0.2 的差,現在是零。
tracker 自身的 relink + bridge 已經把 appearance 這條路能接回的身份接完了。

⇒ [`configs/presets/mamba_whole_graph_m.yaml`](../../../configs/presets/mamba_whole_graph_m.yaml) 的
`reid_mode: "off"` 是正確設定,與 [mot17_default_config.md](../mot17_default_config.md) §ReID 一致。
**跑 A/B 時不要順手把它打開** —— 那會換掉 base,而且換到一個不出貨的組態。

---

## 2. Offline handover:reid-off base 上 **+0.4 IDF1**,FPS 零成本

`--module-lifecycle configs/modules/cheb_gr_offline_mnv4.yaml`,base 為 §1 的 `reid_mode: off`。

| arm | IDF1 | MOTA | HOTA | IDs | handovers | FPS |
|---|---:|---:|---:|---:|---:|---:|
| base | 80.4 | 81.6 | 74.4 | 344 | 0 | 245.1 |
| + offline handover | **80.8** | 81.5 | 74.5 | **338** | 54 | 254.6 |
| **Δ** | **+0.4** | −0.1 | +0.1 | **−6** | 54 | ~0 |

FPS 不受影響:offline handover 是 **output-layer post-process**,跑在 tracker 輸出完成之後,
不在 per-frame critical path 內(245 vs 254 的差是 timing 噪聲)。代價是序列末多一個 pass。

GPU decode 開著時的同一組對照為 80.3 → 80.6(+0.3),方向一致 ⇒ **+0.4 不是雜訊**。

### 這個 lever 已經衰減

歷史(2026-07-05)同一條路是 **79.5 → 80.2(+0.7,90 handovers)**。
現在是 **80.4 → 80.8(+0.4,54 handovers)**。

base 自己漲了 +0.9,把 handover 原本填的 headroom 吃掉大半。
**絕對最佳值是進步的(80.2 → 80.8),但槓桿本身變薄了。**

---

## 3. Live / streaming handover:−4.5 IDF1,且跑不了出貨組態

`--module-lifecycle configs/modules/online_ho_mnv4_{bank,requery}.yaml`。
本節 **無 `--double-buffer`**(該 emit path 缺 `track_uid`),N=3。

| arm | IDF1 | MOTA | HOTA | IDs | handovers accepted |
|---|---:|---:|---:|---:|---:|
| base | 80.4 | 81.6 | 74.4 | 345 | 0 |
| bank(requery off) | 75.9 | 80.2 | 70.7 | 519 | 159 |
| bank + borderline requery | 75.7 | 80.0 | 70.6 | 544 | 168 |

**159 次 handover = −4.5 IDF1 / +174 IDs**,平均每次製造約 1.1 個 ID switch ⇒ 這批 handover 幾乎全錯。

### 兩個結構性問題

1. **它只能跑在 reid-ON base 上。** 兩個 `online_ho_mnv4_*.yaml` 的 header 都強制
   `--reid-mode tracker`,因為 live C++ relinker 靠 `feed_frame_embeddings` 吃 tracker ReID 的 embedding。
   結合 §1 ⇒ 真實生產命題是:**先付 −34% FPS 開一個買 0.0 IDF1 的 ReID stack,再在上面賠 −4.5 IDF1。**
2. **主缺陷不是 borderline requery。** requery 只值 −0.2(75.9 → 75.7),
   在 −4.5 的量級前可忽略;調 band/top 不會改變結論。

### 已作廢的舊數字

2026-07-05 曾記錄 bank 73.0 / requery 72.3,並判 requery 為 **−0.7**。
那組跑在 2026-07-06 才修掉的八個 bug 之上(其中兩個是系統性的:cache key 量化導致
約 10%/box 靜默 miss、以及全零列直接進決策核心造成 k-reciprocal 病理),**不予採用**。
修復確實值約 +3 IDF1,但**對 base 的落差只從 6.7 縮到 4.5,沒有關閉**;
requery 的判決則從 −0.7 收斂到 −0.2,且 IDs 軸符號翻轉(當時減 IDs,現在增 IDs)。

---

## 4. 這份文件建立與未建立什麼

**建立**:上述四個組態在 2026-08-08 main `f1dfc616` 上的可重現 delta;
`reid_mode: off` 的正當性;offline handover 目前值 +0.4 IDF1 / −6 IDs 且 FPS 零成本。

**未建立**:

- **這不是 NO-GO 登記。** 本文只提供證據;online/streaming handover 尚未在
  [no_go_registry](../no_go_registry.md) 登記任何終點,是否登記由 owner 決定。
- **未量測** live handover 在 **reid-off base** 上的表現。現行 live 路在該組態下拿不到 embedding,
  要量必須先實作 crop-ring on-demand 抽取 —— 亦即**必須先付開發成本才知道結果**。
- 絕對值受 in-sample leakage 影響(見開頭)。
- §3 的 `bank` arm 有一次 run 因 DALI `cudaErrorStreamCaptureUnsupported`
  (DALI 於 CUDA graph capture 期間動作)中途失敗,故 n=2;因 §0 的確定性成立,不影響數值。

---

## 5. 重生指令

```bash
# 共同前綴（LD_LIBRARY_PATH 需 prepend torch lib）
export LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
BASE="scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP \
      --double-buffer --no-gpu-decode"

# §1 reid-off base（出貨組態；preset 自帶 reid_mode: off）
.venv/bin/python $BASE --output out/ab_reid_off

# §1 reid-on 對照
.venv/bin/python $BASE --module-reid configs/modules/reid_mnv4_mainline.yaml \
    --reid-mode tracker --output out/ab_reid_on

# §2 offline handover（+0.4）
.venv/bin/python $BASE \
    --module-lifecycle configs/modules/cheb_gr_offline_mnv4.yaml \
    --output out/ab_offline_ho

# §3 live handover（注意：無 --double-buffer，且強制 reid-on）
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP \
    --no-gpu-decode --module-reid configs/modules/reid_mnv4_mainline.yaml \
    --reid-mode tracker \
    --module-lifecycle configs/modules/online_ho_mnv4_bank.yaml \
    --output out/ab_live_bank
```

handover 實際觸發數在 stdout:offline 為 `🧬 Cheb-GR Offline Handover: ...(N handovers, ...)`,
live 為 `🔗 Online handover <seq>: N handovers accepted`。

> ⚠️ C++ **沒有導出 borderline requery 的計數器**。要判斷 requery 有沒有真的觸發,
> 用兩臂 MOT 輸出做 byte-diff —— 兩臂只差 `cheb_gr_online_requery_band` 一個值,
> 輸出 byte-identical 即證明它一個決策都沒改。本輪三次重複皆 5–7/7 序列不同 ⇒ 機制確實觸發。

---

## 相關

- [bridge_gate_stability_20260808.md](bridge_gate_stability_20260808.md) — 同一批 run 的 bridge gate 穩定邊界部分
- [frozen_v2_ablation.md](frozen_v2_ablation.md) — 現行 headline 累積消融(`mamba_whole_graph`,s-variant)
- [mot17_default_config.md](../mot17_default_config.md) — baseline preset 與 CLI fallback
- [no_go_registry.md](../no_go_registry.md) — NO-GO / parked 方向索引
