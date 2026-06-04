# Appearance / ReID 在 MOT17 的能力上限調查（2026-06-03）

> 結論：**appearance 對 MOT17 ID 一致性無 headroom**——根因是 MOT17 行人身份在
> embedding 空間**本質難分**，非機制 / 模型 / 解析度可解。baseline（motion + GMC，
> IDF1 73.3%）已是此資料集合理操作點。所有探索 code 保留、default off，待**更強的
> MOT-域 ReID 特徵**或**換場景**再用既有工具重測。
>
> 模組短結論見 [reid README](../../modules/reid/README.md) / [semantic README](../../modules/semantic/README.md) 的 GO/NO-GO；本文為細節家。

## 動機

目標是「丟失後重連成功率」（track 丟失→重現時接回同一 ID，而非冒新 ID）= ID 一致性。
motion + GMC 在短 gap 撐得住，長 gap 失效；想用 appearance 在長 gap 尋回身份。

## 工具（保留）

- `scripts/eval/reid_id_benchmark.py` — 直接量 embedding 的身份區分力（rank-1 / mAP /
  intra-inter cosine gap / d' / AUC），並依**時間 gap** 與**框高**分層。**未來任何新特徵
  的 cheap gate**：gap 31+ rank-1 撐不住就別接 tracker。
- `scripts/eval/reconnect_rate.py` — GT-based 重連成功率（丟失→重現接回同 ID 比例），依 gap 分層。
- relink C++ 全鏈（見下）+ `scripts/train/reid_domain_probe.py`（MOT-域 head 訓練 probe）。

## 關鍵診斷：embedding 區分力 benchmark（MOT17 GT crops）

5 個現成模型，**全部弱**（好的 ReID 應 d'>2、AUC>95%）：

| 模型 | mAP | 平均 d' | gap 1–10 | gap 31–60 | gap 61–120 | gap 121+ |
|---|---|---|---|---|---|---|
| siglip2_reid (Market) | 19–34% | ~0.76 | 91% | 60% | 33% | 13% |
| siglip2 (base) | 19–36% | ~0.89 | 91% | 61% | 37% | 14% |
| dinov2 | 17–28% | ~0.86 | 89% | 52% | 31% | 10% |
| **transreid** | 20–37% | ~1.0 | 91% | 63% | 38% | 13% |
| osnet | 20–32% | ~1.05 | 91% | 54% | 36% | 10% |

- **intra−inter cosine gap 僅 ~0.03**（同人 0.95 / 不同人 0.92，分佈幾乎重疊）。
- 同一 siglip2_reid 在 **Market mAP 81% → MOT17 mAP 19–34%**：Market 證件照不轉移到
  MOT17 小/遮擋/同裝行人。
- **長 gap 一致崩**：所有模型 gap 61+ 都掉到 10–38%——正是 motion 失效、需要 appearance 的區段。

## 機制嘗試（全 NO-GO）

| 機制 | 結果 |
|---|---|
| online tracker 外觀關聯成本 | mamba_optimal A/B：IDF1 ±0、AssA +0.2、IDs +6（FP −180）；淨值雜訊內 |
| offline Cheb-GR tracklet merge（路徑2） | safe 操作點 IDs↓但 **AssA 0.0pp**；放寬即過度合併傷 IDF1 |
| **birth-time lost-bank relink**（含 Cheb-GR 自適應門檻 + 速度搜捕圈） | 無 λ 能讓復活降 IDs：λ 高→0 復活、λ 中→白做、λ 低→誤接 IDs↑ |
| MOT-域 SupCon 訓 projection head（凍結 backbone） | macro rank1 60.8→60.2（無改善）；re-project 加不了 backbone 沒編碼的資訊 |
| 超解析度 / Lanczos（輸入品質） | **非解析度受限**（見下）；不值得 |

### 為什麼 birth-relink 失敗（即使 Cheb-GR）

長 gap 下 embedding rank-1 僅 13–33% → 復活多半接到**附近的 look-alike**（真正重現的人
常因遮擋期間移動被空間 gate 濾掉）。精度優先門檻只會把復活數壓到 0；放寬就誤接（merge
兩人 = 比碎片更傷 ID 一致性）。

### 為什麼 SR 救不了（決定性）

依**框高分層** rank-1：

| query 框高 | rank-1 |
|---|---|
| 0–50px | 55.8% |
| 50–100px | 63.4% |
| 100–200px | 63.3% |
| 200+px | 57.1% |

**大清晰框(200+px)沒比小框好**（57% vs 56%）→ 瓶頸不是像素模糊，是外觀歧義。
Lanczos（更銳利插值）≈ Bilinear，進一步證實插值品質無關。

## 根因（單一）

MOT17 身份在 appearance 空間**本質難分**：200+px 清晰框也只 57% rank-1；同裝擁擠行人 +
長 gap 姿態/遮擋變化讓 intra-inter gap 僅 ~0.03。**換模型 / 加機制 / 超解析度都撞同一上限。**
唯一理論未試＝full backbone fine-tune，但證據（清晰大框 57%、head 訓練無效）強烈指向它
也撞同一外觀歧義天花板，期望值低。

## 過程中的工程發現（有價值，已修）

1. **graph 路徑不餵 embedding**：`GraphedTrackerUpdate.copy_inputs` 無 embeddings 參數；
   `use_tracker_graph=true`（mamba preset）下 `d_embeddings` 恆 null → online 外觀關聯**從未
   生效**（之前「online reid bit-exact」的真因，不只 has_clean gate）。relink 啟用時改走
   `update_into`（會餵 embedding），並自動關 tracker graph。
2. **relink 同幀重複 ID 兩坑**：① UnionFind 遞移性 → 改 greedy-by-cost + component **frame
   set 嚴格不相交**；② `has_clean_embedding` 在 track lost 後被 bank prune 清掉 → archive 改
   用 **embedding norm>0** gate（`d_features_` lost 後不清零）。

## relink C++ 基建（保留，default off）

`src/tracking/tracker_gpu.cu`：GPU ring-buffer lost bank（archive on expiry / age / Cheb-GR
兩階段 relink kernel / spawn 復活），`set_relink_params` + `--relink-*` flags。啟用：
`--reid-mode tracker --reid-model siglip2_reid --appearance-bank --reid-interval 1 --relink-enabled`。

## 重啟條件

有**MOT/crowd 域訓練（對小框+遮擋魯棒）的 ReID 特徵**時：先過 `reid_id_benchmark.py`，
**gap 31+ rank-1 明顯 > 現有 ~37%/13% 才接 tracker**（relink 基建現成）。
