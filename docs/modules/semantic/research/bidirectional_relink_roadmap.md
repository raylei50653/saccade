# 雙向時空收斂幾何重連 — 長期路線圖

> Bidirectional Space-Time Convergent Kinematic Relink Roadmap
> 狀態：Phase 0 已落地（default off），Phase 1–4 規劃中。最後更新 2026-06-04。

## 背景 (Context)

起點是「semantic relink 的 ID 飛來飛去」：丟失軌跡的空間約束原本是**靜態圈**（`spatial_gate` 中心位移比例），對久丟、移動中的目標不是太鬆（亂跳）就是太緊（漏接）。

逐步沿兩份設計規範補上物理門控：

- `kalman_gating_guide.md`：協方差膨脹 + 馬氏距離 + 卡方門控。
- `bidirectional_spacetime_tracking_spec.md`：各向異性雪茄雲、雙向時空夾擊、延遲認領、GPU two-pass。

**驅動案例 ID4**（custom_seq）：中央往右走的人在 f510 偵測消失後，被左邊另一個真實的人（id10）倒退 ≈1500px / 11 幀（≈27 m/s @60fps）「接走」，內插再畫出一條穿越畫面中央的鬼軌跡。這暴露了**單向擴散 + 首現即認領**的根本缺陷——對稱協方差雲容許物理上不可能的倒退連接，而新框在首次出現、還沒有速度時就被迫決策。

設計原則（延續專案慣例）：**每階段獨立可開關、預設關閉、先 CPU 驗證邏輯再上 GPU、每階段跑 MOT17 GT ablation（HOTA/AssA/IDF1/IDs）才決定 default**。

## 現況 — Phase 0（DONE）

relinker CPU 路徑（C++ `SemanticRelinkerCpp` @ `src/tracking/tracker_gpu_python.cpp` + Python `PythonSemanticRelinker` @ `src/saccade/perception/eval/relink.py` 鏡像）已具備，全部 default off、CLI flags 已接（`scripts/eval/config/semantic.py` → `evaluator.py`）：

| 機制 | Flag | 說明 |
|---|---|---|
| 正向卡方門控 + penalty | `--semantic-kalman-gate` / `--semantic-kalman-chi2`(9.4877) / `--semantic-kalman-penalty-weight` | 丟失軌跡 Kalman 外推+膨脹，馬氏距離卡方硬門 + `1−exp(−D²/2)` 軟懲罰 |
| cosθ 方向閘 | `--semantic-kalman-dir-min-cos` / `--semantic-kalman-dir-min-speed` | 斬斷與速度反向（後方）的連接 |
| 物理各向異性「雪茄雲」 | `--semantic-kalman-person-height-m`(1.65) / `--semantic-kalman-accel-long`(2.0) / `--semantic-kalman-accel-lat`(1.0) / `--semantic-kalman-fps` | 身高 px/m anchoring + 白噪聲加速度，慣性沿航向拉長、橫向收緊 |
| 物理速度上限 | `--semantic-kalman-max-speed-mps`(0=off, 建議 8) | snapshot-independent，隱含平均速度超人類極限→判新 id。**修好 ID4** |

新增 stats：`reject_kalman` / `reject_direction` / `reject_speed`。

**ID4 驗證**：開速度上限後 `reject_speed` 觸發，id4 正確止於 f510、左邊的人保留自身 id10（不被併吞），mean_center_norm 0.178→0.132。

## 關鍵已知資產（對接基礎）

- `kf_gpu::predict / get_R / get_Q / invert4x4`（`include/tracking/kalman_gpu.cuh`）為 `__host__ __device__`，CPU/GPU 共用，零物理重寫。Phase 0 的 `predict_phys`（各向異性 Q）即在 relinker 重用。
- **三層關聯，門控最終需全覆蓋**：
  1. **GPU tracker-core relink bank**：`src/tracking/tracker_gpu.cu` — `archive_expiring_tracks_kernel`(@854)、relink-claim kernel + `atomicCAS(relink_valid)`(@898–966, 已含 N_valid guard)、`parallel_auction_shmem_kernel`(@641)。**Phase 4 擴充此處，非從零。**
  2. **CPU 語意 relinker**（Phase 0 已加閘門）。
  3. **CPU `TrackletLifecycleMergerCpp`**（第二層，目前**無**物理閘門）。
- tracker 已有 **tentative/confirm 機制**：`confirm_streak` / `hit_streak` / `required_confirm_streak` / `get_tentative_candidates` / `TrackStateSnapshot.state`（含速度）。延遲認領可直接騎上。
- **結果流**：per-sequence `results_lines` 收集 → `filter_low_quality_tracklets` → `interpolate_tracklets` → 收尾寫檔（`evaluator.py` ~4255–4280）。⇒ **retroactive id remap 可行**：延遲認領成功時對 `results_lines` 套別名，再交既有內插補死區。

## 路線圖

### Phase 1 — 速度取得 + 延遲認領骨架（CPU）

新框**不再首現即認領**，先緩衝 K(≈confirm_streak=3) 幀讓 Kalman 收斂出穩定 `v_spawn`。

- Spawn 速度來源：優先用 tracker 對「當前解析 track id」的 Kalman 快照（擴充 `get_motion_snapshots_for_track_ids` 取用對象）；fallback 用 relinker `last_boxes_` 歷史估速。
- 延遲認領佇列：would-be-new 的 raw_id 記入 pending；滿 K 幀或 tracker confirmed 才觸發認領博弈。
- **對接點**：relinker 暴露 `deferred_alias: {new_id→claimed_old_id}`；evaluator 在 `interpolate_tracklets` 前對 `results_lines` 套用 remap（新工具 `apply_deferred_alias`，緊鄰 `filter_low_quality_tracklets`）。
- Flags：`--semantic-delayed-claim`、`--semantic-claim-warmup-frames`(3)。

### Phase 2 — 反向傳播 + 雙向聯合門控（CPU）

spec §2/§3。認領須**正向∩逆向**雙過。

- 逆向外推：`x_backward = x_spawn − v_spawn·Δt`、`P_backward = P_spawn + Σ predict_phys(逆向)`（Δt=lostage，重用 `predict_phys` 以 −v 建雲）。
- 雙向馬氏：`D_forward²`（已有）+ `D_backward²`（新軌雲倒推 vs lost 凍結位置）。
- 斷邊：任一 `D²>chi2` 或 `cosθ<0` → 代價 ∞。
- 聯合代價：`Cost = Cost_fwd + Cost_bwd + Cost_shape`，接 relinker best-joint + `reciprocal_margin`（模糊→偏新 id）。
- 死區修補：認領成功沿用 `interpolate_tracklets` 補 [t_lost, t_spawn]。
- Flags：`--semantic-bidirectional`、`--semantic-w-shape`；stats：`reject_backward` / `accept_bidir`。

### Phase 3 — Chebyshev 統計門 + N_valid fallback

spec §1 Pass1，接 Cheb-GR core（`perception/reid/cheb_gr.py`）。

- lost-bank 距離域 `μ−λσ` 截斷取代/疊加固定 chi2；`N_valid≤3` → fallback 固定門檻（防統計塌陷）。
- Flags：`--semantic-cheb-lambda` / `--semantic-cheb-min-valid`(3)。

### Phase 4 — GPU two-pass kernel 化（`tracker_gpu.cu`）

CPU 邏輯驗證後搬進零同步 GPU 核（spec §4），**擴充既有 relink bank**。

- Pass1 reduction/fallback：Birth candidates × lost-bank 算正/逆向馬氏 + N_valid 防呆。
- Pass2 bidirectional match & claim：疊 shape + cosθ 片上斷邊 → `parallel_auction_shmem_kernel` + `atomicCAS(relink_valid)`。
- 各向異性 Q 加進 GPU predict（或 relink 專用 predict）。
- 對接點：與 CPU 路徑共用 flags/語意；env 切 CPU/GPU 做指標對拍。

## 橫向工作（Cross-cutting 對接點）

- **fps 自動接線** ✅：`_resolve_kalman_fps`（`evaluator.py`）每序列 seqinfo.ini → .mp4 探測 → 30；`--semantic-kalman-fps 0`=auto。
- **lifecycle merger 補閘門**：把速度上限/方向/雙向加到 `TrackletLifecycleMergerCpp`（第二層目前裸奔）。
- **三層收斂**：釐清 ID4 類事件實際由哪層產生，確保門控覆蓋。
- **Ablation harness**：每階段 baseline vs on，含 GT 序列比 HOTA/AssA/IDF1/IDs；非 ±0.3pp 雜訊才升 default。
- **appearance-free 模式**（spec 拋棄外觀）：加 ablation toggle，量測純幾何上限。
- **parity test**：沿用 `tests/integration/test_identity_resolver_parity.py` 模式，每階段補 Python↔C++。

## 風險 / 備註

- 延遲認領晚 K 幀定案 id；retroactive remap 解決離線輸出，但**線上即時消費端**需注意延遲語意。
- 雙向需 spawn 速度可靠；warmup 不足→保守偏新 id（IDs 換 fragment，可接受）。
- GPU 各向異性 Q 增加 predict 成本；限 relink bank（數量少），非全 active track。
- 全程 default off，逐階段 ablation 後才動 preset。

## 實驗設定：人工遮擋 (Synthetic occlusion)

原始 `custom_seq` demo **無障礙物**，行人不被遮擋、軌跡乾淨，沒有長 gap 可測重連。為了反覆比較重連/門控效果，用 `scripts/tools/add_occlusion_to_seq.py` **手動注入一根中央遮擋柱**，製造真實的遮擋丟失：

```bash
uv run scripts/tools/add_occlusion_to_seq.py \
  --img-dir datasets/demo/custom_seq/img1 \
  --width-ratio 0.125 --color 60,60,60
```

- 遮擋框：畫面正中、寬 = `width_ratio × W`（0.125→480px @4K，x≈1680–2160）、高 = 55%×H、置中。行人穿越中央時會被吞掉 → 產生 gap → 觸發重連博弈。
- **這就是 ID4 的成因**：id4 在 cx≈1619 往右走、撞進遮擋柱（左緣 x≈1680）後丟失；理想是在柱子另一側（前方）重連回來，而非被左邊另一個人接走。方向/速度閘門正是防這個。

> ⚠️ **此工具就地覆寫 `img1`、不備份**，且目前 `img1` 已是含遮擋版。乾淨來源是
> `datasets/demo/15779246_3840_2160_60fps.mp4`（22MB，正好 821 幀 @3840×2160、59.94fps，
> 對得上 `seqinfo.ini`）。從 mp4 重抽乾淨幀：
> ```bash
> # 1) 從 mp4 還原乾淨幀（覆寫遮擋版）
> ffmpeg -i datasets/demo/15779246_3840_2160_60fps.mp4 -start_number 1 -q:v 2 \
>   datasets/demo/custom_seq/img1/%06d.jpg
> # 2) 一次性備份乾淨圖，之後 A/B 直接從備份還原（比 ffmpeg 快）
> cp -r datasets/demo/custom_seq/img1 datasets/demo/custom_seq/img1_clean
> # 注入遮擋前還原：rm -rf datasets/demo/custom_seq/img1 && \
> #   cp -r datasets/demo/custom_seq/img1_clean datasets/demo/custom_seq/img1
> ```
> 建議 A/B 流程：① mp4 重抽 + 備份乾淨圖 → ② 注入遮擋 → ③ 同一組遮擋圖跑「門控 off vs on」對比（避免每次重注入造成框位/JPEG 壓縮差異污染比較）。`datasets/` 已被 `.gitignore` 涵蓋（含 mp4、img1、`img1_clean`），皆不入庫。

## 快速重現 (Quick repro)

Phase 0 全開（卡方 + 方向 + 物理雪茄雲 + 速度上限），custom_seq demo（已注入遮擋）：

```bash
uv run scripts/eval/mot17.py --data-root datasets/demo --split . \
  --sequences custom_seq --preset mamba_whole_graph --reid-mode semantic \
  --semantic-ttl 120 --semantic-spatial-gate 0.45 --semantic-min-iou 0.0 \
  --semantic-threshold 0.85 \
  --semantic-kalman-gate --semantic-kalman-chi2 9.4877 \
  --semantic-kalman-person-height-m 1.65 \
  --semantic-kalman-accel-long 2.0 --semantic-kalman-accel-lat 1.0 \
  --semantic-kalman-dir-min-cos 0.0 --semantic-kalman-max-speed-mps 8.0 \
  --visualize
```

> `--semantic-kalman-fps` 預設 0 = **每序列自動解析**（`seqinfo.ini` frameRate → `.mp4` 探測 → 30）；custom_seq 自動取 60。需要時才用 `--semantic-kalman-fps <n>` 覆寫。

ID4 回歸檢查（無物理不可能跳變；輸出 MOT 檔在 `results/MOT17_eval/custom_seq.txt`）：

```bash
# 抓某 id 的每幀位移，>120px 視為可疑跳變
awk -F, '$2==4 {cx=$3+$5/2; if(p!=""){d=cx-pcx; if(d<0)d=-d;
  if(d>120) printf "JUMP f%d->f%d move_x=%.0f\n",pf,$1,d} p=1;pf=$1;pcx=cx}' \
  results/MOT17_eval/custom_seq.txt
```

**回歸基準**：Phase 0 全開時 id4 應止於 ~f510（中央往右的人偵測消失處），**無**跨畫面跳變；左邊的人保留自身 id（不被併吞）。relink report 應見 `reject_speed>0`。對照組（拿掉 `--semantic-kalman-*`）id4 會被內插出一條 153px/frame 的鬼軌跡飛到最左邊。

> ✅ fps 已自動接線（`_resolve_kalman_fps` @ `evaluator.py`）：每序列依 `seqinfo.ini` frameRate（configparser 大小寫不敏感，`framerate`/`frameRate` 皆可）→ 同層/上層 `.mp4` 探測（cv2）→ 30。物理模型對 fps 平方敏感，故必須對齊真實來源。

## 參數速查 / 建議起始值

| 參數 | 預設 | 建議起手 | 物理意義 |
|---|---|---|---|
| `semantic_kalman_gate` | off | on（啟用物理門控總開關） | 卡方雲門控 |
| `semantic_kalman_chi2` | 9.4877 | 9.4877（4DoF 95%） | 越小越嚴（5.99=80%、13.28=99%） |
| `semantic_kalman_penalty_weight` | 0 | 0（純硬門）或 0.3 | 軟懲罰 `1−exp(−D²/2)` 入 joint |
| `semantic_kalman_dir_min_cos` | -1(off) | 0.0（斬 >90° 後方） | cosθ 方向閘 |
| `semantic_kalman_dir_min_speed` | 1.0 | 1.0 px/frame | 低於此速度不做方向判斷 |
| `semantic_kalman_person_height_m` | 0(off) | 1.65 | 開啟物理雪茄雲 + scale anchoring |
| `semantic_kalman_accel_long` | 2.0 | 2.0 m/s² | 縱向（加減速）最大加速度 |
| `semantic_kalman_accel_lat` | 1.0 | 1.0 m/s² | 橫向（轉彎）最大加速度，`< long` 才有慣性 |
| `semantic_kalman_fps` | 0(auto) | 0（自動讀 seqinfo/mp4） | m/s²→px/frame² 換算；>0 覆寫 |
| `semantic_kalman_max_speed_mps` | 0(off) | 8.0（人類衝刺） | 隱含平均速度上限，超過→判新 id |

## 相關

- 程式：`src/tracking/tracker_gpu_python.cpp`（C++ relinker）、`src/saccade/perception/eval/relink.py`（Python 鏡像）、`include/tracking/kalman_gpu.cuh`、`src/tracking/tracker_gpu.cu`（GPU relink bank）。
- 模組：[semantic README](../README.md)、[reid](../../reid/README.md)、[lifecycle](../../lifecycle/README.md)、[motion](../../motion/README.md)。
- 規範：`kalman_gating_guide.md`、`bidirectional_spacetime_tracking_spec.md`。
