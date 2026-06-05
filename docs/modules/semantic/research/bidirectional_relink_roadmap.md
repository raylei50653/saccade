# 雙向時空收斂幾何重連 — 長期路線圖

> Bidirectional Space-Time Convergent Kinematic Relink Roadmap
> 狀態：Phase 0 已落地（default off）；Phase 1 CPU 延遲認領骨架已落地（default off，待 MOT17/ID4 ablation）；Phase 2–4 規劃中。最後更新 2026-06-04。

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

### Phase 1 — 速度取得 + 延遲認領骨架（CPU）✅ skeleton landed / default off

新框**不再首現即認領**，先緩衝 K(≈confirm_streak=3) 幀讓 Kalman 收斂出穩定 `v_spawn`。

- Spawn 速度來源：優先用 tracker 對「當前解析 track id」的 Kalman 快照（擴充 `get_motion_snapshots_for_track_ids` 取用對象）；fallback 用 relinker `last_boxes_` 歷史估速。
- 延遲認領佇列：would-be-new 的 raw_id 記入 pending；滿 K 幀或 tracker confirmed 才觸發認領博弈。
- **對接點**：relinker 暴露 `deferred_alias: {new_id→claimed_old_id}`；evaluator 在 `interpolate_tracklets` 前對 `results_lines` 套用 remap（新工具 `apply_deferred_alias`，緊鄰 `filter_low_quality_tracklets`）。
- Flags：`--semantic-delayed-claim`、`--semantic-claim-warmup-frames`(3)。

實作快照（2026-06-04）：
- Python relinker 已支援 pending warmup + `deferred_alias`。
- evaluator 會在 `filter_low_quality_tracklets` / `interpolate_tracklets` 前套用 `apply_deferred_alias`。
- `--semantic-delayed-claim` 目前強制走 Python relinker，避免尚未重建的舊 C++ extension 收到新 kwargs；C++ `SemanticRelinkerCpp` source 已同步鏡像，待本機 CUDA/OpenCV toolchain 可重建後切回預設 C++ 路徑驗證。
- 已補 focused tests：warmup 前保留 raw id、warmup 後產生 deferred alias、MOT lines remap。

### Phase 2 — 反向傳播 + 雙向聯合門控（CPU）✅ skeleton landed / default off

spec §2/§3。認領須**正向∩逆向**雙過。

實作快照（2026-06-05）：
- Python relinker 已具備 `--semantic-bidirectional` 開關，預設關閉。
- **中點橋接閘（線性代數，無 Kalman 依賴）**：lost 方取 tail 4 幀 foot-centre 線性回歸得速度、candidate 方取 head 4 幀同公式反向得速度，各線性外推至 gap/2，取兩預測點歐氏距離除以平均 EMA 框高，與 `--semantic-bridge-px`（框高歸一化單位，預設 1.5）比較。
- **速度回歸閉式解**（4 幀等間隔）：`v = (3p₃ + p₂ − p₁ − 3p₀) / 10`，不需 Kalman snapshot、不需 `predict_phys` 迭代、不需協方差矩陣。
- **框高歸一化**：每 track 維護 EMA 框高（`α=0.05`，≈60 幀衰減窗），中點距離除以 lost/cand 平均框高，消除透視 bias（遠小近大）。
- **foot history**：每 track 保留最近 8 個 foot-centre（`_foot_history`），resolve / motion_only 兩路徑皆更新。
- CLI flags：`--semantic-bidirectional`、`--semantic-bridge-px`（1.5，框高倍數）。
- Stats：`reject_backward`、`accept_bidir`。
- 不依賴 Kalman snapshot（`snapshot is not None` 條件已移除），純幾何模式可獨立運作（`--reid-mode off --pipeline-relink`）。

設計取捨：
- 不用 Mahalanobis D²：長 gap 時協方差膨脹過大，D² 失去鑑別力（真/假配對 D² 值接近）。
- 不用 Kalman 反向傳播：candidate 方 Kalman 未收斂（僅 4–5 幀），回歸速度更可靠。
- 不用完整 bridge_overlap（Bhattacharyya）：中點等面積比較已足夠，閉式解省去矩陣運算。

#### 離線 bridge 驗證（`scripts/tools/render_diffusion_debug.py`，2026-06-04）

雙向夾擊的核心量＝「lost 軌跡正向擴散的腳點」對「candidate 逆向外推的腳點」之距離（`bridge`，L600–619）。先用離線 diagnostic tool 把這個量視覺化、掃 source 速度估計器，再決定要不要進 relinker。

**關鍵發現：source 速度應取「丟失前一段乾淨窗」，而非緊貼 tail 的位移。** tail 末端常已減速/被遮擋汙染，直接估速會低估真實 `v_spawn`，正向腳點落不到 candidate 那側。新估計器（`estimate_model`，`velocity_mode=pre_tail` / `velocity_stat=endpoint`）改從 stable anchor 往前 `velocity_offset` 幀、取 `velocity_window` 幀的 first→last endpoint 斜率：

- 起手值：`velocity_mode=pre_tail`、`velocity_stat=endpoint`、`velocity_offset=13`、`velocity_window=30`。

custom_seq_occ 對照（`--bridge-gate-px 120`）：

| link | bridge | 判定 |
|---|---|---|
| `#2 → #5`（真實重連） | 117.5px | pass |
| `#4 → #5` | 498.1px | reject |
| `#5 → #7` | 428.5px | reject |

兩類分離很寬（117 vs 428/498），方向上驗證了 bridge 量可分真/假重連。原 `--bridge-gate-px 120` 真實 link 只低 2.5px、裕度太薄，已將 distance-mode default **抬到 200**：真實 #2→#5（117.5px）有 ~82px headroom，rejects 皆 >400px 仍安全落在門外。⇒ Phase 2 正向腳點速度建議沿用 pre_tail/endpoint 估計器。

**bridge 已從硬距離升級成概率波重疊（`--bridge-mode overlap`，default）。** 原本 `bridge = hypot(Δfoot)` 對固定 px 門，丟掉兩端各向異性「雪茄雲」的協方差。新版把正向/逆向各建 2D 高斯 `Σ = R(θ)·diag(σ_long²,σ_lat²)·Rᵀ`（`cloud_cov`），以兩量門控（`bridge_overlap`）：

- **Mahalanobis 相遇檢定**：`D² = δᵀ(Σ_f+Σ_b)⁻¹δ`，卡方 2DoF 門（`--bridge-chi2` 5.991=95%）。scale-adaptive——遠處兩朵大雲容許較大 px 間距，近處兩朵小雲收緊，取代固定 px。
- **Bhattacharyya 重疊權重**：閉式高斯重疊係數 ∈(0,1]（`--bridge-min-overlap`，0=off），其行列式項同時懲罰形狀/尺度不匹配，部分吸收既有 scale check。

`--bridge-mode distance` 保留舊硬距離行為供 A/B。⇒ Phase 2 正式門控採概率重疊，非固定 px。

**速度倍率範圍補償（`--vel-range-frac`，default 0.2）。** 原 σ 只來自 `base_size + accel·dt²`，完全不看物體實際速度——快速新 ID 拿到跟靜止 ID 一樣緊的範圍，門失去參考意義。每端（正向 lost cloud + 逆向 candidate cloud）航向 σ 各加 velocity-range 項：快軌得到正比於自身速度×gap 的更寬接受範圍，慢/靜止軌維持收緊。**關鍵：candidate（新 ID）範圍須按新 ID 自身速度補正，非沿用 lost 軌速度。** 0=off 退回 base+accel。

**速度單位＝框高（box-height），非 m/s。** 框高就是「就地量到的透視尺度」，省掉 `person_height_m` 假設（`px_per_m = h/1.65`，框高與 m/s 僅差此常數）。式子寫成 `vel_long = frac · (speed·gap / h) · scale_h`：`heights_travelled = speed·gap/h` 為框高歸一化位移、`scale_h` 為回投 px 的尺度（目前 = 當前 h）。數值與直接 px 等價（h 抵消），但語意變「走過幾個框高的 frac」＋把 `scale_h` 顯式化——**這正是透視/深度修正的掛鉤點：把 `scale_h` 換成相遇點的預測框高，雲就自帶深度修正**（近大遠小）。

**ReID 式品質閘 EMA bank（`--velocity-mode bank`）。** 丟失前數幀框不穩（遮擋起點→縮框/截斷），直接從 tail 估速/估幾何不可靠。借鑑 ReID reference-quality bank：`quality_bank()` 前→後走訪，維護幾何 EMA，只收「尺寸撐過 running EMA × ratio + score 過閘」的高品質幀；不穩尾幀自動落選。velocity/geometry 取最後一段穩定 stretch，diffusion 從**最後可信 snapshot**（`last_frame`=最後穩定幀，非 raw last）外推、自然跨過不穩尾。合成測試：20 穩定幀 +5 縮框尾 → anchor 止於 f20、EMA h 保 200（非污染 60）、vx≈10。把固定 `velocity_offset=13` 變成自適應（實際有幾幀不穩就跳幾幀）。Flags：`--quality-score-thr`（score 閘，<0 如 GT 恆過）、`--bank-alpha`（EMA 率 0.1）。對接 Phase 1 spawn 速度（穩定 Kalman snapshot）；真 relinker 有 appearance 時 bank 可再掛 embedding tier。

**透視/深度修正已接（`--perspective-scale`，default off）。** `predicted_height()` 用地平面 pinhole proxy（apparent height ∝ image row，horizon 近似 y=0）：`h_pred = h · (foot_y_pred / foot_y_ref)`，ratio clamp 到 `[1/3, 3]` 防高處塌陷。開啟後 `scale_h` 改用 `h_pred`：目標往畫面下方（近）走→預測框高放大→velocity-range σ 隨之放大；往上（遠）走則收緊。正向 lost cloud 與逆向 candidate cloud 各用自身 `foot_y_ref→pred_foot_y`。default off（改變數值），先 A/B 再決定。下一槓桿：把同一 `predicted_height` 套到 scale-acceptance check（candidate.h vs 預測 source.h），修正久丟期間 source 尺度漂移。

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

原始素材 **無障礙物**，行人不被遮擋、軌跡乾淨，沒有長 gap 可測重連。為了反覆比較重連/門控效果，用 `scripts/tools/add_occlusion_to_seq.py` **手動注入一根中央遮擋柱**，製造真實的遮擋丟失。eval **寫死讀 `<seq>/img1`**（不看 `seqinfo.ini` 的 imdir），所以乾淨/遮擋各做成**獨立 sequence**，直接 `--sequences` 對比、輸出兩個可比對檔，免每次 cp：

| sequence | `img1` 內容 | 中央像素驗證 |
|---|---|---|
| `custom_seq_clean` | 無遮擋（從 mp4 重抽） | `[127,111,99]` 真實場景 |
| `custom_seq_occ` | 含中央遮擋柱 | `[60,60,60]` 遮擋框 |

- 遮擋框：畫面正中、寬 = `width_ratio × W`（0.125→480px @4K，x≈1680–2160）、高 = 55%×H、置中。行人穿越中央時被吞掉 → gap → 觸發重連博弈。
- **這就是 ID4 的成因**（在 `custom_seq_occ`）：id4 在 cx≈1619 往右走、撞進遮擋柱（左緣 x≈1680）後丟失；理想是在柱子另一側（前方）重連回來，而非被左邊另一個人接走。方向/速度閘門正是防這個。

> 重新生成（乾淨來源 = `datasets/demo/15779246_3840_2160_60fps.mp4`，22MB，正好 821 幀 @3840×2160、59.94fps）：
> ```bash
> # 乾淨幀
> ffmpeg -i datasets/demo/15779246_3840_2160_60fps.mp4 -start_number 1 -q:v 2 \
>   datasets/demo/custom_seq_clean/img1/%06d.jpg
> # 遮擋幀（先複製乾淨幀，再就地注入）
> cp -r datasets/demo/custom_seq_clean/img1 datasets/demo/custom_seq_occ/img1
> uv run scripts/tools/add_occlusion_to_seq.py \
>   --img-dir datasets/demo/custom_seq_occ/img1 --width-ratio 0.125 --color 60,60,60
> ```
> `datasets/` 整個被 `.gitignore` 涵蓋（mp4 與兩個 sequence 皆不入庫；fresh clone 無 demo 資料，需自備）。

## 快速重現 (Quick repro)

Phase 0 全開（卡方 + 方向 + 物理雪茄雲 + 速度上限），遮擋實驗組 + 乾淨對照組**一次跑兩個 sequence**：

```bash
uv run scripts/eval/mot17.py --data-root datasets/demo --split . \
  --sequences custom_seq_occ,custom_seq_clean --preset mamba_whole_graph --reid-mode semantic \
  --semantic-ttl 120 --semantic-spatial-gate 0.45 --semantic-min-iou 0.0 \
  --semantic-threshold 0.85 \
  --semantic-kalman-gate --semantic-kalman-chi2 9.4877 \
  --semantic-kalman-person-height-m 1.65 \
  --semantic-kalman-accel-long 2.0 --semantic-kalman-accel-lat 1.0 \
  --semantic-kalman-dir-min-cos 0.0 --semantic-kalman-max-speed-mps 8.0 \
  --visualize
```

> `--semantic-kalman-fps` 預設 0 = **每序列自動解析**（`seqinfo.ini` frameRate → `.mp4` 探測 → 30）；兩個 sequence 都自動取 60。需要時才用 `--semantic-kalman-fps <n>` 覆寫。

ID4 回歸檢查（遮擋組不應有物理不可能跳變；輸出在 `results/MOT17_eval/custom_seq_occ.txt`）：

```bash
# 抓某 id 的每幀位移，>120px 視為可疑跳變
awk -F, '$2==4 {cx=$3+$5/2; if(p!=""){d=cx-pcx; if(d<0)d=-d;
  if(d>120) printf "JUMP f%d->f%d move_x=%.0f\n",pf,$1,d} p=1;pf=$1;pcx=cx}' \
  results/MOT17_eval/custom_seq_occ.txt
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
| `semantic_bidirectional` | off | on | 啟用中點橋接雙向閘 |
| `semantic_bridge_px` | 1.5 | 1.5（框高倍數） | 中點雲心距離 ÷ 平均框高，越小越嚴 |

## 相關

- 程式：`src/tracking/tracker_gpu_python.cpp`（C++ relinker）、`src/saccade/perception/eval/relink.py`（Python 鏡像）、`include/tracking/kalman_gpu.cuh`、`src/tracking/tracker_gpu.cu`（GPU relink bank）。
- 模組：[semantic README](../README.md)、[reid](../../reid/README.md)、[lifecycle](../../lifecycle/README.md)、[motion](../../motion/README.md)。
- 規範：`kalman_gating_guide.md`、`bidirectional_spacetime_tracking_spec.md`。
