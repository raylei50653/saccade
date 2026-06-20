# Saccade NO-GO 全局登記表 (Global NO-GO Registry)

> **用途**：跨模組「已結案/已踩雷方向」總覽，避免重複探索。每列只記結論一行，數據細節以對應模組的 `research/` 或 `decisions/` ADR 為準。
> 彙整自 `decisions/`、`modules/*/README.md`、`archive/`、`reference/PIPELINE_REFERENCE.md`、`TODO.md`（路徑相對 `docs/`）。
> 最後更新：2026-06-20

**死因分類**（中性結案前必須回答：訊號不存在，還是被遮蔽？）：

| 類型 | 意義 | 復活可能 |
|------|------|----------|
| 🔻 **有害** | 開啟後指標明確退步 | 無，除非前提改變 |
| ❌ **結構天花板** | 訊號本身無區分力（AUC≈隨機、物理瓶頸） | 無 |
| ⚪ **中性（被遮蔽）** | 增益在雜訊內，但已識別 blocker（上游門/低 base rate/記憶長度） | **有** — blocker 移除後可組合復活，見[復活前例](#中性-no-go-的復活前例) |

> **分類門檻**：分類必須附**歸因實驗 + 統計資料**（AUC、候選攔截率、ablation 對照組之類），不可從結果倒推。無歸因實驗的條目**不分類**，只記結果（下表多數歷史條目即屬此類 — 結果論記錄，死因未歸因）。日後若補做歸因實驗，再回填分類。

**GO 判定門檻**（2026-06-12 修訂）：

- 固定 `+3pp` GO 門檻**已廢止** — 它定於 baseline ~50% IDF1 的時代；現行 baseline ~75%，剩餘空間只剩 25pp，固定幅度門檻會系統性誤殺。先例：[mamba 雙解析度研究](../modules/detection/research/mamba-dual-resolution-original-detail-plan.md)已在 detection recall 場域廢止同一門檻。
- 現行原則：**任何超出同日 paired 噪聲（±0.3pp）的變化即有討論價值**。GO 判定看（1）方向跨 seq / 跨 score threshold 一致、（2）paired 95% CI 排除 0、（3）無其他指標嚴格退步 — 不看固定幅度。
- 高 baseline 下同時報告 absolute delta 與**相對剩餘空間的比例**（例：75% 時 +1pp = 吃掉 4% 剩餘 headroom，等價於 50% 時代的 +2pp）。

---

## 核心 NO-GO（有完整 ablation 證據）

| # | 項目 | 時間 | 結論 | 關鍵數據 | 代碼狀態 |
|---|------|------|------|----------|----------|
| 1 | **Option D** Track-Conditioned YOLO | 2026-05-19 | ❌ NO-GO | IDF1 31.7% vs baseline 52.0%，gate ∆ <0.2pp | 歸檔 `docs/archive/option-d/` |
| 2 | **Appearance ReID Bank** (GMC ON) | 2026-05-13 | ❌ NO-GO | IDF1 ±0.0pp、FPS **−17.3**（零增益高代價） | default OFF |
| 3 | **Semantic Relink** (GMC ON) | 2026-05-13 | ❌ NO-GO | GMC + GPU bridge relink 取代；86.8% 候選被 age gate 拒絕 | default OFF |
| 4 | **Appearance 能力上限** | 2026-06-03 | ❌ **結案** | 5 模型 × 4 機制 × SR × 域訓練全撞同一天花板；清晰 200+px 框 rank-1 僅 57%，intra-inter gap ~0.03 | 模組保留 default OFF |
| 5 | **Tiled Detection** (960p 2×2/3×2) | ~2026-05 | ❌ NO-GO | FP ~8000（native_960 的 2 倍），truncation + score 污染 | code 保留，非 default |

## 輔助 NO-GO（邊際/中性/代價過高）

| # | 項目 | 時間 | 結論 | 關鍵數據 |
|---|------|------|------|----------|
| 6 | Motion-based Relinking | 2026-05-17 | ❌ NO-GO | 89% 候選被 age gate 攔截，增益 ≈ 雜訊 |
| 7 | OA-SORT OAO | 2026-05-20 / 06-12 / **06-17 復活** | ✅ **GO（duration-ramp）** | 舊：⚪ 被遮蔽（整列加 cost；舊 baseline ±0.3pp、06-12 新 baseline −1.1pp）。**06-17：new whole-graph+bidir bridge baseline 下 plain OAO tau0.25 翻盤 +1.6 IDF1/+2.6 AssA，但集中 MOT17-04（42% 權重）且 05 −3.4**。6 空間判別信號（contention/score_w/union/crowd/height-gate/foot-gate）全 NO-GO（05 害與 10/13 益在空間軸纏結不可分）；**破牆=時間軸**：05 重疊短暫~10f、04 持久~49f → duration-ramp `tau·occ·min(1,frames/ramp)` 保 04 抑 05。06-17 evidence 使用 tau0.30+ramp25 且 Pareto 支配 plain（每指標 ≥，ex-04 AssA +0.8，05 70.6→72.3）；目前 preset 已 retune 為 `oao_tau=0.50`, `oao_ramp_frames=25`。 [復活分析](../research/eval/oao_duration_ramp_revival_20260617.md) |
| 8 | NSA-Kalman (Noise Scale Adaptive) | 2026-04-27 / **06-12 歸因+重校準** | ⚪ 被遮蔽 | 前提成立（Spearman −0.52，低分框殘差 5×）但與 `kalman_r_scale=2.8` 雙重補償；新 baseline 端到端 −3.2pp。**06-12 重校準驗證**：f(score) v2（只放大、s0=0.9305 錨定）IDF1 **+1.5** / AssA +0.86（6/7 seq 正）— score 訊號可轉化；但 DetA −1.44 / MOTA −0.4，移動相機 05/13 嚴格退步（13: MOTA −3.4/IDs +33），per-seq CI 含 0 → default 不開。g(h) 形狀修正單獨 **−0.4**（h 代理假說反證：score 是主訊號）。[歸因分析](../research/eval/neutral_nogo_signal_attribution_20260612.md) |
| 9 | PostMerge | 2026-04-27 / **06-12 歸因** | ⚪ 被遮蔽 | combined AUC 0.868 但 base rate 2.4% → 作用點 precision ~20%（FP +1308）；default 路徑因強制 appearance gate + ReID off **失能**；direction 分量 AUC 0.487 純噪聲。[歸因分析](../research/eval/neutral_nogo_signal_attribution_20260612.md) |
| 10 | Per-frame Detection Cap / Adaptive Cap | — | ❌ NO-GO | 密集場景 adaptive cap 壓至 ~21，破壞 recall |
| 11 | P5-2 Stage2 QualityGate | ~2026-05 | ❌ NO-GO | IDF1/MOTA 統計中性 |
| 12 | P5-3 ConsecutiveBirthGate | ~2026-05 | ❌ NO-GO | 統計中性 |
| 13 | P5-4 Scene-Adaptive | 2026-05-11 | ❌ NO-GO | — |
| 14 | P5-5 Proximity Birth Gate | 2026-05-18 | ❌ NO-GO | prox=0.3 → FN +1038 / Rcll -5.6pp |
| 15 | LaSt-ViT pre-hoc embedding quality | 2026-05-02 | ❌ NO-GO | +0.09pp，SigLIP2 未訓練無區分力 |
| 16 | ROI FPN ReID | 2026-05-19 | ❌ NO-GO | cos_thr 全設定 IDs↑、IDF1 持平 |
| 17 | Horizontal-flip TTA | 2026-05-18 | ❌ NO-GO | 精度在雜訊內 |
| 18 | MOT20 混訓 | 2026-06-01 | ❌ NO-GO | domain shift 退步 |
| 19 | Pose box expansion | 2026-05-10~11 | ❌ NO-GO | 靜態 FP 無法靠 spatial 區分 |
| 20 | GMC FG Mask | 2026-05 | ❌ NO-GO | 背景紋理主導 PCR peak |
| 21 | Vel_dir gate | 2026-06-01 / **06-12 歸因** | ⚪ 被遮蔽 | 訊號分層真實（fast >3px/f AUC 0.751、slow <1 AUC 0.526≈隨機、慢速樣本占 46%）；無速度條件化 → 新 baseline 端到端 −4.6pp。[歸因分析](../research/eval/neutral_nogo_signal_attribution_20260612.md) |
| 22 | Cheb-GR offline tracklet merge | 2026-06-03 | ❌ NO-GO | AssA 0.0pp |
| 23 | Birth-time lost-bank relink (GPU) | 2026-06-03 | ❌ NO-GO | 無 λ 能降 IDs；長 gap rank-1 僅 13–33% |
| 24 | YOLO non-end2end (cxcywh output) | 2026-05-08 | ❌ NO-GO | 整體退步，不升格 default |
| 25 | Cascade Filter (CrowdHuman→MOT17) | 2026-05-14 | ❌ NO-GO | MOT17 FP score 與 TP 重疊嚴重 (P≈4%)，rule 僅砍 13.3% FP |
| 26 | Pose Bio gate (Biometric relinker) | 2026-05-10 | ❌ NO-GO | Gate 僅 3 veto / 7-seq，FPS -47% |
| 27 | Narrow person score bonus | 2026-05-11 | ❌ NO-GO | 全局 IDF1 -0.3pp，FP +378 |
| 28 | Mamba temporal block (SSM, v15/17) | 2026-05-31 | ❌ NO-GO | R1→R2 grad 崩潰無法收斂 |
| 29 | Per-channel SSM A + MOT20 mix | 2026-06-01 | ❌ NO-GO | DetA 退化 -1.8pp（domain shift） |
| 30 | Cheb-GR standalone (Market-1501) | 2026-06-03 | ❌ 方法成立但不優於 fixed-k | +8.76pp vs classic +10.03pp |
| 31 | Relink bridge **scale gate** (speed 方向) | 2026-06-11 | ❌ NO-GO | MOT17-SDP 小幅正向但速度方向全線死；P0 L_med 復核不重現 |
| 32 | **Appearance relink gate**（顏色直方圖 + OSNet hard pool） | 2026-06-11 | ❌ **結案** | 全 gate AUC≈0.50、短 gap 反向 0.33；外觀方向結案 |
| 33 | **occ_cover live relink**（gap-path 占用門） | 2026-06-11 | ❌ NO-GO | live accepts 全 gap≤1；長 gap 族群被 track_buffer=30 結構性消滅；tb90 解鎖反 −0.8 IDF1 |
| 34 | **GMC box-residual 共模修正**（innovation 自迴饋） | 2026-06-12 | ❌ NO-GO | 全 4 模式負（最佳 lost-only 74.7 vs 75.1，IDs +26）；劑量反應單調、lost-only 仍負 → GT affine 共模上限（13: 41%）**不轉移到 innovation 空間**（Kalman 速度已吸收持續殘差）。[結案文件](../research/eval/gmc_residual_correction_20260612.md) |
| 35 | **Mamba head 特徵作 relink embedding**（含 T3→T1 一致性特徵） | 2026-06-13 | ❌ NO-GO | 21k 候選離線 AUC：T3→T1 hard pool **0.438**、replica **0.438**（配對差 ~0.001）、full pool 均 0.51；短 gap 反向 0.33-0.34 與外觀探針（#32）完全同構。歸因：**consistency ≠ discriminability** — T3→T1 curriculum 端到端 AssA +3.2 全部經由 box/score 穩定性（IoU 路徑）傳導，不在特徵空間攜帶身分訊號；detection 特徵編碼「人＋幾何」對個體無區分力（與 #16 ROI FPN ReID 一致）。探針 `scripts/tools/mamba_relink_features.py` |
| 36 | **小目標高解析度恢復**（B1-H dense / 1024 unified / strip-oracle routing） | 2026-06-13 | 🔻 **ROI NO-GO（成本判定 · 未取 oracle 數字 · ⏸ parked）** | **增益天花板**（MOT17-02）：`min_4to8` 2259 GT@0.826 → 完美救回上限 ~390 框、`h_lt32` 僅 44 GT → 最樂觀 **<0.5pp IDF1**（dense B1-H 實測僅 +0.3pp）。**成本**：1024 unified 重編 TRT + backbone 1.5–2×；dense B1-H **+1.4ms/frame**（all cells）；strip-oracle ~500 行 + 兩階段訓練 + per-cell gather/scatter。根因：640 resize 對小目標是不可逆資訊損失（24px→P3 ~1px），唯一補救是讀原圖像素，但補救成本/增益比過低。**未取得 oracle routing recall 天花板（屬成本判定非 signal 判定，不可當結構天花板引用）**。strip-oracle Phase 1 已實作+6 tests pass+uint8 VRAM 修復，代碼保留 default off。**不在此 NO-GO 內**：level-routing postprocess（`small_p3_max_threshold`，免訓練、重用既有多層 score、非高解析度）獨立待評。**復活條件**：其他方向耗盡時重啟，先跑 strip-oracle Phase 1 取 oracle 天花板再定生死。設計：`docs/modules/detection/research/mamba-strip-detail-routing-design.md` |
| 37 | **顯式跨幀一致性保護項**（route 3：相鄰幀同 track P3 cls logit L2） | 2026-06-13 | ❌ NO-GO | weight sweep {0.1,0.3,1.0} 對照 weight=0 control：項機械上**確實生效**（raw cons 2.04→0.15-0.37，weight 越大壓越緊）但 tracking **單調退步** — AssA cw0 65.4 > cw0.1 64.3 > cw1.0 63.2 > cw0.3 62.6；IDF1 75.7 > 74.3 > 73.5 > 73.3。歸因：硬性 L2 強制跨幀 cls 相等會**壓掉合法的逐幀 score 動態**（遮擋/解遮擋時信心本應變化，該變化攜帶 association 訊號）；AssA 所需的「一致性」是 T=3 壓力下的 emergent/soft 性質，不能當 loss 硬編（呼應 #35 consistency≠discriminability、§3.2 塑形在權重空間 local/fragile）。**副產物（單 seed 暗示性、已 park）**：weight=0 control（T=3 SSM-unfreeze 15ep + T1 re-adapt）IDF1 75.7/DetA 70.7/AssA 65.4，同時拿到 DetA(≈ssmft 71.0)+AssA(≈star 66.0)，而 doc 的 T1-ssmft 把 AssA 崩到 62.8 → 暗示「解凍 SSM 須在 T=3 模式做才保得住塑形」；但 +0.3 IDF1 在 0.4pp seed band 內，未跑 matched multi-seed 故**不可當已證結論**。腳本 `run_v14replica_consistency{,_sweep}.sh`、`_temporal_consistency_loss` |
| 38 | **框高條件化出生門檻**（height-conditioned `new_track_thresh`：大框升 / 小框降） | 2026-06-13 | ❌ NO-GO（雙向 oracle 上限結案，未寫 C++） | 動機=分數分佈呈飽和左尾、門檻坐 GT 0.3% 薄尾、precision 隨框高遞減（detection 層大框低分 79% FP）。**大框升門檻**：最寬鬆 oracle（baseline 輸出 post-filter 刪所有 frame 的大框低分框，birth-only 的嚴格超集）h≥128&s<0.28 IDF1 **持平 75.4**（MOTA −0.3/Rcll −0.4 換 Prcn +0.2），其餘 cut 全淨負 → 上界 ≤0。死因：存活到輸出的「大框×低分」框**只 33% 是 FP**（s<0.5 時 75% 是真人壞幀），confirm gate(streak3+score0.5)早濾掉純 ghost，detection 層 precision 梯度**不轉移到輸出空間**（同 #5 GMC「GT 上限不轉移 innovation 空間」）。**小框降門檻**：recall-side oracle 上界僅 **0.31% GT**（小框 FN 且偵測落 [0.15,0.28)，且高估—該帶 69% FP），實測全域 ntt0.15 Rcll 反降；真瓶頸=**3788 個已達門檻仍漏的 GT（關聯/confirm 失敗，門檻救不了）**，是門檻可救量(324)的 12×。**副產物（曾誤判 GO，已撤回）**：全域 `new_track_thresh` 0.28→0.20 在 7-seq 聚合看似 IDF1 +0.1/IDs −4.6%，但**逐序列 2/7 正 3/7 負、聚合全靠 MOT17-09 +2.9 撐起**（其餘 −0.9~+0.6），違反 GO 第一條「跨 seq 一致」→ 場景過擬合，preset 維持 0.28。教訓：確定性 eval 的聚合 Δ 仍需逐序列一致性檢驗（test GT 在 MOTChallenge server 保留，本地無法評，per-seq 一致性是唯一本地過擬代理）。腳本 `oracle_height_birth_ceiling.py`、`oracle_small_birth_ceiling.py`、`analyze_score_distribution.py`、`run_threshold_strategies.sh`；研究 `modules/detection/research/mamba-score-distribution-20260613.md` |
| 39 | **Depth-ordering crossing-swap fix**（occluder-side 深度互斥；佔 22% IDs 的遮擋交叉 swap） | 2026-06-14 | ⚪ **conditional / scene over-fit（非 default，code default-off bit-exact）** | 訊號真實：probe 遮擋前 foot_y 預測前後 **90%（decisive 97%）7/7 一致**、oracle 完美修正天花板 **+4.1 IDF1/+4.4 AssA**（crossing 佔總關聯 headroom ~⅓）。**機制歸因**（`analyze_crossing_swaps.py`）：72% ABSORB（occluder 吃 box，63% 確為 occluder）、79% 兩框可 auction 修。**Phase-1 occludee-side NO-GO**：cost term **bit-identically inert（w=0≡w=0.6）**+ OCCLUDED state 蠶食 bridge → IDF1 −1.2；hook 錯邊。**Phase-2 occluder-side**：latched front-flag→occluder back-box penalty（無 state 變更、bridge 不動）；peak `ttl4 w0.5 foot0.15` aggregate **IDF1 +0.5/AssA +0.3/IDs −18/FP −293**（真生效），但 **per-seq 不一致**（05 −1.1/02 −0.4，靠 09+2.9/10+1.5 撐）= **ntt0.20(#38) 同型 over-fit**。05 負**三假說全否證**：非 over-firing（exposure 最低）、非 precision（04 fire 7× 仍 +）、**非 depth 可靠度（05 foot 93.5%，worst-foot 10/13 反 gain，反相關）→ per-cam horizon 修不了**。結論：正確 hook+真增益但 scene over-fit、非 default；depth 天花板 ~90%，閉合殘量需交叉點 **identity 訊號**（繞回 ReID 牆 #2/#32/#35）。腳本 `oracle_occlusion_hold.py`/`analyze_crossing_swaps.py`/`analyze_front_flag_exposure.py`/`depth_ordering_probe.py`；研究 `modules/semantic/research/depth_ordering_crossing_swap.md` |
| 40 | **tile-PCR affine GMC**（per-tile 相位相關擬 4-DOF similarity，補 global 平移-only PCR 缺的 rotation/scale；memory 標為「唯一未否證」的旋轉路線） | 2026-06-15 | ❌ **NO-GO（結構性；史上最佳 affine 嘗試，勝 LK）** | 動機=GT 共模殘差 affine 額外增量 13:**41%**、11:33%、05/09/10:19–22%（GMC affine 探針 memory `project_gmc_affine_probe`）。**實作**（`eval/gmc.py:TilePhaseCorrAffineGMC`，`--gmc-mode tile`，default off）：tile 切格→批次相位相關得 per-tile (disp,PCR)→PCR 加權 + Huber robust LS 擬 similarity(s,θ,tx,ty)；**失敗 fallback global PCR 平移（非 identity）= 現行 GMC 嚴格超集**。合成測試還原旋轉誤差 **<0.05°/s≈1.0**（估計器本身無誤）。**7-seq v1**：aggregate IDF1 **49.99→49.74（−0.25）**、3 贏（04 +0.4 / 09 +1.0 / 13 +0.1）4 輸（02 −1.0 / 05 −0.2 / 10 −1.2 / 11 −0.8）→ 違反 per-seq 一致；**但勝舊 LK affine（memory `project_gmc_affine_probe`：−0.8、10 崩 −4.6）**。**四槓桿全否證**：①選擇性接受門（v2）**反效果**（間歇 affine = warp 時序不一致；13 IDF1 跌破 baseline、10 AssA −2.4）；②FG 逐-tile 排除（`--gmc-fg-mask`，前景框占比降權）**wash**（aggregate 仍 −0.25，只把 09/13 贏家與 02/10/11 輸家**同步縮小**）；③frame-FG 門（整幀乾淨才啟用 affine）**離線探針事前否決** — 前景占比不分贏輸：**MOT17-10（最慘輸家）meanFG 11.8% 為次低、< 贏家 04 的 18.5%**；④框高 tile 排除（框高=深度代理、視差∝框高)**視差–框高探針否決** — Spearman 殘差 vs 最大框高 **10:+0.08(≈0)、11:+0.28(弱)**,且 **10 的背景 tile(無框)殘差 1.15px ≥ 多數前景 tile**。**根因(終局)**:不是前景、不是人流、不是框高 —— **走路相機背景本身橫跨多深度,背景 tile 之間就差 ~1.15px(背景內視差),單一 2D similarity(連平移)都 fit 不了背景 flow → affine 額外 DOF 只放大這個固有 misfit**;misfit 在背景 tile,那裡沒框可排除。純平移 GMC 安全是因為只取一階共同位移、殘餘視差交給 per-track Kalman 速度態(同 #34「affine 上限不轉移」、#5「GT 上限不轉移」同型)。旋轉 flow 與深度無關(故 13 真旋轉贏)、平移/zoom flow 依賴深度(視差)→ 套到 off-plane track 必錯。**結構天花板 = 2D 影像 warp 無法表示 3D 視差**,門檻/前景/框高調不掉。**復活條件**:走 3D(per-track 深度 / 分層 homography),成本 >> 收益(13 那 41% 旋轉殘差不值)。code default off、`--gmc-mode` 預設 `gpu` bit-exact。腳本 `eval/gmc.py:TilePhaseCorrAffineGMC`;離線探針(前景占比分佈 + 視差–框高相關,ad-hoc) |
| 41 | **Horizon / depth prior（自驗證透視 prior）**（頂點 homothety 估地平線 → 下游 GMC 限旋轉 / §8.4 motion normalization；設計 `/tmp/horizon_depth_prior_design.md`） | 2026-06-16 | ❌ **NO-GO（訊號真實但無利可圖；估計 GO、下游全死）** | **方法 GO**：bbox 四對應頂點連線最小二乘交點（消失點）→ 跨對共線性擬地平線。GT 上限 **4/7 收斂**（02/05/09/13，判據 λ1/λ2≫36 區分真線 vs 團雲；04 淺透視團雲、10/11 移動失敗）；**撐過 detector 截斷**（真 tracker 框 Vy 與 GT 差 **0.00–0.02H**、無 GT✓→DET✗）。方法學：homothety 前提成立（concRMS~1px）、標準長寬 w=k·h 去寬度雜訊（GT 中性 concRMS→0）、子框不變性（top-70% Vy 不變）、配對 gate ratio≥1.30（drift 拐點實測：<30% 高度差 std(Vy)/H 0.11–0.45 飄，≥30% ≤0.10）。**下游全 NO-GO**：①**horizon-as-GMC 三形態全死** — (a)估相機運動：windowed frame-to-frame，移動機位 13 horizon 抖動 0.013H **= 靜態 02 的 0.013H**（win=10/30 皆同、自相關 0.71–0.80 是滑窗重疊假平滑非運動）→ MOT17 相機是 pan/平移（horizon 不敏感）非 pitch/roll；(b)逐幀選擇性 veto：已證 backfire（gmc.py:613 間歇 affine 時序不一致）；(c)一致 rotation clamp：**無標的** — 默認 GMC（`estimateAffinePartial2D`）旋轉實測 max **2.50°**、>3° 占 **0%**（RANSAC 已隱式 bound），clamp 改 0% 幀。**三方互證相機不 roll**：horizon 靜態 + GMC 輸出 max 2.5° + selective-veto backfire。②**§8.4 motion normalization NO-GO** — true/false 關聯判別 AUC：raw **0.917** ≥ /h(bbox 高度) **0.915**（平手、逐序列 4:3 分裂，僅 static-elevated 穩定 +0.03、移動機位全輸）> /dp(horizon depth_proxy) **0.887**（每序列墊底）；收緊 reach 2.0/3.0 模式一致。機制：**物件自身 h 就是最直接 per-object 尺度，horizon 序列全域常數繞道只加估計誤差+丟局部性 → 原理上不可能贏 h**；distance 在 5–30 幀 gap 已近天花板無正規化空間。**根因 = 訊號真實但無利可圖**（同 #35 consistency≠discriminability、#2/#32 appearance 牆、#34/#5「GT 上限不轉移」）：地平線估得準，但承載它的相機 DOF（pitch/roll）MOT17 幾乎不動 + 下游尺度有更優的本地代理(h)。**與 #40 互補**：#40 證 affine GMC 估計子本身結構性失敗（2D warp 無法表 3D 視差），#41 證即使 horizon 估得準也無 GMC 訊號可餵 + depth prior 下游無利。**復活條件**：有真 camera roll 的資料集（空拍/穿戴式）GMC 限旋轉才有標的；MOT17 上整方向關閉。探針保留 `scripts/tools/horizon_{convergence,homothety,detector,window}_probe.py`、`gmc_rotation_probe.py`、`motion_norm_probe.py`（全 GT/真框離線、可重現） |
| 42 | **Auction freshness bid（recency 偏置關聯）**（≤5f 原地震盪 A_respawn 修復：auction bid 加 `freshness_w/(1+age)` 讓 time_since_update 小的在任軌出價高，env `SACCADE_FRESHNESS_W` default 0） | 2026-06-17 | ❌ **NO-GO（機制歸因正確但全域不可分離）** | **動機+歸因（C++ assoc dump instrument）**：ID-switch gap 解剖（`diagnose_id_switches.py`）痛點在短距非長距：**≤5f 原地震盪 53%、長 gap 91+f 僅 4.5%**。≤5 桶拆解 = B_swap 交換 60%（瞬時因素全反向 acc≤20% → 無解,要 appearance/temporal）+ A_respawn 40%。A_respawn 機制查清（dump 每軌 預測框/候選/cost/assigned,幾何認 hold 因 hyp id≠live id）：①GMC 推歪 ❌否決（真實預測 IoU 0.82）②候選截斷 ❌否決（det_correct 在候選 88%、cost 0.237≪門 0.50）③✅真因 **auction 無 recency:mg=0 異常 85% 被 outbid,hold(age1 連續在任) 輸給 winner(age2 重接管陳舊軌),winner 較新鮮僅 7%**。**實作要點**：freshness 加在絕對 **bid** 非 val（val 偏移在 best−second margin 抵消、且多數 contest n_cands=1 margin=eps）。**掃描裁決**：w∈{.01,.03,.1,.3} **結果完全相同**（freshness=二元 tiebreaker,量級無關、無 sweet spot）；vs baseline（IDF1 75.9/AssA 66.4/IDs 484/FP 3033/FN 21203）→ **IDF1 74.9/AssA 64.5/IDs 468/FP 2828/FN 21588**：✅IDs −16/FP −205 但 ❌**FN +385/AssA −1.9/IDF1 −1.0 淨負**。**根因**：「93% 在任者對」是 switch 子集（本就出錯那群）量的,bias 套全部 contested det 否決了大量「陳舊軌正確重接管」→ FN 暴增,且二元 tiebreaker 連調小都做不到（同 occ cw1.0 #?、#21 vel_dir、訊號層歸因「前提真+全域有害」同型）。**復活條件**：需「分離 incumbent-right vs reacquirer-right」的額外識別訊號（繞回 identity/ReID 牆）。code env-gated default off：`SACCADE_FRESHNESS_W`（freshness bid）+ `SACCADE_ASSOC_DUMP`（診斷 dump），`tracker_gpu.cu:parallel_auction_shmem_kernel` |
| 43 | **Auction stability bid + Mahalanobis-as-cost**（per-track bid 加框高匹配 `stability_w/(1+dh_rel)` env `SACCADE_STABILITY_W`；及把 Kalman 協方差 Mahalanobis 從 gate 升成 cost 排序項，分辨交叉處 IoU 簡併候選） | 2026-06-17 | ❌ **NO-GO（二元 tiebreaker + 運動層不可分離）** | **stability bid**：sweep `{0.05,0.1,0.2,0.4,0.8}` **結果完全相同 = 二元 tiebreaker**（量級無關，同 #42 freshness — bid 只需贏對手任意一點，排序由 dh_rel 定）。on(0.1) vs off：IDF1/HOTA/MOTA 全平、**IDs −42 但 FP +568/AssA −0.2** = 橫向交換無主指標增益；**FP +568 100% 經插值產生**（interp off 時 ΔFP=−7，見 #44）。`history_w`/`hit_streak` 接到 kernel 但**從未使用=死碼**。**Mahalanobis-as-cost**（S_inv dump oracle）：現行 cost=`1−IoU(Kalman 預測框, det)`，Kalman **均值框**有用、**協方差 S_inv 只當 admit gate 不進排序**。升成 cost 後 multi-cand 選對率 **Maha 63% < IoU 66%**，IoU 簡併集（top-2 cost gap<0.05、swap 高發區）**Maha 47% < IoU 54%**，IoU 錯時 Maha 只救 24%。根因：交叉處兩候選 innovation 同落不確定橢圓→協方差無判別力，且 Maha 把 aspect/h 噪聲算入比 IoU 更吵。**≤5f swap 在整個運動層（GMC 相機運動 + Kalman 均值 + 協方差）全不可分**，需 appearance/identity（ReID 牆 #2/#32/#35）。code：`SACCADE_STABILITY_W` default 0.1（**建議改 0**，偏離 stability-off 基線無增益）、`SACCADE_HISTORY_W`+`hit_streak`（死碼建議移除） |
| 44 | **插值 FP 降低**（interpolation FP 佔總 FP ~70%；試 GMC-aware 插值 / Kalman 端點速度 Hermite / 二次三次 Bézier 窗口擬合 / endpoint-score gate） | 2026-06-17 | ❌ **NO-GO（插值 FP 不可約；病根在偵測/關聯非插值幾何）** | **動機**：`--no-interpolate-tracklets` 對照 → 插值加 **~2096 FP**（3033→937）但回收 **~4000 FN**（淨 **+0.8 IDF1/+1.9 MOTA/−177 IDs**，**不可關**）。**逐填補框分類**（ip_on vs ip_off vs GT，6038 fills）：78% TP/22% FP；FP 集中**小框×移動相機**（13:47%、10:36% vs 02/04/09:0-6%）。**端點橋接歸因（決定性）**：correct_bridge（兩端同 GT）**1% FP**、WRONG_bridge（兩端不同 GT=ID switch）24%=總FP **19%**、unmatched_endpoint（端點不對任何 GT=端點本身 FP 偵測）66%=總FP **78%**。正確 gap 的 **GT 路徑近線性**（沿軌偏離中位 **1.7%**h、橫軌 **1.6%**h、p90~6%、74% 平滑單側弧）→ **軌跡/速度不是病因**。**五修法全 NO-GO**：①`max_gap` 縮短 = FP↔FN 此消彼長（35 已 IDF1/MOTA 最優，50 同）；②**GMC-aware 插值**（用累積 GMC 彎曲路徑，env `SACCADE_GMC_INTERP`，**已 revert**）：修正中位 2.7px ≪ 偏差 ~35px，FP −2/IDF1 −0.1（線性已含淨相機位移，只差非線性殘差）；③**Kalman 端點速度 Hermite** oracle 淨 **−72**（端點單幀速度=噪聲、外推 overshoot 破壞 220 TP > 救 148 FP）；④**Bézier 窗口擬合** oracle：三次 −68（overshoot，≡Hermite）、二次 +35（0.6%=噪聲、救 198/破 163≈擲硬幣）；⑤**endpoint-score gate**：unmatched 端點分數中位 **0.164** vs correct **0.458**（均值分開），但 per-fill 門檻砍 **1.5–1.8× 更多 TP**（correct 有 25% 低分、unmatched 有 34% TP，分佈重疊，同框高 gate）。**根因**：插值 FP = **端點本身 FP 偵測（78%）+ 錯橋（19%）**，插值只放大；只能上游修（detection FP / association），各自撞牆（ReID #2/#32/#35、bridge geometry AUC 0.55 `project_bidir_relink_analysis`）。**附帶修復 Bug #1**：`gmc_kernel.cu:launch_phase_correlation` 的 25% 位移 cap 套在**未解繞**原始峰值索引上→所有負位移被誤拒（diff 新引入），改**先解繞再判**對齊 `peak_to_translation_warp_kernel`（eval 走 warp 路徑故 MOT 指標不變，修的是 `estimate()` API 正確性）。腳本 ad-hoc（ip_on/off 比對 + GT 分類 + 沿/橫軌分解 + 各 oracle） |
| 45 | **fuse_score_weight 提高**（mamba_whole_graph preset default 0.0 → 0.1：把偵測分數灌進關聯成本 `(1−w)·iou + w·(1−score)`，BoT-SORT 式低分懲罰；曾在舊 `baseline` preset 0.4 有效 FP −12%/MOTA +1.6） | 2026-06-19 | ❌ **NO-GO（聚合負 + 四探針一致證偽任何可分訊號）** | **聚合**（7-seq SDP，paired vs fw0.0）：IDF1 78.2→77.9 / MOTA 78.4→78.1 / HOTA 70.2→70.0 / AssA 69.7→69.5 / DetA 70.9→70.6 全退；FP −104（唯一贏）被 FN +338/IDs +31 蓋過。**逐序列雙峰**：GO=09（+2.0）/10（+2.1），NOGO=11（−1.8）/13（−3.1），其餘中性——聚合 −0.3 蓋掉真實分裂。**四個正交探針全部無法分離 10(GO) vs 13(NOGO)**（皆跑 fw0.0 output txt + GT，分鐘級可重現）：①低分桶 [0.075,0.20) **ghost-rate** 10:47% ≈ 13:51%；②**ghost 來源分解** REAL_BADBOX（真人壞框 IoU∈[0.1,0.5)）10:**87%** vs 13:52%——證偽「10 靠抑制背景幻覺」（背景僅 13%）；③**冗餘度**（REAL_BADBOX 所屬 GT 是否另有高分框備援）%unique 10:60% ≈ 13:57%——證偽「unique-evidence 假說」；④**相機運動**（ORB+RANSAC 全域逐幀位移）10:**4.42px/f（全場最快）** vs 13:3.89——**證偽「13 快 / 10 慢」運動歸因**（最快的 10 反而 GO；靜態 09 也 GO）。**根因**：兩條 GO 機制互相矛盾（09 靜態·ghost≈0·純 tie-break；10 最快·ghost-heavy·去碎片框），NOGO 線在每條軸上都落在它們中間 → fw 的效果不在偵測框可分的任何維度（score/ghost-source/redundancy/motion），是序列特異多因子混淆,不可約成單一觸發訊號。redundancy-gated / motion-gated fw 兩個 conditional 候選均被自家探針事前否決（同 #38 ntt0.20 per-seq over-fit、#42/#43 全域不可分離、`project_mamba_score_distribution` distribution-overlap 牆）。**caveat**：探針跑追蹤後 output 框（代理），嚴格版需 dump 關聯當下原始偵測;但四路代理一致重疊,乾淨訊號存在機率已極低。preset 維持 fw=0.0。腳本 `probe_ghost_rate_by_score.py`/`probe_ghost_source.py`/`probe_redundancy.py`/`probe_camera_motion.py`（皆 output txt + GT 離線、可重現） |
| 46 | **Head 激活遮擋訊號（visibility head）**（Mamba head 是否「看出」被遮框 → 餵 occlusion-aware association；起於「relink/occ 缺乾淨遮擋源頭、靠其他訊號猜」的觀察） | 2026-06-19 | ❌ **NO-GO（訊號真實但 4 應用全無可獲利接口；同 #41「訊號真實+下游有更優本地代理」型）** | **訊號層（真實存在）**：head 架構無 visibility 輸出通道（只 cls+reg+預設關 emb）。**score 對遮擋盲** AUC 0.559（單尺度 P3/P4/P5 0.42/0.51/0.51、User 的 **P3−P5 差值方向對但弱 0.567**、3 尺度 probe 0.563）；但 **768-D `x_cls` 激活 linear probe 0.836**（GroupKFold by identity 防身份洩漏、shuffle 0.497），**迴歸掉框幾何後 leakage-free per-fold 仍 0.793**（> 幾何基線 0.670）→ 帶**非幾何 appearance 遮擋線索**。機制：head 內部學到遮擋但 **1×1 cls conv 投影掉**故 score 盲（解釋 `mamba-score-distribution`「crowd→FN 非壓分」）。**4 應用全測 NO-GO**：①**relink 分真假橋**（join `relink_candidates.csv` gt_match + GT visibility）oracle 遮擋 AUC **0.56–0.60 < 同池幾何 0.83**，歸因=「被遮」是候選池常態非判別量（軌跡被遮消失→旁邊另一被遮者成假候選、真假橋端點都被遮，true 0.278 vs false 0.337 重疊巨大）；與 #33 occ_cover 互補→遮擋訊號兩種接法全死。②**OAO/occ 缺乾淨源頭 = 證偽**（`occ_event_values.py` GT crossing 187 events）幾何 foot-y 前後判定 vs visibility-truth 吻合 **operating gate 100%、同深度 foot_gap≤0.10 仍 92%**，激活 0.79≈75% acc **比幾何降級**；crossing-swap 真瓶頸=分開後 identity 錯接（ReID 牆 #2/#32/#35）非源頭。③**crossing-swap state-consistency 懲罰**（User「ID(遮擋)→ID(可見) 懲罰」；`probe_occ_swap_disambiguation.py` N=20 occluder-ABSORB）機制正確（占 63% 失效、72% ABSORB、79% 兩框可救）但 oracle occlusion 85% **只贏 foot-line 80% 共 5pp**，foot-AMBIG 子集 occ 75% vs foot 67%（+8pp 但僅 35% 事件、真 head 75% < foot 80% 是降級）；真瓶頸=auction 無 recency（#42）非訊號。occ outcome 天花板 +4.1 IDF1（`oracle_occlusion_hold.py`，但=identity oracle≠visibility）。④**low-IoU gate relaxation = 反向 NO-GO**（User「低 IoU 但 OCC 豁免」；`analyze_assoc_fn.py` 量 prize + `probe_lowiou_occ_gate.py` n=167k 候選對）prize 小（mid_break 僅 14.4% 的 4187 assoc-FN、其中只 30% 重度遮擋，大多快速運動/GMC；遮擋主導的 lost_reappear=relink 已死）；**Part B 反向**：低 IoU band [.05,.50) 遮擋 AUC **0.414 < 0.5（反指標）**、very-low 0.402、base rate 2.86%/0.65%、IoU 自己 0.88/0.74 仍最佳——低 IoU 的**真**配對多是快速運動（可見人移動）、**被遮**低 IoU 偵測多是人群雜訊/別人→遮擋指錯方向（比 relink 0.56 中性更糟）。**根因 = 訊號真實但無利可圖**（同 #41 horizon、#35 consistency≠discriminability、#34/#5「上限不轉移」）：MOT17 的幾何本地代理（foot-line 92–100%、IoU 0.88）已吃掉 appearance-based 遮擋訊號的全部空間，且想拿遮擋當判別量處它在候選池太普遍/反向。**不建議訓 visibility head**。**復活條件**：幾何代理失效的場景（真 3D / 相機 roll / 嚴重 ReID 依賴資料集）。腳本 `probe_occ_separability.py`/`probe_occ_activation_separability.py`(+npz raw dump)/`probe_relink_occlusion_signal.py`/`probe_occ_swap_disambiguation.py`/`probe_lowiou_occ_gate.py`、重用 `occ_event_values.py`/`analyze_assoc_fn.py`/`oracle_occlusion_hold.py`（全 detector+GT/output txt 離線、可重現） |

## NO-GO 的結構性根因

1. **Appearance 天花板**：MOT17 身份在 embedding 空間本質難分 — 5 個模型 + 4 種機制 + SR + 域訓練全撞同一個上限。這是物理瓶頸，非演算法缺陷。

2. **GMC 壓倒性主導**：GMC ON 後 IDF1 +2.8pp、IDs −133，是唯一顯著貢獻模組。其他模組在 GMC 開啟後基本冗余（∆ <0.4pp）。

3. **「密集 = FP 多」假設錯誤**：MOT17 中高密度場景是真實人多，非 FP。以 density 為信號的 filtering 策略必然傷 recall。

4. **時序資訊難進特徵層**：Mamba temporal block（v15/v17）、per-channel SSM A 全部退步。R1→R2 grad 崩潰無法收斂。

5. **Relink gate 的訊號天花板**：幾何/運動殘差對「真 vs 假橋接」AUC≈0.55（近隨機），外觀 gate AUC≈0.50；scale/occ/appearance gate 一律死在門作用區或被 `track_buffer=30` 結構性消滅。長 gap（80+）目前無單一可靠訊號 — 唯一已驗證正向是 GPU 雙向橋接本身（見 GO 表）。

---

## 中性 NO-GO 的復活前例

**Relink 系列證明：中性 ≠ 訊號不存在。** 軌跡如下：

1. Motion relink（#6）、Semantic relink（#3）單獨測試中性 → 根因是 **age gate 攔掉 86–89% 候選**，運動訊號根本沒機會作用。
2. 雙向中點橋接改變候選生成（farewell archive + 中點外推，繞過 age gate 結構）→ 同樣的運動殘差訊號變成 **IDF1 +2.1 / AssA +2.8 全指標勝利**（見 GO 表）。
3. Scale gate（#31）單獨無速度增益 → 組合進 bridge（px=0.25 + scale gate）成為 preset default 的一部分。

**OAO（#7）證明：復活條件可以是「機制重設計」，不只是 blocker 移除。** 06-12 登記的復活條件就是「occ-conditioned 機制重設計（嚴格 gate / 出生延遲），非 cost 偏移」。06-17 兌現：先確認新 baseline 下 plain OAO 已翻盤（+1.6 IDF1，但 04 撐 88% AssA、05 −3.4），再逐一證偽 6 個**空間**判別信號（05 害與 10/13 益在每個空間軸都纏結），最後用**時間軸**（重疊持續幀數）破牆——05 短暫交叉(~10f)被 damp、04 持久人海(~49f)爬滿 → Pareto 支配 plain。教訓：**先量「增益是否集中在單一序列」（ex-04 拆解），再找「害與益分離的軸」——空間試遍才換維度**。

**結論**：中性結果登記時必須附 blocker 歸因；blocker 被後續工作移除時，回來重測。

### 已識別 blocker 的中性項目（⚪ 候補復活名單）

> 僅收錄有歸因實驗數據者；blocker 欄的數字即出處實驗的統計量。

| # | 項目 | Blocker（實測） | 復活條件 |
|---|------|---------|----------|
| 3 | Semantic Relink | age gate 拒 86.8% 候選 | 候選生成結構改變（已部分由 bidir bridge 實現） |
| 6 | Motion-based Relinking | age gate 攔 89% 候選 | 同上（已由 bidir bridge 復活） |
| 7 | OA-SORT OAO | 整列加 cost 不改排序（舊）；new whole-graph baseline 下空間判別信號 6/6 不可分離 05 害 vs 10/13 益 | **已復活（06-17）**：duration-ramp（時間軸 = 唯一分離 05 短暫 / 04 持久的軸）。06-17 evidence：tau0.30+ramp25；目前 preset：tau0.50+ramp25 |
| 33 | occ_cover live relink | `track_buffer=30` 結構性消滅長 gap 族群；base rate 1.3% | bridge 專用過期檔案庫（不靠延長全局記憶 — tb90 已證 −0.8 IDF1） |
| 23 | Birth-time lost-bank relink | 長 gap rank-1 僅 13–33%（接近結構性，但混雜短 gap 易池） | 若有可靠長 gap 訊號源（外觀方向已結案，需新模態) |
| 8 | NSA-Kalman | 前提真（ρ=−0.52）但與 r_scale=2.8 雙重補償。**06-12 重校準已測**：f(score) v2 IDF1 +1.5/AssA +0.86 證實訊號可用，剩餘 blocker = 濾波態輸出的 localization 損失（DetA −1.44，集中移動相機 05/13）+ per-seq CI 含 0 | score-conditioned R 只作用於 association/gating、輸出箱改用量測（或 GMC 品質條件化啟用）；flags `--kalman-nsa-s0` 已 plumbed default off |
| 21 | vel_dir gate | fast AUC 0.751 被 46% 慢速噪聲樣本（AUC 0.526）淹沒 | speed-conditioned：僅 \|v\|>3px/f 啟用 |
| 7 | OA-SORT OAO | 前提真（AUC 0.727）但整列加 cost 不改變該 track 的 det 排序 | occ-conditioned 機制重設計（嚴格 gate / 出生延遲），非 cost 偏移 |
| 9 | PostMerge | combined AUC 0.868 vs base rate 2.4%（precision ~20%）；direction 分量純噪聲 | direction_weight 歸零 + 需正交訊號拉 precision |

> 2026-06-12 歸因方法與完整數據：[訊號層歸因分析](../research/eval/neutral_nogo_signal_attribution_20260612.md)。
> 共同模式：**前提成立 + 機制形式錯誤/失準** 是「中性→有害」的主要死法，而非訊號不存在。

---

## 對照：目前 GO / 穩定好用的模組

| 模組 | 狀態 | 貢獻 |
|------|------|------|
| **GPU GMC** (phase correlation) | ✅ default ON | IDF1 +2.8pp, IDs −133 |
| **Option F / Mamba Whole-Graph** | ✅ production preset | 現行 headline preset 為 `mamba_whole_graph`（native_640, ReID off）：IDF1 77.6 / MOTA 78.3 / HOTA 69.9 / AssA 69.1 / IDs 430 / 221.59 FPS |
| **GPUByteTracker + Sinkhorn-Auction** | ✅ default ON | 關聯延遲 0.67ms (10x 提升) |
| **Async ReID** | ✅ legacy speed feature | 舊 pipeline speed optimization；現行 headline baseline ReID off |
| **Pipeline Relink** | ✅ legacy speed feature | 舊 pipeline speed optimization；accuracy headline 由 GPU 雙向橋接 relink / tracker path 決定 |
| **GPU 雙向橋接 Relink** (px=0.25 + scale gate) | ✅ preset default ON | IDF1 +2.1, AssA +2.8, IDs −13.6%, FP −14%（06-11 全指標嚴格優勢） |
| **FP Hard Filter** (area=40000) | ✅ default ON | FP 移除 9021, TP 移除僅 153 |
| **Kalman R Scale** (`kalman_r_scale=2.8`) | ✅ current preset | 見 `configs/presets/mamba_whole_graph.yaml` |
| **Detection Quality Scaling** | ✅ legacy / non-headline | 現行 `mamba_whole_graph` headline preset 為 `detection_quality_scaling=false` |
| **Interpolation** (max_gap=35) | ✅ default ON | — |
| **OA-SORT OAO duration-ramp** (current preset: tau=0.50, ramp_frames=25) | ✅ preset default ON | 06-17 evidence（tau0.30+ramp25）Pareto 支配 plain OAO：IDF1 77.5→77.6 / HOTA 69.7→69.9 / AssA 68.8→69.1 / ex-04 AssA +0.8 / MOT17-05 70.6→72.3；目前 config retune 為 tau0.50+ramp25。registry #7 復活 |
