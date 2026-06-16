# GMC Box-Residual 共模修正 — NO-GO(含完整歸因鏈)

**日期**:2026-06-12
**Worktree**:`exp/nogo-neutral-rescreen`(main@85a4564a)
**前置**:[Kalman 重校準](kalman_h_recalibration_20260612.md)的 GMC regime 討論 +
LK affine 探針(memory `project_gmc_affine_probe`)

## 假說與動機

GT 上限診斷顯示:每幀對 GT 軌跡位移移除 affine 共模,比只移除平移共模再降
殘差 19–41%(13: 41%、11: 33%;靜態 02/04 僅 5–9% ≈ 過擬地板)。
⇒ 假說:association 後 matched (track, det) innovation 的共模分量 = 像素
GMC 殘差(含未建模旋轉),robust 擬合後套回全部 active track(lost 最受益)
可免 FFT 修正 GMC 誤差。

## 實作(保留,default off)

`gmc_residual_correction_kernel`(tracker_gpu.cu,association 後、Kalman
update 前;單 block,thread-0 串行收集+擬合保 bit-exact 重跑確定性):
median 錨點(counting selection)→ ±trim_px 內點圈 → trimmed-mean 平移 /
centered affine LS(2×2 normal eq,spread guard 50px std)→ cap(|t|≤10px,
|A|≤0.02)→ 套用 position + velocity 旋轉。入口:confirmed + score≥0.5,
min_pairs=8。CLI:`--gmc-rc-mode {0..4}` + min-pairs/trim-px/cap-px。
`set_gmc_residual_params` 獨立 setter。CUDA-graph 相容。
Default-off MOT17-04 bit-exact 已驗;`test_gmc_residual_correction.py` 4 tests。

## 結果(MOT17-SDP,mamba_whole_graph,A=75.1/IDs 482/AssA 66.65)

| 臂 | IDF1 | IDs | AssA | 備註 |
|---|---|---|---|---|
| RC1 平移、全 track | 74.3 | **598** | 65.89 | 10: −4.2、13: IDs +79 |
| RC2 affine、全 track | 73.0 | 586 | 64.16 | 10: −8.2;affine 在 innovation 噪聲上過擬 |
| RC1 cap 2px(劑量) | 74.8 | 540 | 66.5 | 傷害 ∝ 施加幅度,單調 |
| RC3 平移、lost-only | 74.7 | 508 | 66.2 | 10: −2.6、02: −0.6;仍無任一 seq 轉正 |
| RC4 affine、lost-only | 74.2 | 507 | 65.5 | — |

小序列(05/09)matched pairs 常 < min_pairs,從未啟動(輸出 bit-same);
啟動最多的 10/13 傷最重 — 與意圖完全相反。

## 歸因(三個實驗)

1. **劑量反應**(cap 10→2px):傷害單調縮小但不過零 → 傷害來自施加的修正
   本身,非實作 bug。
2. **範圍切除**(全 track→lost-only):傷害再縮小(IDs 598→508)→
   matched track 的「雙重計算」(innovation 含 Kalman 穩態滯後,update 步
   本來就要吸收)是傷害的一部分 — 但**不是全部**。
3. **lost-only 仍負** → 根本死因:**GT 上限不轉移到 innovation 空間**。
   Kalman 速度態已吸收持續性相機運動殘差,逐幀 innovation 共模剩下的是
   滯後 + 人群共模 + 配對噪聲,可修正的相機訊號占比過低。GT 位移共模
   (相對靜止座標)與 tracker innovation 共模(相對已自適應的預測)是
   不同的量 — 前者大不代表後者可用。

## 判決與復活條件

全模式 NO-GO,flags 保留 default off。復活條件:修正訊號必須來自
**獨立於 tracker 狀態的量測**(如 tile-based phase correlation 的子塊位移
→ affine),不能來自 innovation 自迴饋;tile-GMC 仍是旋轉殘差的唯一
未否證路線(LK 探針:13 +2.3 證明旋轉價值存在,但估計器須保 phase-corr
魯棒性)。

## 原料

- 臂輸出:`results/kalman_ablation/RC{1,2}_*、RC1_cap2、RC{3,4}_lostonly/`
- 日誌:`rescreen_logs/kalman_RC*.log`
- GT 上限診斷:per-frame 共模移除(平移 vs affine),見 memory
  `project_gmc_affine_probe`
