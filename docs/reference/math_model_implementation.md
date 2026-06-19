# 數學模型實作指南

> 這是 [math_model.md](math_model.md) 的 companion guide。當你要修改 tracker、
> association、GMC、relink 或 lifecycle math 時，先看這份。這份文件刻意寫成
> operational checklist：要改哪裡、哪些 invariants 不能破、要怎麼驗證。

---

## 1. 先重建目前 Baseline

改任何 term 前，先從 source 重新確認 baseline：

1. 確認 arguments 與 presets 載入順序。
   - CLI entry：[scripts/eval/mot17.py](../../scripts/eval/mot17.py)
   - Preset：[configs/presets/mamba_whole_graph.yaml](../../configs/presets/mamba_whole_graph.yaml)
   - Parsed config object：[src/saccade/perception/eval/config.py](../../src/saccade/perception/eval/config.py)

2. 確認 eval path。
   - Main implementation：[src/saccade/perception/eval/evaluator.py](../../src/saccade/perception/eval/evaluator.py)
   - Shim only：[src/saccade/perception/eval/runner.py](../../src/saccade/perception/eval/runner.py)

3. 確認 tracker parameters 如何注入 C++。
   - `set_params(...)`
   - `set_oao_params(...)`
   - `set_occ_params(...)`
   - `set_multiplicative_cost(...)`
   - `set_sinkhorn_lambda(...)`
   - `set_stability_cost_w(...)`
   - `set_relink_params(...)`

4. 確認 active cost path。
   - `reid_mode: "off"` 代表 `stage1_cost_fused_kernel(...)` 是 headline path。
   - ReID/semantic kernels 仍存在，但不是 baseline 主線。

5. 確認 output 行為。
   - `materialize` 是 tracker output boundary。
   - `relink_write` 仍是 stage name，即使主要 identity work 是 tracker-core
     bridge relink。

---

## 2. 新增或修改 Association Term

association terms 主要在 [src/tracking/tracker_gpu.cu](../../src/tracking/tracker_gpu.cu)。
多數變更應同時處理兩個 cost kernels：

- `stage1_cost_fused_kernel(...)`：no-ReID fast path，也是目前 baseline。
- `compute_conditional_cost_kernel(...)`：appearance-aware path。

### 2.1 先決定 Term 類型

若 match 本身不合法，用 gate：

```text
if condition fails:
    c_ij = 1
    return
```

若 candidate 仍可接受、但可信度較低，用 positive penalty：

$$
\Pi_{ij} \mathrel{+}= \beta \cdot s_{\mathrm{norm}}(i,j)
$$

若 signal 確實代表穩定性，且有明確上界，才用 reward：

$$
\Pi_{ij} \mathrel{-}= R_{\mathrm{bounded}}(i,j)
$$

目前 multiplicative cost：

$$
c_{ij} =
\mathrm{clamp}\left(1 - A_{ij}e^{-\Pi_{ij}}, 0, 1\right)
$$

不要加入 unbounded signals。geometry signal 進入 `Penalty` 前，應先依 `h`、
detection height、frame size 或已驗證的 scale normalize。

### 2.2 保持 Candidate Sparsity

tracker 效能與穩定性依賴 compact candidates：

$$
\mathrm{candidate}_{ij} \iff
\mathrm{IoU}_{ij} > \tau_{\mathrm{iou}}
\;\lor\;
d^2_{ij} < \tau_{\mathrm{maha}}
$$

$$
\mathrm{enqueue}_{ij} \iff
c_{ij} \le
\max(c_{\mathrm{DDA}}, \tau_{\mathrm{match}}, \tau_{\mathrm{stage2}})
$$

若新 term 常把 Mahalanobis-only、幾乎不重疊的 boxes cost 降低，就可能塞滿
`K_MAX_CANDIDATES`，讓真正低成本 match 被擠掉。擁擠 sequence 特別要看
candidate counts 與 association regression。

### 2.3 保持 Cost Range

任何寫入 `cost_matrix` 或 `cand_costs` 的 path 都應維持：

$$
0 \le c_{ij} \le 1
$$

multiplicative path 在 `1 - Q * exp(-Penalty)` 後 clamp。若新增很大的負 reward，
要確認它不會讓大量 cost 被 clamp 到 0，否則會抹平原本有意義的差異。

### 2.4 更新 Config Plumbing

新增參數時，逐層接好：

| Layer | File |
|:--|:--|
| CLI/module default | [scripts/eval/config](../../scripts/eval/config) |
| Parsed field | [src/saccade/perception/eval/config.py](../../src/saccade/perception/eval/config.py) |
| baseline 需要時才改 preset | [configs/presets/mamba_whole_graph.yaml](../../configs/presets/mamba_whole_graph.yaml) |
| Python tracker wrapper | [src/saccade/perception/tracking/tracker_gpu.py](../../src/saccade/perception/tracking/tracker_gpu.py) |
| C++ tracker API | [include/tracking/tracker_gpu.hpp](../../include/tracking/tracker_gpu.hpp) |
| C++ setter/state | [src/tracking/tracker_gpu.cu](../../src/tracking/tracker_gpu.cu) |
| Kernel launch arguments | [src/tracking/tracker_gpu.cu](../../src/tracking/tracker_gpu.cu) |

如果 term 是 optional，default 應是 off，或完全等價於既有行為。

---

## 3. 修改 OAO 或 Occlusion Logic

OAO 有兩層意義：

1. `occ_coeff_i`：估計 track `i` 有多 occluded/confusable。
2. `P_oao(i,j)`：把 coefficient 轉成 association penalty。

coefficient 由 `compute_track_occlusion_kernel(...)` 計算。

Max-overlap mode：

$$
o^{\mathrm{base}}_i =
\max_{k \ne i,\; k\ \mathrm{active}}
\mathrm{IoU}(B(x_i), B(x_k))
$$

Union mode：

$$
o^{\mathrm{base}}_i =
\frac{\#\{\mathrm{covered\ cells\ in\ an\ }8\times8\mathrm{\ raster}\}}{64}
$$

Duration ramp：

$$
o_i =
o^{\mathrm{base}}_i
\min\left(1, \frac{n^{\mathrm{overlap}}_i}{N_{\mathrm{ramp}}}\right)
$$

matching penalty：

$$
P_{\mathrm{OAO}}(i,j)
= \tau_{\mathrm{OAO}}\,o_i\,g_s(s_j)
$$

實作注意：

- `occ_partner_all` 要和 depth-gated/front-occluder state 分開。它是給
  contention-gated OAO 判斷 strongest overlapping partner 用的。
- duration ramp 是 per-track counter。track inactive 或不再 overlap 時要 reset。
- 若新增 crowd/height/foot gates，先定義清楚它只影響 `occ_coeff`，還是也影響
  front-occluder state。混在一起會改變語義。

建議先跑：

```bash
PYTHONPATH=. uv run pytest tests/unit/tracking/test_auction_simple.py tests/unit/tracking/test_auction_real_scale.py -q
```

再跑至少一條 MOT17 sequence profile：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --sequences MOT17-02-SDP --profile-stages
```

---

## 4. 修改 Sinkhorn/Auction Behavior

目前 path 是 sparse top-k plus auction，不是 dense assignment solve。相關 kernels：

- `fused_sinkhorn_multistage_kernel(...)`
- `parallel_auction_shmem_kernel(...)`
- `commit_auction_results_kernel(...)`

stage value：

$$
p_{ij} = e^{-\lambda c_{ij}}G_{\mathrm{aspect}}(b_j)
$$

auction value：

$$
v_{ij} = p_{ij} - \rho_j
$$

auction bid：

$$
\mathrm{bid}_i =
\rho_{j^*} +
\left(v_{ij^*} - v_i^{(2)} + \epsilon\right)
$$

若沒有 second candidate，increment special-case 為 `epsilon`。

實作注意：

- 修改 `sinkhorn_lambda` 會改變小成本差異的影響力。baseline 刻意用 `10`，
  不是舊的 sharper `30`。
- 若在 cost space 加 reward，考慮是否要像 `stability_cost_w` 一樣用 `lambda`
  normalize。
- 不要在 bidding kernel 直接寫 assignment pair。目前設計是所有 block winners
  完成後再 commit，避免 `trk_to_det` / `det_to_trk` 不一致。
- 保持 tie-breaking deterministic。packed key 使用 bid bits 加 track/cand ordering。

建議測：

```bash
PYTHONPATH=build:src UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit/tracking/test_gpu_bidir_bridge.py tests/unit/tracking/test_auction_simple.py -q
```

---

## 5. 修改 Kalman 或 Mahalanobis Gating

Kalman math 集中在 [include/tracking/kalman_gpu.cuh](../../include/tracking/kalman_gpu.cuh)。

state order 必須保持：

```text
(cx, cy, a, h, vx, vy, va, vh)
```

若修改 process noise 或 measurement noise：

- 保持 `Q` 與 `R` positive。
- 保持 4D measurement shape `(cx, cy, a, h)`。
- 檢查 `compute_S_inv(...)` 的 callers；目前它計算 top-left 4x4 covariance block
  加 `R` 後的 inverse。
- 如果加入 score-adaptive 或 lighting-adaptive noise，確認傳入 `get_R(...)` 的值
  在 update 與 gating 中一致使用。

baseline 重要參數：

```text
kalman_r_scale = 2.8
```

較低 `R` 代表更信 detection；較高 `R` 代表更信 prediction。Mahalanobis gating
與 update 都依賴 uncertainty model，所以改 `R` 會同時影響 candidate admission
與 state smoothing。

---

## 6. 修改 GMC

GPU phase correlation 在 [src/tracking/gmc_kernel.cu](../../src/tracking/gmc_kernel.cu)。

active path（現行啟用路徑）：

```text
CHW RGB -> grayscale downscale -> Hanning window -> FFT
prev/curr cross-power spectrum -> IFFT -> peak -> translation warp
```

修改 GMC 時：

- 保留 wraparound correction，且要在 displacement cap 前做。
- 除非 replacement model 有明確 reliability test，否則保留 25% dimension
  plausibility cap。
- 若改 PCR threshold 或 scaling，文件也要同步寫清楚。
- 加 host sync 前，先檢查 graph-capture paths（`estimate_into_direct`、
  whole-graph detector path）。

快速 sanity check：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --sequences MOT17-02-SDP --profile-stages
```

看 stage breakdown 裡的 `gmc_gray_downscale`、`gmc_phase_corr`、`gmc_handoff`。

---

## 7. 修改 Bridge Relink

tracker-core bridge relink 在 [src/tracking/tracker_gpu.cu](../../src/tracking/tracker_gpu.cu)。
appearance/semantic relink gate 是另一條 path：
[src/tracking/relink_gate.cu](../../src/tracking/relink_gate.cu)。

baseline bridge 參數：

```text
relink_bridge_enabled = true
relink_bridge_px = 0.25
relink_bridge_margin = 0.05
relink_bridge_h_lo = 0.75
relink_bridge_h_hi = 1.33
relink_bridge_dir_bonus = 0.8
```

`relink_bridge_px` 是歷史命名；baseline 的 `0.25` 實際上是
reference-height-normalized distance threshold，不是 pixel threshold。

bridge flow：

```text
update foot history for observed slots
candidate with hit_streak == bridge_at scans live lost confirmed slots
compute normalized bridge distance
apply height/speed/spatial/occupancy/margin gates
winning candidate adopts lost id
```

實作注意：

- foot history 存 `(cx, cy, h)`，但 foot anchor 在 foot mode 下用
  `cy + 0.5*h`。公式中要明確寫 center 還是 foot。
- 4-point regression velocity 是 closed-form：

$$
v = \frac{3p_3+p_2-p_1-3p_0}{10}
$$

- `bridge_margin` 比較 best 與 second-best distances。越小越寬鬆，越大越會拒絕
  ambiguous relink。
- commit stage 會在 id adoption 後 deactivate lost slot。不要留下兩個 active
  slots 共用同一 id。

建議測：

```bash
PYTHONPATH=build:src UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit/tracking/test_gpu_bidir_bridge.py -q
```

---

## 8. 修改 Semantic Relink

目前 baseline 關閉 semantic relink；若要啟用，請同時檢查 Python decision layer
與 GPU gate/scoring kernels：

- [src/saccade/perception/eval/relink.py](../../src/saccade/perception/eval/relink.py)
- [src/tracking/relink_gate.cu](../../src/tracking/relink_gate.cu)
- [include/tracking/relink_gate.hpp](../../include/tracking/relink_gate.hpp)

GPU gate columns：

```text
0 kalman_d2
1 bridge_dist
2 center_norm
3 iou
4 speed_exceeds
5 dir_behind
```

GPU gate 刻意只輸出 raw quantities；Python side 保留 frame ordering、
alias/split-collision checks 與部分 margin behavior。若把 decision 從 Python 移到
GPU，先用固定 candidates 驗 bit-faithfulness，再看 aggregate MOT metrics。

---

## 9. Tests And Metrics

local correctness 先用 fast unit tests，再跑 MOT sequence 看 behavior。

快速 checks：

```bash
PYTHONPATH=. uv run pytest tests/unit/eval/test_core.py::test_tracker_update_basic tests/unit/tracking/test_gpu_bidir_bridge.py -q
```

auction/cost checks：

```bash
PYTHONPATH=. uv run pytest tests/unit/tracking/test_auction_simple.py tests/unit/tracking/test_auction_real_scale.py -q
```

單 sequence profiling：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --sequences MOT17-02-SDP --profile-stages
```

完整 detector-family aggregate：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --output results/<name>
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/eval/calculate_mota.py --results results/<name>
```

模型變更至少回報：

- IDF1
- HOTA
- AssA
- MOTA
- IDs
- FP/FN
- per-sequence deltas，不只 aggregate
- 若碰到 GPU kernels 或 synchronization，也要回報 latency/stage changes

---

## 10. Documentation Checklist

模型行為改變時，同步更新：

| 變更內容 | 要更新 |
|:--|:--|
| Stage order 或 source map | [pipeline_flow.md](pipeline_flow.md), [DATAFLOW.md](../DATAFLOW.md) |
| Baseline preset values（目前推薦設定） | [mot17_default_config.md](mot17_default_config.md), [math_model.md](math_model.md) |
| Association equations | [math_model.md](math_model.md), 本指南 |
| Experiment result 或 no-go | 適用時更新 [no_go_registry.md](no_go_registry.md) |
| Config flags | 對應的 `scripts/eval/config/*.py` help text |

commit docs 前，至少跑：

```bash
python scripts/tools/check_doc_links.py
```

如果新增很多 code-span paths，也要做 path sanity check。
