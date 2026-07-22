<!-- doc-status: draft -->
<!-- doc-promotion: none; D1 canonical model specification (seed); §2–§3 frozen at WP-A1 -->
<!-- doc-date: 2026-07-22 -->
<!-- doc-module: semantic -->

# Gap-conditioned stochastic transition model — canonical specification (D1, v1)

**用途：** GCTM 的 canonical model specification（charter 所定義的 D1）。本檔由
active charter 的 work packets 逐段填入；**已凍結段落只能經 append-only
correction 修改**。

**Authority 邊界：**

- Task/lifecycle owner：[GCTM task charter](../threads/gap_conditioned_stochastic_transition_model_task.md)
  （Issue [#175](https://github.com/raylei50653/saccade/issues/175)）。
- Production-object 描述來源：[existing-online object analysis](../../modules/semantic/research/existing_online_object_analysis_for_gctm_alignment_20260718.md)
  （supplemental；runtime authority 仍是 `src/tracking/tracker_gpu.cu`）。
- 本檔**不**做 bridge-runtime claim：無 H0 faithful capture、無 accepted
  runtime-fidelity edge；所有指向 production 的欄位都只是 *declared
  correspondence target*，其 runtime 忠實性留待未來 H0/B1 路徑另行建立。
- 不授權 data、fitting、runtime、online、production 工作。

## §1 Document state

| Section | State | Frozen by |
|:--|:--|:--|
| §2 Canonical observation/time interface (nine fields) | **frozen** | WP-A1 (owner merge) |
| §3 Observation modes and causal availability | **frozen** | WP-A1 (owner merge) |
| §4 Canonical state, equations, admitted parameter domains | reserved — not yet specified | WP-A2 |
| §5 Identifiability and leakage matrix | reserved — not yet specified | WP-A2/WP-A3 |
| §6 Schema-only interface for a future B1 input | reserved — not yet specified | after WP-A5 gates |

Reserved sections carry no obligations-resolved claim; charter obligations 2–4
remain unresolved until their owning packets land.

## §2 Canonical observation/time interface（obligation 1 — nine fields, frozen）

Binding-class vocabulary（描述性使用 charter obligation 3 的 typed relations）：

- **normative** — A 層 canonical convention，由本檔定義；
- **documented-production** — 照抄 existing-online analysis 對 production
  operator 的描述；照抄即權威轉引，**不**構成 runtime-fidelity claim；
- **declared-target** — 指名未來 correspondence 的 production 對象；在 accepted
  fidelity edge 之前地位是 proxy/hypothesis。

| # | Field | Frozen definition | Binding class · source |
|:--|:--|:--|:--|
| 1 | `coordinate_substrate_id` | A 層 latent state 定義在 substrate-agnostic 的 \(\mathbb R^k\)。宣告的 production observation projection 目標為 **`S_A`**：active CUDA bridge 的 native geometry substrate（image-plane position，經 \(h_{\mathrm{ref}}=\max\big(\tfrac{h_L^{\mathrm{ema}}+h_C^{\mathrm{ema}}}{2},1\big)\) 高度正規化；Step-0 substrate audit rev.3 的 A 支）。 | declared-target · substrate audit + existing-online §4.1 |
| 2 | `frame_time_unit` | Canonical 時間單位 = **1 frame interval**（discrete frame index）。所有 gap/horizon 量以 frames 計；A 層不綁 wall-clock/fps；秒級換算須另行宣告映射。 | normative |
| 3 | `physical_gap_definition` | \(g_{\mathrm{phys}}\) := **physical inter-endpoint gap** = lost track 最後被觀測 frame（exit endpoint）到 candidate 第一次被觀測 frame（entry endpoint）之間的 frame-interval 數。以 production 量表示：\(g_{\mathrm{phys}} = \mathrm{gap\_len} = \mathrm{la} - \mathrm{bridge\_at} + 1\)。`gap_len` 在 production 只用於 occupancy/fidelity 路徑，**不是** score-extrapolation factor。 | documented-production · existing-online §3 |
| 4 | `online_horizon_definition` | \(\Delta_{\mathrm{on}}\) := \(\mathrm{la} = \mathrm{age[lost]} \in \mathbb N\)，唯一 production score-extrapolation horizon（進 lost forward / candidate backward extrapolation、speed-gate time、directional blend gap scale）。 | documented-production · existing-online §3（boxed authority）+ §9.5 |
| 5 | `g_phys_to_delta_on_mapping` | Exact identity：\(\boxed{\Delta_{\mathrm{on}} = g_{\mathrm{phys}} + (\mathrm{bridge\_at} - 1)}\)。GCTM 擁有此映射；**兩者永不得默認相等**。Canonical A 層 transition index：\(K_\Delta\) 的 \(\Delta\) = 被建模 transition 兩個 anchor time 之間的 physical elapsed frames；bridge pair 的 canonical anchors 為 **exit endpoint → entry endpoint**，故 canonical \(\Delta = g_{\mathrm{phys}}\)（entry observation \(y_1\)、其不確定度 \(R_1\) 都活在 entry endpoint）。Production 的 fire-frame 比較（anchor 在 \(t_{\mathrm{exit}}+\Delta_{\mathrm{on}}\)）是 **derived construction**，任何 production-corresponding instantiation 必須宣告其 anchor pair 並套用上式。 | normative（mapping 本身）；anchor 事實為 documented-production |
| 6 | `bridge_at_convention` | `bridge_at` = candidate hit-streak 觸發門檻：bridge event 在 `hit_streak == bridge_at` 時 fire；fire 時 candidate 已連續被觀測 `bridge_at` frames，故 entry endpoint = fire frame − (`bridge_at` − 1)。`bridge_at` 是 runtime-configured 值（當前 production default 4）；所有 identity 使用 event 當下綁定的值，**不**凍結常數。 | documented-production · runtime binding `tracker_gpu_python.cpp` / `assoc_basis.py`（trigger timing） |
| 7 | `continuous_dt_conversion` | Canonical \(dt \equiv 1\) frame interval。連續時間 SDE（M1/M2）參數以 per-frame-unit 表示；離散 transition = 在 \([0,\Delta]\)（\(\Delta\in\mathbb N\) frames）上的 exact integration。物理秒需另行宣告 fps 映射，不在 A 層範圍。 | normative |
| 8 | `production_cv_null_offset` | Null model（exact CV motion、matched identity、exact velocity）下，active production operator 的外推殘差在兩個 anchor 均為零：\(p_L^{\rightarrow}(\Delta_{\mathrm{on}})-p_C=0\)、\(p_C^{\leftarrow}(\Delta_{\mathrm{on}})-p_L=0\)——**production CV 的 additive null offset = 0**（無 midpoint 項）。Legacy「中點外推」\(\mathrm{gap}/2\) 只存在於 Path C mirrors，**不是** production baseline object。 | documented-production · existing-online §4.1 + relink decision contract |
| 9 | `null_offset_treatment` | Canonical A 層 family 必須在其 production-corresponding member 重現 zero null offset：M2 於 \(\gamma\to 0\)（nesting）精確回到 CV mean。\(\gamma>0\) 時 M2 mean 相對 raw-CV 外推的偏差 \((a-\Delta)u_0\) 是**宣告的 model drift**（記在 transition mean 內），不得當成 unaccounted bias；任何非零 additive offset（如 \(\bar v(c)\neq 0\) 的 context drift \(d_\Delta(c)\)）必須是顯式宣告的 model component，**不得**被 production-corresponding claim 靜默繼承。 | normative |

**Freeze 邊界：** 本表凍結的是 *定義與 convention*。它不宣稱任何 runtime
擷取值、不建立 fidelity edge、不選 model 參數、不解除 charter obligations
2–4。修改須 append-only correction（原行保留、註記 superseded）。

## §3 Observation modes and causal availability（frozen）

沿用 charter 的兩種 observation mode，凍結其地位：

\[
H_x=\begin{bmatrix}I&0\end{bmatrix}
\qquad\text{(position-only)},
\qquad
H_{xv}=I
\qquad\text{(joint position–velocity)}.
\]

1. 兩種 mode 是**不同的 claim object**，永不可互換使用。
2. **Production-corresponding instantiation 綁 \(H_x\)**（position-only entry
   observation，於 entry endpoint 取 \(y_1\)，不確定度 \(R_1\)）。
3. \(H_{xv}\) 只在**顯式宣告 causal availability** 後可用：entry velocity 若需
   lagged/future frames，宣告必須寫明其可得時點與來源；未宣告即不可引用。

## §4–§6 Reserved

`GCTM_MODEL_SPEC_SEALABLE` 之前必須完成（見 charter frozen terminal partition
與 obligation-status table）；在各自 work packet 落地前，本檔不預先陳述。

## History

- 2026-07-22 — D1 seed created by WP-A1: §2 nine-field canonical
  observation/time interface + §3 observation modes frozen (charter
  obligation 1); §4–§6 reserved.
