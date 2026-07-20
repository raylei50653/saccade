# Audit: Feasibility of Matrix/Table Representation of the Online Tracking Decision Path

**Date:** 2026-07-20  
**Scope:** Active online path in `GPUByteTracker` (`src/tracking/tracker_gpu.cu`)  
**Authority:** Runtime source over docs; docs used only as secondary maps  
**Status:** Analysis only — **no runtime changes**  
**Implementation authorized:** NO

---

## A. Executive verdict

**Verdict: `PARTIAL-GO`**

**Justification.** Large parts of the online path are *already* matrix- or table-shaped:

* dense pairwise cost matrix \(C_{td}\) with hard gate fill;
* sparse per-track candidate lists capped at \(K=16\);
* multi-stage softmin top-\(k\) tables;
* auction ownership as a bipartite claim operator with explicit packed tie-breakers;
* bridge pair features / gates as candidate×lost ranking with atomic claim.

What cannot be reduced to independent pair matrices without semantic distortion is the **ordered, stateful resolution stack**: multi-stage cascade carry-over, race-sensitive sparse enqueue, lifecycle mutation, birth/spawn, bank ReID claim, and bridge commit. A faithful decomposition is therefore:

> **pair-local \(G, S\) (and related features) + sparse reductions + explicit claim operators + thin imperative commit.**

This boundary already matches production structure. Making it more declarative would improve explainability, contract tests, and diagnostic exports — not by inventing a new global assignment model. A full “one matrix solves tracking” redesign is **not** justified and would mislabel auction/cascade/bridge semantics as Hungarian/global optimum.

---

## B. Current runtime flow

Source anchor: `GPUByteTracker::Impl::run_update_device` in `src/tracking/tracker_gpu.cu` (~L3466–3936). Host orchestration injects knobs via `pipeline.py` → `GPUByteTracker` setters (see `docs/research/tracker-decision/audit/callpoints.md`).

### Ordered pipeline (one frame)

```text
0. Pre-tracker (host, not in this kernel stack)
   private continuation may append low-score dets for CONTINUE-only
   quality scaling can rescale det scores on entry

1. Optional quality rescale of det scores
   apply_detection_quality_scaling_kernel

2. Optional ReID bank prep (relink_enabled && embeddings)
   archive_expiring_tracks_kernel
   age_relink_bank_kernel

3. Predict + GMC + S_inv (stateful motion)
   predict_gmc_sinv_fused_kernel
   inputs: Kalman states/covs, active, age, W_f, r_scale, optional occ vel damp
   outputs: predicted boxes/states for association; S_inv for Mahalanobis gate

4. Inter-track occlusion features (global over tracks, not track-det pairs)
   compute_track_occlusion_kernel → occ_coeff, occ_front_ttl, occ_partner[_all], duration ramp

5. Pair cost construction (dense N_trk × N_det) + sparse enqueue
   path A (no embeddings): stage1_cost_fused_kernel
   path B (embeddings): count_stage1_candidates_kernel + compute_conditional_cost_kernel
   outputs: d_cost_matrix, d_cand_{n,costs,indices} (≤ K_MAX_CANDIDATES=16)

6. Pre-association track-state housekeeping
   track_state_update_pre_kernel

7. Multi-stage softmin top-k materialization
   fused_sinkhorn_multistage_kernel → 5 stages × top-3 (p, det_idx)

8. Ordered cascade auction claim (5 stages)
   for stage in [S0 DDA, S1 Hi, S1b Mid, S1c Tentative, S2 Lo]:
     reset prices
     parallel_auction_shmem_kernel
     commit_auction_results_kernel
   carries trk_to_det / det_to_trk across stages

9. Association lifecycle commit
   track_state_update_post_kernel   (age, streak, confirm/kill tentative)
   inline_kalman_update_kernel      (matched measurements only)

10. Birth / free-slot / optional bank ReID revive
    collect_free_slots_kernel
    optional seed_reference_features_kernel
    optional archive_lost + invalidate_tracked_bank + relink_births_kernel + retire_revived_slots
    spawn_new_tracks_kernel
    init_covariance_if_new_kernel

11. Geometry bridge relink (bidirectional_ / relink_bridge_enabled path)
    update_foot_history_kernel
    optional occupancy_update_kernel
    relink_bidir_propose_kernel   (pair gates + bdist rank + margin + claim key)
    relink_bidir_commit_kernel    (winner adopts lost ID; deactivate lost)

12. Emit
    compact_results_kernel → result buffers (IDs after bridge rewrite)
```

### Stage table (association core)

| Stage | Inputs | Outputs | State read | Predicates / scores | Ordering | Mutations | Early exits |
| ----- | ------ | ------- | ---------- | ------------------- | -------- | --------- | ----------- |
| Predict+GMC | boxes history, W_f | predicted \(x\), \(S^{-1}\) | Kalman, age, active | — | per-slot parallel | predict in place | inactive skip |
| Occ features | predicted boxes | occ_coeff, front TTL, partners | active set | track-track IoU/foot/duration | O(n²) scan | TTL/duration counters | tau=0 no-op |
| Hard pair gate | pred box, det, S_inv | pass/fail; cost=1 on fail | track, det, frame | IoU > 0.30 OR Maha² < 9.4877 | pair-local | none | return cost=1 |
| Cost | pair + occ + optional ReID | \(c_{td}\in[0,1]\) | streak, score_sum, occ | multiplicative or additive Π | pair-local | none | — |
| Sparse enqueue | \(c_{td}\) | cand list ≤16 | race on atomicAdd | \(c \le\) cand_cost_cap | race if overflow | cand buffers | slot≥16 dropped |
| Stage filters | cost, det score, trk state | stage top-k p | active, unmatched | 5 stage bands | row-wise top-k | topk tables | unmatched-only |
| Auction stage s | top-k of stage s | assignments | prior stage claims | bid − price + biases | multi-track vs det | trk↔det maps | skip matched |
| Post assoc | assignments | lifecycle | state enum | confirm streak/score | per-track | age, active, state | tentative kill |
| Bank relink | unmatched high dets | revive id | bank valid | Cheb-GR + spatial + CAS | det-local + CAS | bank valid, det_revive | N_valid<3 |
| Spawn | free slots, revive | new tracks | free list order | new_track_thresh, prox | det chunk order | create slots | no free slots |
| Bridge propose | young cand vs lost | proposal + claim key | foot ring, age | many hard gates + bdist | cand-local min | track_revived=1 | streak≠at |
| Bridge commit | claim winners | ID rewrite | claim array | score-packed max | per-lost unique | track_ids, active | non-winner skip |

**Important naming note:** `fused_sinkhorn_multistage_kernel` builds softmin values \(p\propto e^{-\lambda c}\). It does **not** run iterative Sinkhorn OT. Assignment is **single-round parallel auction per cascade stage**, not Hungarian and not converged Sinkhorn (see `docs/reference/math_model.md` §8).

---

## C. Branch classification inventory

Classification legend:

* **A** pair-local pure predicate  
* **B** pair-local score contribution  
* **C** row/column reduction  
* **D** global compatibility / ownership  
* **E** ordered/stateful control flow  
* **F** side effect / lifecycle mutation  

| Runtime location | Current condition or operation | Inputs | Output/effect | Classification | Matrixizable? | Ordering-sensitive? | Mutation-sensitive? | Notes |
| ---------------- | ------------------------------ | ------ | ------------- | -------------- | ------------- | ------------------- | ------------------- | ----- |
| `apply_detection_quality_scaling_kernel` | rescale det scores by aspect/center/area | det scores/boxes, frame size | mutated scores | B (column feature), F | Partial (vector) | No | Yes (scores feed cascade bands) | Pre-matrix feature mutation |
| `predict_gmc_sinv_fused_kernel` | KF predict + GMC warp + S_inv | states, covs, W_f, r_scale | predicted state | E, F | No (state filter) | Age order per track | Yes | Feeds all pair geometry |
| `compute_track_occlusion_kernel` | max/union track-track IoU; duration ramp; front latch | all active predicted boxes | occ_coeff[t], TTL, partners | C (over tracks), E | As track×track matrix, not track×det | Scan order for argmax ties | Yes (duration/TTL) | Global over tracks; couples into Π |
| `oao_penalize_match` | partner-pred IoU ≥ contest thresh | t, det, occ_partner_all | whether OAO applies | A (pair+partner), D | Mask with partner feature | No | No for mask itself | Partner is track-global |
| `oao_score_scale` | \(1 - w\cdot\)det_score | det score | scale ∈[0,1] | B | Yes | No | No | |
| `stage1_cost_fused_kernel` IoU/Maha gate | IoU>0.30 OR Maha²<9.4877 | pred box, det, S_inv | pass or cost=1 | A | Yes \(G_{td}\) | No | No | Dense fill |
| same, fuse_score_weight path | score penalty vs track avg | streak, score_sum, det, occ | fused affinity A | B | Yes | No | Uses history stats | Baseline w=0 |
| same, multiplicative Π OAO | `penalty += τ·occ·scale` if contest | occ, det | cost | B | Yes | No | Reads occ state | |
| same, vel_dir_weight | reverse direction penalty | velocity, det center | penalty | B | Yes | No | No | Latent unless knobs on |
| same, occ under-foot | front TTL>0 and under>0 | footy, det bottom | penalty | B | Yes | No | Uses TTL latch | |
| same, association_energy | score/height energy terms | det score, heights | penalty | B | Yes | No | Flag-gated | |
| same, stability_cost_w | height-consistency reward (−Π)/λ | heights, λ | reward | B | Yes | No | No | Distinct from bid stability |
| same, cost form | \(c=\mathrm{clamp}(1-A e^{-\Pi})\) or additive | A, Π | \(c_{td}\) | B | Yes \(S_{td}\) | No | No | Dense matrix already |
| same, cand enqueue | if c≤cand_cost_cap then atomic slot | final_cost | sparse cand | C + E | Sparse list | **Yes if n>16** | No | **Race-ordered K slots** |
| `count_stage1_candidates_kernel` | IoU/Maha count for ReID branch | same gates | candidate_count[t] | A, C | Yes | No | No | Appearance gate input |
| `compute_conditional_cost_kernel` appearance | cos + IoU + score if clean emb & n≥min | embeds, counts | min(iou_cost, app_cost) | A+B | Yes | No | Embedding state | Off when no emb |
| `track_state_update_pre_kernel` | empty→confirmed cleanup | active | state housekeeping | E, F | No | No | Yes | |
| `fused_sinkhorn_multistage_kernel` stage membership | score band × trk state × cost cap | cand, scores, state | stage top-k | A + C | Yes per stage mask then top-k | Within-row cand order affects equal-p | No | insert_topk: `val<=` keeps old |
| same, aspect penalty | G_aspect on det box | det box | p scale | B | Yes | No | No | Hardcoded 0.8/0.15 |
| same, softmin | \(p=e^{-\lambda c}G\) | cost, λ | p | B | Yes | No | No | Name “sinkhorn” is historical |
| `parallel_auction_shmem_kernel` skip claimed det | det_to_trk[d]!=-1 | ownership | skip | D, E | As constraint | Stage order | Yes | Cascade coupling |
| same, bid = price + margin + biases | best-vs-second + ε | topk p, prices | bid | C + B | Partial | Equal bid → slot order | No | Freshness/stability env biases |
| same, packed bid atomicMax | float bits ‖ (n_trk−t) | bid, track slot | price winner | D | Claim op | **Yes (tie_breaker)** | Yes | Not Hungarian |
| `commit_auction_results_kernel` | g_prices[d]==pending_bid[t] | pending | trk_to_det, det_to_trk | D, F | Claim commit | Deterministic given prices | Yes | |
| Multi-stage cascade host loop | S0→S2 sequential | previous maps | progressive matching | E, D | Cascade of ops, not one matrix | **Yes** | Yes | Core ordered behavior |
| `track_state_update_post_kernel` matched | age=0, streak++, confirm if thresh | assignment, scores | state 1→2 | E, F | Transition table possible | No | Yes | |
| same, unmatched tentative | deactivate | state==1 | kill | E, F | Transition | No | Yes | |
| same, unmatched confirmed | reset streak; age continues via predict | — | lost aging | E, F | Transition | No | Yes | age++ in predict |
| `inline_kalman_update_kernel` | update matched only | det z, adapt modes | states/covs | E, F | No | No | Yes | Stateful features later |
| `relink_births_kernel` Cheb-GR | μ−λσ, spatial, floor, min arg D | bank, emb | best entry | A+B+C | Pair bank×det | Loop order on ties | Yes | |
| same, atomicCAS claim | claim bank entry once | valid flag | revive id | D, F | Sparse claim | CAS contention | Yes | |
| `retire_revived_slots_kernel` | kill live slots with revived id | revive list | active=false | D, E, F | No | Scan | Yes | Dup-id guard |
| `spawn_new_tracks_kernel` | unmatched score≥thresh; free slots | free list order | new tracks | E, F | No | **Free-slot / det chunk order** | Yes | Optional revive id |
| birth proximity gate | too close to confirmed | boxes | reject birth | A (det vs all trk) | Column mask | Scan | Reads live state | |
| `update_foot_history_kernel` | ring append + ema_h | matched states | history | F | No | Yes | Yes | Bridge features |
| `occupancy_update_kernel` | raster confirmed age0 | states | occ grid | F | Grid tensor | Yes | Yes | Optional bridge gate |
| `relink_bidir_propose` candidate fire | hit_streak==bridge_at, foot≥4, not revived | lifecycle | attempt | E | Row filter | Streak timing | Yes sets track_revived | Single-fire |
| same, lost structural eligibility | confirmed unmatched, age∈[min,ttl], foot≥1 | lost state | candidate set | A | Sparse mask | Lost slot iteration | Yes | |
| same, height/speed/spatial gates | ratio, m/s, cdist | ema, centers | reject | A | Yes | No | No | Optional when 0 |
| same, bdist score | speed-weighted fwd/bwd + dist_h + dir blend | foot anchors, v | bdist | B | Yes \(S_{cl}\) | No | History-derived | Lower better |
| same, cutoff bdist≤bridge_px | hard | bdist | reject | A | Yes | No | No | After dir blend |
| same, occ expand path | long-gap cover vs expand_px | occ grid | accept expand | A/E | Partial | Grid state | Yes | Optional |
| same, appearance veto | cos < app_veto | embeds | reject | A | Yes | No | Emb state | Off by default |
| same, portable OR-tail | multi-atom reject | thr vector | reject | A | Yes | No | Research flag | Default off |
| same, cand-local best/second | min bdist | eligible losts | best_lost | C | Yes argmin | Strict `<` on ties keeps earlier lost slot | No | |
| same, margin | second−best ≥ margin | best/second | proposal or reject | C | Yes | No | No | |
| same, claim key atomicMax | (q_score≪16)\|cand | score, cand idx | bridge_claim[lost] | D | Claim | **Higher cand index wins ties** | Yes | Not bdist at claim |
| `relink_bidir_commit_kernel` | winner cand adopts lost id | claim, maps | track_ids; active[lost]=0 | D, E, F | Thin commit | Winner only | **Yes** | No second-choice retry |
| `compact_results_kernel` | confirmed & age≤coast | states, ids | MOT rows | F | Emit | Compact order | Yes post-bridge IDs | Output only |

### Pair-local vs coupled summary

| Pure / nearly pure pair-local | Coupled / ordered / stateful |
| ----------------------------- | ---------------------------- |
| IoU/Maha gate → \(G_{td}\) | Occ coefficient over all tracks |
| Cost terms → \(S_{td}\) (given occ & history features) | Sparse K race when >16 eligible |
| Bridge height/speed/spatial/bdist/cutoff | 5-stage cascade carry-over |
| Softmin p from c | Auction multi-track vs det ownership |
| Stage membership masks (fixed inputs) | Confirm/kill/spawn/revive mutations |
| | Bridge claim by det score + no retry |
| | Bank CAS, free-slot assignment |

---

## D. Proposed decomposition (minimum faithful)

Do **not** force one giant matrix. Use the structures the runtime already has:

### 1. Pair feature tensor \(F_{ij,:}\) (or sparse pair list)

| Property | Assessment |
| -------- | ---------- |
| Exists implicitly? | Yes: IoU, Maha residual, heights, velocities, det score, occ_coeff[i], optional emb cos, bridge foot residuals |
| Dense vs sparse | Dense track×det for assoc; sparse cand×lost for bridge (structurally sparse) |
| Static vs frame | Frame-dependent; many features from stateful estimators |
| Deterministic? | Features yes if state fixed; sparse list membership under overflow is not fully host-deterministic under GPU races |
| Independent eval? | Yes **given** frozen frame state (predicted tracks, dets, occ buffers, history) |
| Outside | Building history, KF, GMC, occ duration/TTL |

### 2. Hard eligibility mask \(G_{ij}\)

| Layer | Mask content |
| ----- | ------------ |
| Assoc | IoU/Maha gate; optionally c≤cand_cost_cap as enqueue mask \(G^{\mathrm{enq}}\) |
| Stage s | \(G^{(s)}_{td}\) = active unmatched track × score band × state band × cost≤stage_cap |
| Bridge | structural lost eligibility × height × speed × spatial × cutoff × occ/app/tail |

Already exists as early returns and continue-paths. Fully matrixizable as Boolean (or multi-reason bitfield for diagnostics).

### 3. Score / cost matrix \(S_{ij}\)

| Domain | Symbol | Direction | Role |
| ------ | ------ | --------- | ---- |
| Assoc cost | \(c_{td}\) | lower better | enqueue + stage caps |
| Softmin | \(p_{td}=e^{-\lambda c}G_{\mathrm{aspect}}\) | higher better | auction value |
| Bridge | bdist | lower better | cand-local ranking only |

Already exists as `d_cost_matrix` and per-cand costs. **Do not mix domains** when comparing margins (see `scoring_semantics.md`).

### 4. Ranking / reduction \(R_i\)

| Operator | Reality |
| -------- | ------- |
| Per-track sparse list | atomic-capped top-of-gate list (not sorted top-16 costs) |
| Per-track stage top-k | fixed K=3 by p with equal-p retention rule |
| Bridge per-cand argmin bdist | best/second for margin |
| Bridge per-lost max claim key | det-score pack, not bdist |

Matrix-friendly **if** reduction semantics (including ties and race rules) are specified as operators, not left to “argmin in BLAS”.

### 5. Claim compatibility \(C\)

Represent as **operators**, not independent pair masks:

1. **Cascade auction claim:** one det per track and one track per det within/between stages; prior stage removals.
2. **Bank CAS claim:** one bank entry per revive.
3. **Bridge claim:** many-to-one cand→lost proposals; lost-side atomicMax on packed score.

Bipartite conflict graph / assignment operator language is accurate. Labeling this “Hungarian” or “global OT” is **not**.

### 6. State transition table \(T(s,e)\)

Feasible as an explicit table for lifecycle:

| Prior | Event | Next |
| ----- | ----- | ---- |
| EMPTY | spawn | TENTATIVE (or confirmed rules via birth_low) |
| TENTATIVE | match + streak/score | CONFIRMED |
| TENTATIVE | miss | EMPTY (deactivate) |
| CONFIRMED | match | CONFIRMED age=0 |
| CONFIRMED | miss | CONFIRMED age++ (via predict) until max_age archive/deactivate |
| any | bridge commit | cand keeps motion state; ID ← lost; lost inactive |

This is declarative **policy**, still applied by imperative kernels.

### 7. Commit layer

Must remain thin, imperative, mutation-owning:

* write trk_to_det / det_to_trk;
* KF update;
* spawn/free slots;
* bank archive/CAS/retire;
* bridge ID rewrite;
* compact emit.

**Nothing above the commit layer should mutate track identity or active sets.**

### Dense vs sparse recommendation

| Structure | Recommendation |
| --------- | -------------- |
| Assoc costs | Keep dense \(N_t\times N_d\) for simplicity (already allocated) **or** export only sparse eligible edges for diagnostics |
| Auction input | Sparse only (≤16×stages) — dense top-k over all dets would change cost if gates differ |
| Bridge | **Sparse** cand×lost — dense max_objs² is wasteful and invites evaluating structurally illegal pairs |
| Faithful default | **Sparse pair list + indexed reductions** for any prototype that claims equivalence |

---

## E. Semantic equivalence risks

### 1. Tie-breaking

| Site | Rule today | Risk if matrixized naively |
| ---- | ---------- | -------------------------- |
| Sparse cand slots | `atomicAdd` order; first K winners keep slots | Deterministic sort by cost would **change** who is enqueued when n_cand>16 |
| Stage top-k insert | equal p does not displace (`val <= existing` continues) | Stable sort by (p, −det_idx) may change equals |
| Top-k warp merge | on equal p prefers left (`av >= bv`) | Different reduction tree changes equals |
| Auction bid ties | packed `tie_breaker = n_trk - t` (higher slot wins) | Argmax without slot key breaks matches |
| Bridge bdist ties | strict `<` keeps earlier lost slot index | Argmin with different order changes best_lost |
| Bridge claim ties | higher quantized score; else higher cand index in low 16 bits | Using bdist or lower index would change winners |
| Bank best_d | strict `<` + CAS | Parallel race different from sequential if not CAS-identical |
| Spawn free slots | free list order + det chunking | Reordering births changes IDs |

### 2. Early cutoff ordering

Cutoffs are **not** algebraically free to reorder:

1. IoU/Maha hard reject **before** scoring (cost forced to 1, no enqueue).
2. Full cost including bonuses/penalties **before** cand_cost_cap.
3. cand_cost_cap uses **loosest** of {dda_max, match_thresh, stage2_match_thresh}.
4. Stage-specific caps applied again at softmin top-k time (stricter subset).
5. Bridge: structural gates → pre-score gates → bdist → cutoff/expand → vetoes → rank → margin → claim.

Applying threshold after ranking, or ranking before bonuses, is **not** equivalent.

### 3. Claim / commit model (label carefully)

| Component | Actual semantics | Not |
| --------- | ---------------- | --- |
| Assoc stages | Ordered greedy cascade of single-round auctions on stage top-k | Global Hungarian; multi-iter auction to equilibrium; Sinkhorn OT |
| Within stage | Parallel bid with price atomicMax + slot tie-break | Independent row-wise argmin (unless no contention) |
| Bridge | Cand-local argmin bdist + lost-side det-score claim; **loser no retry** | Mutual nearest neighbor; global assignment on bdist |
| Bank relink | Local best under Cheb+spatial + CAS | Global multi-matching |

### 4. Sparse structure hazards

* Dense evaluation of all pairs is **fine for scores** if gates set cost=1, but **not fine for sparse lists**: enqueue uses atomic race when oversubscribed.
* Fabricating pairs that runtime never considers (inactive tracks, already-matched, wrong stage) pollutes diagnostics.
* Bridge dense \(N\times N\) includes illegal pairs (self, non-lost, non-cand) — must mask.

### 5. Stateful features feeding matrices

These can **feed** \(F,G,S\) but are not themselves pair-matrix ops:

* Kalman state/cov, GMC warp, age, hit_streak, score_sum, confirm_req;
* occ_coeff duration ramp, occ_front_ttl;
* foot ring, ema_h, track_revived;
* relink bank embeddings/positions/valid;
* free slot occupancy and next track id counters.

### 6. Floating-point / regrouping

* Multiplicative \(1 - A e^{-\Pi}\) vs additive chain: already config-gated; do not “simplify” algebraically.
* Softmin \(e^{-\lambda c}\) sensitive near λ=10 for cost edges.
* Stability reward deliberately divides by λ so bid effect is λ-stable — regrouping into unnormalized terms changes auction.
* Bridge bdist uses several sqrt/rsqrt chains; reassociation of sum-of-squares can change equality at cutoff boundaries.
* Quantized claim score (×32767) vs float compare.
* GPU atomic order vs host reference evaluator must pin the same tie rules.

### 7. Dual stability (common confusion)

* `stability_cost_w` → cost Π (matrix \(S\)).
* `SACCADE_STABILITY_W` → auction bid bias **after** p (resolution layer).

Matrixizing only \(S\) without bid biases does **not** reproduce assignment.

---

## F. Candidate prototype boundary

Given **PARTIAL-GO**, the smallest justified prototype is **not** a tracker rewrite.

### Recommended prototype (narrowest useful)

**Option 1 + 2 hybrid (preferred): read-only instrumentation + test-only reference evaluator**

1. **Export (read-only, default off)** for a fixed frame snapshot after predict+occ, before auction mutation:
   * sparse eligible pairs (t,d) with gate reason bits;
   * \(c_{td}\) for enqueued pairs (and optionally dense for gated failures as c=1);
   * stage membership flags;
   * optional bridge pair table: (cand,lost,bdist,gate_reason) for fire candidates only.
2. **Host reference evaluator** in tests that recomputes \(G,S\) from dumped state tensors and asserts:
   * identical eligible set and reject reasons;
   * cost within declared ULPs/tolerance;
   * identical stage top-k under documented tie rules;
   * optional shadow: recompute auction with same packed bids → same trk_to_det.

### Explicitly out of scope for first prototype

* Replacing auction with Hungarian/Sinkhorn;
* Changing sparse K policy to sorted top-K;
* Merging cascade stages into one matrix solve;
* Moving bridge claim to bdist-global matching;
* Any frozen-online metric claim.

### Acceptable alternate single-stage prototype

**Hard-gate only:** reimplement IoU/Maha \(G_{td}\) as pure function of (pred box, det, S_inv) and contract-test against kernel dump. Lowest risk; smaller benefit.

Existing hooks that already lean this way: `SACCADE_ASSOC_DUMP`, H0 bridge pair/claim/commit records, bridge fidelity events — extend rather than invent parallel truth.

---

## G. Validation requirements (future implementation)

A future behavior-preserving change is complete only if **all** relevant checks pass for authorized sequences:

| Check | Criterion |
| ----- | --------- |
| Eligible pair set | Exact match of (track_slot, det_idx) passing IoU/Maha |
| Reject reasons | Per-pair reason codes stable (gate taxonomy) |
| Pair cost | Bit-identical preferred; else declared float tolerance + no rank flips at thresholds |
| Sparse list | Same membership **or** documented race policy replaced by deterministic rule with A/B metric authorization |
| Stage top-k | Identical indices and p ordering under tie policy |
| Auction assignment | Identical trk_to_det / det_to_trk per stage and final |
| Bridge pairs | Identical final_pair_eligible set; identical best/second; identical margin outcomes |
| Bridge claims | Identical winners under packed key rule |
| Lifecycle | Identical active/state/age/hit_streak transitions |
| IDs | Identical emitted track IDs (post-bridge) |
| MOT output | Identical rows (or authorized metric delta=0 on freeze suite) |
| Determinism | Repeated runs same stream config produce same exports |
| Frozen online metrics | **No change** unless separately authorized |

Offline pair tables and Python semantic relinker are **non-authoritative** for this online contract (`relink_bridge.md` three-contract warning).

---

## Benefits assessment (not assumed)

| Benefit | Expected? | Why |
| ------- | --------- | --- |
| Explainability | **High** | Gate reason bits + term-wise Π already map to “why rejected / why ranked” |
| Testability | **High** | Contract tests on \(G,S,R,C\) without full MOT | 
| Comparability | **Medium-High** | Stable interface for score ablations vs offline diagnostics |
| Vectorization / speed | **Unproven** | Path is already GPU dense/sparse kernels; no speedup claimed |
| Governance | **High** | Clarifies online truth vs math abstraction vs experiment hooks |
| Full redesign as linear algebra | **Low / harmful** | Would blur cascade/auction/bridge semantics |

Primary expected benefit: **clarity + contract tests + diagnostic parity**, not FPS.

Primary semantic risk: **silent change of tie-breaking, sparse K membership, or claim model** while claiming matrix “equivalence.”

---

## Answers to audit questions (concise)

1. **Actual pipeline:** predict/GMC → occ → dense cost+sparse cand → 5-stage softmin top-k → cascade auction → lifecycle/KF → spawn/(optional bank) → bridge propose/claim/commit → compact.  
2. **Pair-local branches:** IoU/Maha gates, most cost Π terms, softmin, most bridge pair gates/scores.  
3. **Order/mutation/global:** occ over tracks, sparse race, cascade ownership, auction ties, lifecycle, bank CAS, bridge claim, spawn IDs.  
4. **Smallest faithful decomp:** \(F,G,S\) + sparse \(R\) + claim operators \(C\) + \(T(s,e)\) + thin commit.  
5. **Improve clarity/test/analyze/vectorize?** Clarity/test/analyze: yes. Vectorize: already partly done; no free lunch.  
6. **Risks:** ties, cutoffs, cascade≠global assign, sparse K, FP, dual stability, offline proxy drift.  
7. **Justify later task?** Yes, **bounded** prototype (export + reference \(G/S\)); **no** full tracker matrix rewrite.

---

## Preferred conclusion block

```text
Verdict: PARTIAL-GO
Matrixizable boundary: pair features F; hard masks G (assoc + bridge); costs/scores S (c, p, bdist); row-local top-k / argmin reductions with explicit tie rules; declarative lifecycle table T(s,e) as documentation/policy
Non-matrixizable boundary: sparse K race policy; multi-stage cascade carry-over; auction ownership + bid biases + packed ties; bank CAS; spawn/free-slot ordering; bridge score-claim without retry; KF/history/occ TTL mutations; emit
Primary expected benefit: explainability and contract-testable separation of gates/scores/claims without changing frozen online truth
Primary semantic risk: non-equivalent tie-breaking or claim/cascade semantics when “matrixizing” reductions
Recommended next action: read-only export of (G,S, pair IDs, stage top-k, bridge pair table) + host reference evaluator; no production path swap
Implementation authorized: NO
```

---

## Source anchors (non-exhaustive)

| Area | Primary source |
| ---- | -------------- |
| Update driver | `src/tracking/tracker_gpu.cu` `run_update_device` |
| Cost + sparse | `stage1_cost_fused_kernel`, `compute_conditional_cost_kernel` |
| Softmin stages | `fused_sinkhorn_multistage_kernel` |
| Auction | `parallel_auction_shmem_kernel`, `commit_auction_results_kernel` |
| Bridge | `relink_bidir_propose_kernel`, `relink_bidir_commit_kernel` |
| Bank relink | `relink_births_kernel` |
| Semantics docs | `docs/research/tracker-decision/scoring_semantics.md`, `relink_bridge.md` |
| Math | `docs/reference/math_model.md` §7–8, §10 |
| Callpoints | `docs/research/tracker-decision/audit/callpoints.md` |

**Code wins** if documentation drifts.
