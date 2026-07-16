# Relink Gates & Formulas — Complete Reference

Summary of all relink thresholds and mathematical formulas in the saccade tracker.
Based on `mamba_whole_graph` preset (`reid_mode: "off"`, `relink_bridge_enabled: true`).

> Location anchors are symbol-level (config field / kernel / class); line numbers are
> intentionally omitted because they do not survive edits. Bridge knobs live on
> `EvalConfig` (`config.py`) and the tracker constructor — not `lifecycle.py`.
> Source/runtime authority: `src/tracking/tracker_gpu.cu`; formula derivations:
> `docs/reference/math_model.md` §10.3–10.5. The H0 declaration
> (`docs/modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md`)
> is a **pre-seal draft capture contract** (policy target = `m` preset), not a
> description authority for this sheet.

---

## Active Gates (mamba_whole_graph)

### 1. Speed-Weighted Bridge Score (base + direction adjustment)

```
w       = sqrt(clamp(s_lost / 0.12, 0, 1))
fwd_r   = ‖(lost_pos + v_lost·gap) − cand_pos‖ / h_ref
bwd_r   = ‖(cand_pos − v_cand·gap) − lost_pos‖ / h_ref
sym_fb  = 0.5·(fwd_r + bwd_r)
dist_h  = ‖lost_pos − cand_pos‖ / h_ref
b0      = w·sym_fb + (1−w)·dist_h        # base score (bdist_before_direction)

# Direction adjustment — ACTIVE on mamba_whole_graph (dir_bonus = 0.8).
# Fires only when dir_bonus > 0, both velocities are trusted, and
# cos_sim(v_lost, v_cand) > 0.5; otherwise bdist = b0.
d_cross = 0.5·(fwd_cross + bwd_cross)    # cross-track-only residuals / h_ref
α       = clamp(dir_bonus · cos² · speed_trust · min(gap/30, 1), 0, 1)
bdist   = (1−α)·b0 + α·d_cross           # bdist_after_direction

Accept if bdist ≤ bridge_px
```

The cutoff, candidate-local ranking, and margin all consume the
**post-direction** `bdist`, never `b0`.

| Parameter | Default | Location |
|-----------|---------|----------|
| `relink_bridge_px` | 0.25 (s) / 0.4 (m) | `EvalConfig.relink_bridge_px` (`config.py`) → `relink_bidir_propose_kernel` (`tracker_gpu.cu`) |
| `relink_bridge_dir_bonus` | **0.8 (s preset)** / 0.0 (m, explicit) / 0.0 (schema) | `EvalConfig.relink_bridge_dir_bonus` → same kernel |
| `h_ref` | `max(avg(ema_h_lost, ema_h_cand), 1)` | `relink_bidir_propose_kernel` |
| `s_lost` | `‖v_lost‖ / h_ref` | speed in heights/frame |
| saturation | 0.12 h/f | hardcoded in sqrt ramp |

### 2. Scale Gate (Height Ratio)

```
ratio = ema_h_lost / ema_h_cand
Reject if ratio < bridge_h_lo OR ratio > bridge_h_hi
Disabled when bridge_h_hi ≤ 0
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `relink_bridge_h_lo` | 0.75 (s) / 0.6 (m) | `EvalConfig.relink_bridge_h_lo` (`config.py`) → `relink_bidir_propose_kernel` |
| `relink_bridge_h_hi` | 1.33 (s) / 1.7 (m) | `EvalConfig.relink_bridge_h_hi` → same kernel |

### 3. Margin Gate (Ambiguity Rejection)

```
Reject if (second_best_dist − best_dist) < bridge_margin
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `relink_bridge_margin` | 0.05 | `EvalConfig.relink_bridge_margin` (`config.py`) → `relink_bidir_propose_kernel` |

> **Winner stages (not a joint score, not global assignment).** `bdist` ranks losts
> **within one candidate** (lower is better); the margin above applies to that
> candidate-local best-vs-second. When multiple candidates pass gates for the same
> lost, the winner is picked by an atomic packed key of the **quantized detection
> score** (candidate index tie-break) — `bdist` is not re-compared across candidates.
> A claim loser does not retry its second-best lost. Commit mutates only
> `track_ids[cand]` and `active[lost]`. There is no Hungarian / bipartite
> re-ranking. Source/runtime authority: `relink_bidir_propose_kernel` /
> `relink_bidir_commit_kernel` (`tracker_gpu.cu`); decision semantics:
> `docs/research/tracker-decision/relink_bridge.md`.

### 4. TTL & Min-Lost Gate

```
age = current_frame − last_seen
Reject if age < min_lost OR age > ttl
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `relink_bridge_min_lost` | 2 | `EvalConfig.relink_bridge_min_lost` (`config.py`) → `relink_bidir_propose_kernel` |
| `relink_bridge_ttl` | 120 | `EvalConfig.relink_bridge_ttl` → same kernel |

### 5. Hit-Streak Trigger

```
Fire bridge attempt when hit_streak[cand] == bridge_at
  AND foot_len ≥ 4
  AND track_revived == 0
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `relink_bridge_at` | 4 | `EvalConfig.relink_bridge_at` (`config.py`) → `relink_bidir_propose_kernel` |

### 6. Velocity Regression (4-Point OLS)

```
v = (3·p₃ + p₂ − p₁ − 3·p₀) / 10
```

Closed-form least-squares over the last 4 foot positions.

| Location |
|----------|
| `PythonSemanticRelinker` (`relink.py`) · `relink_gate.cu` · `bridge_vel4` (`tracker_gpu.cu`) |

### 7. EMA Height Tracking

```
ema_h = 0.95·ema_h_old + 0.05·current_h
```

| Location |
|----------|
| `PythonSemanticRelinker` (`relink.py`) · `update_foot_history_kernel` (`tracker_gpu.cu`) |

---

## Disabled Gates (available but OFF in mamba_whole_graph)

### 8. Spatial Pre-Filter (Bridge)

```
cdist = ‖cand_center − lost_center‖ / h_ref
Reject if cdist > bridge_spatial_gate
Disabled when bridge_spatial_gate == 0
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `relink_bridge_spatial_gate` | 0.0 (disabled) | `EvalConfig.relink_bridge_spatial_gate` (`config.py`) → `relink_bidir_propose_kernel` |

### 9. Physical Speed Gate

```
dpx       = ‖center_query − center_lost‖
dt_s      = max(age, 1) / fps
px_per_m  = 0.5·(h_lost + h_query) / person_height_m
speed_mps = dpx / dt_s / px_per_m
Reject if speed_mps > max_speed_mps
Disabled when max_speed_mps == 0
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `relink_bridge_max_speed` | 0.0 (disabled) | `EvalConfig.relink_bridge_max_speed` (`config.py`) → `relink_bidir_propose_kernel` |
| `relink_bridge_person_height` | 1.65 | `EvalConfig.relink_bridge_person_height` |
| `relink_bridge_fps` | 30.0 | `EvalConfig.relink_bridge_fps` |

### 10. Birth-Bank Chebyshev-GR Relink (GPU ReID)

```
D      = 1 − cos_sim(query, bank_entry)
μ      = ΣD / N,  σ² = ΣD²/N − μ²
T_cheb = μ − λ·σ
r_max  = γ·h + 0.8·‖v‖·dt

Accept if D ≤ T_cheb AND D ≤ (1−sim_thresh)
           AND spatial_dist ≤ r_max AND N_valid ≥ 3
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `relink_enabled` | False | `EvalConfig.relink_enabled` (`config.py`) → `relink_births_kernel` (`tracker_gpu.cu`) |
| `relink_sim_thresh` | 0.6 | `EvalConfig.relink_sim_thresh` → same kernel |
| `relink_lambda` | 2.5 | `EvalConfig.relink_lambda` → same kernel |
| `relink_spatial_gate` | 4.0 (γ) | `EvalConfig.relink_spatial_gate` → same kernel |
| `relink_max_age` | 300 | `EvalConfig.relink_max_age` → same kernel |
| `relink_bank_cap` | 256 | `EvalConfig.relink_bank_cap` → same kernel |

### 11. Cosine Similarity Threshold (Python Semantic Relinker)

```
sim = dot(query_emb, stored_emb)   (both L2-normalized)
Reject if sim < sim_threshold
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `sim_threshold` | 0.985 | `PythonSemanticRelinker` (`relink.py`) |

### 12. Spatial Gate (Python Semantic Relinker)

```
center_norm = ‖center_query − center_lost‖ / max(frame_w, frame_h)
Reject if center_norm > spatial_gate
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `spatial_gate` | 0.11 | `PythonSemanticRelinker` (`relink.py`) |
| `min_iou` | 0.0 | `PythonSemanticRelinker` (`relink.py`) |

### 13. Mahalanobis Gate (Static Snapshot)

```
residual = measurement(box) − state[:4]
S        = P[:4,:4] + R    (R diagonal, pos_std = h/20)
maha     = residual^T · S⁻¹ · residual
Reject if maha > mahalanobis_threshold
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `mahalanobis_threshold` | 6.6 | `PythonSemanticRelinker` (`relink.py`) |

### 14. Kalman Probabilistic Gate (Chi²)

```
Extrapolate state delta frames: x, P = predict_phys(x, P) repeated
S         = P[:2,:2] + R   (2-dim) or P[:4,:4] + R (4-dim)
kalman_d² = residual^T · S⁻¹ · residual
Reject if kalman_d² > kalman_chi2
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `kalman_gate` | False | `PythonSemanticRelinker` (`relink.py`) |
| `kalman_chi2` | 9.4877 | `PythonSemanticRelinker` (`relink.py`) |
| `kalman_penalty_weight` | 0.0 | `PythonSemanticRelinker` (`relink.py`) |

### 15. Direction Gate (Velocity Cosine)

```
cos_angle = dot(displacement, velocity) / (‖displacement‖·‖velocity‖)
Reject if cos_angle < kalman_dir_min_cos AND speed ≥ kalman_dir_min_speed
Disabled when kalman_dir_min_cos ≤ −1
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `kalman_dir_min_cos` | −1.0 (disabled) | `PythonSemanticRelinker` (`relink.py`) |
| `kalman_dir_min_speed` | 1.0 px/frame | `PythonSemanticRelinker` (`relink.py`) |

### 16. Unified Joint Score (Python Semantic Relinker)

```
w_sim += shift_ambiguity · min(1, (n_passed−1)/8)
w_sim += shift_lost_age  · min(1, age/ttl)
Normalize: ws, wi, wm = ws/(ws+wi+wm), ...

joint = ws·sim + wi·iou + wm·maha_score + motion_bonus
maha_score = max(0, 1 − maha/threshold)
kalman_penalty: joint −= kpw · (1 − exp(−0.5·kalman_d²))
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `w_sim_base` | 1.0 | `PythonSemanticRelinker` (`relink.py`) |
| `w_iou_base` | 0.0 | `PythonSemanticRelinker` (`relink.py`) |
| `w_maha_base` | 0.0 | `PythonSemanticRelinker` (`relink.py`) |
| `shift_ambiguity` | 0.0 | `PythonSemanticRelinker` (`relink.py`) |
| `shift_lost_age` | 0.0 | `PythonSemanticRelinker` (`relink.py`) |

### 17. Dynamic Margin (Python Semantic Relinker)

```
effective_margin = reciprocal_margin
    + dynamic_margin_crowd · min(1, (n_passed−1)/8)
    + dynamic_margin_age   · min(1, lost_frames/ttl)
Reject if (best_joint − second_best_joint) < effective_margin
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `reciprocal_margin` | 0.0 | `PythonSemanticRelinker` (`relink.py`) |
| `dynamic_margin_crowd` | 0.0 | `PythonSemanticRelinker` (`relink.py`) |
| `dynamic_margin_age` | 0.0 | `PythonSemanticRelinker` (`relink.py`) |

### 18. Biometric Gate

```
bio_dist = Σ|ratio_a[k] − ratio_b[k]|  for k ∈ {leg, shoulder, head}
Reject if bio_dist > biometric_threshold
Disabled when threshold == 0
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `biometric_threshold` | 0.0 (disabled) | `PythonSemanticRelinker` (`relink.py`) |

### 19. Detection Quality Filter

```
Unclean if: score < clean_score_threshold
         OR box_in_margin(clean_margin_ratio)
         OR aspect < clean_min_aspect
         OR aspect > clean_max_aspect
Unclean detections → use strict_sim_threshold instead of sim_threshold
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `clean_score_threshold` | 0.0 | `PythonSemanticRelinker` (`relink.py`) |
| `clean_margin_ratio` | 0.0 | `PythonSemanticRelinker` (`relink.py`) |
| `clean_min_aspect` | 0.0 | `PythonSemanticRelinker` (`relink.py`) |
| `clean_max_aspect` | 99.0 | `PythonSemanticRelinker` (`relink.py`) |
| `strict_sim_threshold` | 0.0 | `PythonSemanticRelinker` (`relink.py`) |

### 20. Density-Adaptive Mahalanobis

```
density   = count(neighbours within k·h of lost track)
threshold = mahalanobis_threshold · exp(−eta · density)
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `exp_density_gating` | False | `PythonSemanticRelinker` (`relink.py`) |
| `exp_density_k` | 2.0 | `PythonSemanticRelinker` (`relink.py`) |
| `exp_density_eta` | 0.15 | `PythonSemanticRelinker` (`relink.py`) |

---

## Quick Reference: mamba_whole_graph Active State

| Gate | Status | Key Threshold |
|------|--------|---------------|
| Bridge distance (speed-weighted, post-direction) | **ON** | bdist ≤ 0.25 |
| Bridge direction bonus (blend toward cross-track) | **ON** | dir_bonus = 0.8 (m preset: 0.0 explicit) |
| Scale gate | **ON** | ratio ∈ [0.75, 1.33] |
| Bridge margin | **ON** | Δ ≥ 0.05 |
| TTL | **ON** | 2 ≤ age ≤ 120 |
| Hit-streak trigger | **ON** | streak == 4 |
| Velocity regression | **ON** | 4-pt OLS |
| EMA height | **ON** | α = 0.05 |
| Spatial pre-filter (bridge) | OFF | 0.0 |
| Physical speed gate | OFF | 0.0 |
| Birth-bank Cheb-GR | OFF | relink_enabled=False |
| Cosine sim (semantic) | OFF | reid_mode="off" |
| Mahalanobis (semantic) | OFF | reid_mode="off" |
| Kalman chi² | OFF | kalman_gate=False |
| Direction gate | OFF | dir_min_cos=−1 |
| Unified joint score | OFF | reid_mode="off" |
| Dynamic margin | OFF | reciprocal_margin=0 |
| Biometric | OFF | threshold=0 |
| Quality filter | OFF | clean_score=0 |
| Density gating | OFF | exp_density=False |
