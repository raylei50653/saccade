# Relink Gates & Formulas — Complete Reference

Summary of all relink thresholds and mathematical formulas in the saccade tracker.
Based on `mamba_whole_graph` preset (`reid_mode: "off"`, `relink_bridge_enabled: true`).

> Line anchors in the Location columns are historical and may lag the source; bridge
> knobs live on the tracker config/constructor, not `lifecycle.py`. For the
> authoritative bridge decision contract see the H0 declaration
> (`docs/modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md`).

---

## Active Gates (mamba_whole_graph)

### 1. Speed-Weighted Bridge Score

```
w       = sqrt(clamp(s_lost / 0.12, 0, 1))
fwd_r   = ‖(lost_pos + v_lost·gap) − cand_pos‖ / h_ref
bwd_r   = ‖(cand_pos − v_cand·gap) − lost_pos‖ / h_ref
sym_fb  = 0.5·(fwd_r + bwd_r)
dist_h  = ‖lost_pos − cand_pos‖ / h_ref
bdist   = w·sym_fb + (1−w)·dist_h

Accept if bdist ≤ bridge_px
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `relink_bridge_px` | 0.25 | `lifecycle.py:84`, `tracker_gpu.cu:1435-1443` |
| `h_ref` | `avg(ema_h_lost, ema_h_cand)` | `tracker_gpu.cu:1432` |
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
| `relink_bridge_h_lo` | 0.75 | `lifecycle.py:95` |
| `relink_bridge_h_hi` | 1.33 | `lifecycle.py:96` |

### 3. Margin Gate (Ambiguity Rejection)

```
Reject if (second_best_dist − best_dist) < bridge_margin
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `relink_bridge_margin` | 0.05 | `lifecycle.py:91`, `tracker_gpu.cu:1448` |

> **Winner stages (not a joint score, not global assignment).** `bdist` ranks losts
> **within one candidate** (lower is better); the margin above applies to that
> candidate-local best-vs-second. When multiple candidates pass gates for the same
> lost, the winner is picked by an atomic packed key of the **quantized detection
> score** (candidate index tie-break) — `bdist` is not re-compared across candidates.
> A claim loser does not retry its second-best lost. Commit mutates only
> `track_ids[cand]` and `active[lost]`. There is no Hungarian / bipartite
> re-ranking. Authoritative decision contract: H0 declaration
> (`docs/modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md`);
> decision semantics: `docs/research/tracker-decision/relink_bridge.md`.

### 4. TTL & Min-Lost Gate

```
age = current_frame − last_seen
Reject if age < min_lost OR age > ttl
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `relink_bridge_min_lost` | 2 | `lifecycle.py:86` |
| `relink_bridge_ttl` | 120 | `lifecycle.py:87` |

### 5. Hit-Streak Trigger

```
Fire bridge attempt when hit_streak[cand] == bridge_at
  AND foot_len ≥ 4
  AND track_revived == 0
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `relink_bridge_at` | 4 | `lifecycle.py:85`, `tracker_gpu.cu:1372` |

### 6. Velocity Regression (4-Point OLS)

```
v = (3·p₃ + p₂ − p₁ − 3·p₀) / 10
```

Closed-form least-squares over the last 4 foot positions.

| Location |
|----------|
| `relink.py:471-489`, `relink_gate.cu:17-24`, `tracker_gpu.cu:1173-1178` |

### 7. EMA Height Tracking

```
ema_h = 0.95·ema_h_old + 0.05·current_h
```

| Location |
|----------|
| `relink.py:1339`, `tracker_gpu.cu:1350` |

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
| `relink_bridge_spatial_gate` | 0.0 (disabled) | `lifecycle.py:92`, `tracker_gpu.cu:1425-1429` |

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
| `relink_bridge_max_speed` | 0.0 (disabled) | `lifecycle.py:88` |
| `relink_bridge_person_height` | 1.65 | `lifecycle.py:89` |
| `relink_bridge_fps` | 30.0 | `lifecycle.py:90` |

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
| `relink_enabled` | False | `lifecycle.py:73` |
| `relink_sim_thresh` | 0.6 | `lifecycle.py:75` |
| `relink_lambda` | 2.5 | `lifecycle.py:76` |
| `relink_spatial_gate` | 4.0 (γ) | `lifecycle.py:77` |
| `relink_max_age` | 300 | `lifecycle.py:78` |
| `relink_bank_cap` | 256 | `lifecycle.py:74` |

### 11. Cosine Similarity Threshold (Python Semantic Relinker)

```
sim = dot(query_emb, stored_emb)   (both L2-normalized)
Reject if sim < sim_threshold
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `sim_threshold` | 0.985 | `relink.py:53` |

### 12. Spatial Gate (Python Semantic Relinker)

```
center_norm = ‖center_query − center_lost‖ / max(frame_w, frame_h)
Reject if center_norm > spatial_gate
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `spatial_gate` | 0.11 | `relink.py:56` |
| `min_iou` | 0.0 | `relink.py:58` |

### 13. Mahalanobis Gate (Static Snapshot)

```
residual = measurement(box) − state[:4]
S        = P[:4,:4] + R    (R diagonal, pos_std = h/20)
maha     = residual^T · S⁻¹ · residual
Reject if maha > mahalanobis_threshold
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `mahalanobis_threshold` | 6.6 | `relink.py:59` |

### 14. Kalman Probabilistic Gate (Chi²)

```
Extrapolate state delta frames: x, P = predict_phys(x, P) repeated
S         = P[:2,:2] + R   (2-dim) or P[:4,:4] + R (4-dim)
kalman_d² = residual^T · S⁻¹ · residual
Reject if kalman_d² > kalman_chi2
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `kalman_gate` | False | `relink.py:84` |
| `kalman_chi2` | 9.4877 | `relink.py:85` |
| `kalman_penalty_weight` | 0.0 | `relink.py:86` |

### 15. Direction Gate (Velocity Cosine)

```
cos_angle = dot(displacement, velocity) / (‖displacement‖·‖velocity‖)
Reject if cos_angle < kalman_dir_min_cos AND speed ≥ kalman_dir_min_speed
Disabled when kalman_dir_min_cos ≤ −1
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `kalman_dir_min_cos` | −1.0 (disabled) | `relink.py:87` |
| `kalman_dir_min_speed` | 1.0 px/frame | `relink.py:88` |

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
| `w_sim_base` | 1.0 | `relink.py:66` |
| `w_iou_base` | 0.0 | `relink.py:67` |
| `w_maha_base` | 0.0 | `relink.py:68` |
| `shift_ambiguity` | 0.0 | `relink.py:69` |
| `shift_lost_age` | 0.0 | `relink.py:70` |

### 17. Dynamic Margin (Python Semantic Relinker)

```
effective_margin = reciprocal_margin
    + dynamic_margin_crowd · min(1, (n_passed−1)/8)
    + dynamic_margin_age   · min(1, lost_frames/ttl)
Reject if (best_joint − second_best_joint) < effective_margin
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `reciprocal_margin` | 0.0 | `relink.py:63` |
| `dynamic_margin_crowd` | 0.0 | `relink.py:71` |
| `dynamic_margin_age` | 0.0 | `relink.py:72` |

### 18. Biometric Gate

```
bio_dist = Σ|ratio_a[k] − ratio_b[k]|  for k ∈ {leg, shoulder, head}
Reject if bio_dist > biometric_threshold
Disabled when threshold == 0
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `biometric_threshold` | 0.0 (disabled) | `relink.py:73` |

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
| `clean_score_threshold` | 0.0 | `relink.py:75` |
| `clean_margin_ratio` | 0.0 | `relink.py:76` |
| `clean_min_aspect` | 0.0 | `relink.py:77` |
| `clean_max_aspect` | 99.0 | `relink.py:78` |
| `strict_sim_threshold` | 0.0 | `relink.py:79` |

### 20. Density-Adaptive Mahalanobis

```
density   = count(neighbours within k·h of lost track)
threshold = mahalanobis_threshold · exp(−eta · density)
```

| Parameter | Default | Location |
|-----------|---------|----------|
| `exp_density_gating` | False | `relink.py:113` |
| `exp_density_k` | 2.0 | `relink.py:114` |
| `exp_density_eta` | 0.15 | `relink.py:115` |

---

## Quick Reference: mamba_whole_graph Active State

| Gate | Status | Key Threshold |
|------|--------|---------------|
| Bridge distance (speed-weighted) | **ON** | bdist ≤ 0.25 |
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
