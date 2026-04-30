# GPU Pipeline M4-b — Native Identity Resolver

Date: 2026-04-29 （設計） / 2026-04-30 （Phase A/B/C 全完成）

## 背景

`M4-a / M4-b` 已把 runner 的 hot path 從「逐筆混合控制流」整理成 staged batch pipeline：

```
HostTrackBatch -> PreparedTrackCandidate -> RelinkedTrackCandidate -> ResolvedTrack -> output
```

但 `PreparedTrackCandidate -> RelinkedTrackCandidate -> ResolvedTrack` 這一段仍存在 Python 縫合：runner 先在 `_relink_prepared_candidates(...)` 蒐集 `RelinkedTrackCandidate` list，再重新拆 4 條 list 餵進 `lifecycle_merger.resolve_many_packed(...)`。

本文件記錄把這段合併成單一 native identity resolve pass 的設計草稿，作為下一輪實作的對齊基準。

---

## 1. 目標 / 非目標

### 目標
- 把 `_relink_prepared_candidates(...)` 與 `lifecycle_merger.resolve_many_packed(...)` 之間的 Python 縫合（建 `relinked_ids` list、再重新拼 4 條 list 進 lifecycle）合併成單一 native call，runner 看到的就是 `prepared → resolved`。
- 為後續真正的 C++ identity pass 鋪好「介面 + parity test」這條路，下一步換 native 實作不再動 runner。

### 非目標
- 不更動 relink / lifecycle 的演算法語意（包含 reciprocal margin、Mahalanobis、EMA、stats 計數順序）。
- 不把 Python `TrackletLifecycleMerger` 「順手」改寫成 C++（拆成另一個 phase；見 §6）。
- 不動 `_finalize_frame_side_effects(...)` 內的 bank inject / dynamic reid 等 frame-end 流程。

---

## 2. 介面

新增 facade（不擴張既有兩個 class 的責任）：

```python
class IdentityResolver:
    """Compose semantic relink + tracklet lifecycle merge into one call."""

    def __init__(
        self,
        relinker: Any | None,                  # SemanticRelinker (C++ or Python) or None
        lifecycle_merger: TrackletLifecycleMerger,
    ) -> None: ...

    def resolve_pass(
        self,
        local_ids: list[int],
        embeddings: list[torch.Tensor | None],
        boxes: list[tuple[float, float, float, float]],
        scores: list[float],
        *,
        frame_id: int,
        frame_w: int,
        frame_h: int,
    ) -> list[int]:
        """Return resolved (output) IDs in input order. No attached candidate
        metadata; runner still owns PreparedTrackCandidate → ResolvedTrack zip."""
```

對應 C++ 端 (Phase B 才會出現)：

```cpp
// include/tracking/identity_resolver.hpp
class IdentityResolver {
public:
    IdentityResolver(SemanticRelinker* relinker,
                     TrackletLifecycleMerger* lifecycle);  // both raw, owned by Python

    py::list resolve_pass(py::sequence local_ids,
                          py::sequence embeddings,
                          py::sequence boxes,
                          py::sequence scores,
                          int frame_id,
                          int frame_w,
                          int frame_h);
};
```

**關鍵設計**：facade 不持有 state，只組合既有的兩個 stage。所有 alias / features / buffers / states / stats 仍歸各自 stage 所有。

---

## 3. 行為對照表（每 candidate）

| 步驟 | 現況 (`_resolve_frame_tracks`) | 新 facade (`resolve_pass`) |
|---|---|---|
| relink | `relinker.resolve_many_packed(local_ids, embeddings, boxes, scores, frame_id, w, h)` | 內部 inline；同一支 relinker call signature |
| 中介 | Python list `relinked_candidates: list[RelinkedTrackCandidate]`, 重組 4 條 list | **無**；relink_id 直接餵下一階段 |
| lifecycle 的 embedding 入參 | `candidate.embedding if relinker else None` | 同行為：`relinker is None` 時整個 facade 不會被建（runner 走舊路徑），所以這裡永遠是 `embedding` |
| lifecycle | `lifecycle_merger.resolve_many_packed(relinked_ids, boxes, scores, embeddings, frame_id, w, h)` | 內部 inline；同一支 lifecycle call signature |
| 副作用 | relinker.{alias,features,buffers,last_seen,last_boxes,stats,accept_*}; lifecycle.{alias,states,stats} | 完全相同（stats 計數順序 byte-equal） |

**保留的不變式**
- 兩個 stage 各自仍有自己的 `assigned` set（per-pass scoped）。
- 同一筆 candidate 內 relink 先 commit、再餵 lifecycle，順序與現況一致。
- relinker 為 `None` 時不啟用 facade（保留現有 short-circuit at runner.py:742）。

---

## 4. Runner 整合

`_resolve_frame_tracks` 變成優先順序：

```python
identity_resolver = getattr(..., "identity_resolver", None)  # injected at construction
if identity_resolver is not None:
    resolved_ids = identity_resolver.resolve_pass(
        [c.local_track_id for c in prepared_candidates],
        [c.embedding       for c in prepared_candidates],
        [c.box             for c in prepared_candidates],
        [c.score           for c in prepared_candidates],
        frame_id=frame_id, frame_w=frame_w, frame_h=frame_h,
    )
else:
    # 現有 _relink_prepared_candidates → lifecycle.resolve_many_packed/many/resolve fallback chain 不動
    ...
return [ResolvedTrack(...) for c, rid in zip(prepared_candidates, resolved_ids)]
```

`_relink_prepared_candidates` 整段就成為 facade 不可用時的 fallback；Python `IdentityResolver` 內部 call 的還是現有 `resolve_many_packed`，所以 fallback 行為等於今天的 path。

---

## 5. Fallback 鏈

由上而下，第一個可用的勝出：

1. **Native `IdentityResolver`** (Phase B)
   需要 C++ relinker **且** C++ lifecycle merger 都存在 → 單一 native call。
2. **Python `IdentityResolver`** (Phase A)
   存在但只是組合 `relinker.resolve_many_packed` + `lifecycle.resolve_many_packed`；省掉 runner 的 Python list 拼接，但本質還在 Python。
3. **既有 fallback chain**（runner 內 `resolve_many_packed → resolve_many → resolve` 三段）—不刪。

**為什麼 Phase A 還是值得做**：把 facade boundary 先在 runner 裡定下來，後續換 native 實作不再動 runner，也讓 parity test 有 stable 介面。

---

## 6. 分階段落地

### Phase A — Python facade + parity test ✅ 完成（2026-04-30）

落地檔案：
- `perception/eval/relink.py`：新增 `IdentityResolver`（純 Python composer，兩段 fallback chain：`resolve_many_packed → resolve_many → resolve`）。
- `perception/eval/runner.py`：import `IdentityResolver`；relinker 建完後建 `identity_resolver`；`_resolve_frame_tracks` 加 `identity_resolver` 參數，有值時走 `resolve_pass`。
- `tests/test_identity_resolver_parity.py`（新）：5 個 parity test，固定 seed 比對 IDs / alias / stats。

### Phase B — C++ lifecycle merger + `IdentityResolver` ✅ 完成（2026-04-30）

落地檔案：
- `src/tracking/tracker_gpu_python.cpp`：
  - `SemanticRelinkerCpp::resolve_cpp()`：接受 C++ types + `std::unordered_set<int>`，供 `IdentityResolverCpp` 直接呼叫（不過 pybind 邊界）。
  - `TrackletLifecycleMergerCpp`：忠實對譯 Python `TrackletLifecycleMerger`（IoU / center gate / cosine / age / EMA；stats 計數順序相同）。完整 Python-facing API（`resolve_many_packed` / `resolve_many` / `prune` / `alias` / `stats` / `report`）。
  - `IdentityResolverCpp`：inputs 解析一次，stage 間共用 C++ vectors，無 Python list 中介。pybind bindings。
- `perception/eval/runner.py`：module-level try-import C++ 兩類；lifecycle 優先用 C++ class；identity_resolver 在 C++ relinker + C++ lifecycle 時升級至 C++ resolver。
- `tests/test_phase_b_parity.py`（新）：8 個 parity test，C++ vs Python byte-equal（IDs / alias / stats）。

### Phase C — 清理 ✅ 完成（2026-04-30）

落地檔案：
- `perception/eval/runner.py`：刪除 `RelinkedTrackCandidate`、`_relink_prepared_candidates`；`_resolve_frame_tracks` 移除 `relinker` 參數；lifecycle-only fallback（`identity_resolver is None`）直接走 `lifecycle.resolve_many_packed`，不再過 relink 中介。
- `tests/test_runner_batch_helpers.py`：移除 `_StubBatchResolveRelinker` 及兩個已失效測試；`test_resolve_frame_tracks` 改走 `IdentityResolver`。
- `tests/test_identity_resolver_parity.py`：legacy 比較測試改為「`_resolve_frame_tracks` via `IdentityResolver`」vs「直接 `resolve_pass`」。

---

## 7. 實際落地與設計草稿的差異

1. **Phase B 未建獨立 .hpp/.cpp**：`TrackletLifecycleMergerCpp` 與 `IdentityResolverCpp` 直接加在 `tracker_gpu_python.cpp` 的匿名 namespace，與 `SemanticRelinkerCpp` 同一檔案。符合既有慣例，避免多餘的 CMakeLists 改動。
2. **Phase C 比預估量小**：實際刪除約 100 行（`_relink_prepared_candidates` ~90 行 + `RelinkedTrackCandidate` ~5 行 + 呼叫端 1 行）；相關測試更新約 60 行 diff。
3. **風險 3（EMA on `embedding=None`）已處理**：`TrackletLifecycleMergerCpp::resolve_cpp()` 的 EMA 分支與 Python 版本行為一致，parity test 覆蓋。
4. **風險 4（pybind 邊界）已處理**：`IdentityResolverCpp::resolve_pass()` 解析 Python inputs 一次後，透過 `resolve_cpp()` 直接呼叫 C++ 內部實作，兩個 stage 之間無 Python list 中介。

---

## 8. 驗證結果

| 測試組 | 數量 | 說明 |
|---|---|---|
| `test_phase_b_parity` | 8 | C++ vs Python byte-equal：IDs / alias / stats；含 None emb、disabled、relink match |
| `test_identity_resolver_parity` | 5 | `_resolve_frame_tracks` via resolver vs 直接 `resolve_pass`；lifecycle-only no-op |
| `test_runner_batch_helpers` + `test_runner_materialization` + `test_relink_motion_candidates` | 14 | M4-b 原有 unit tests（移除 2 個已失效）|
| `test_e2e` | 3 | 完整 pipeline smoke |
| **總計** | **30 passed** | 全綠 |

---

## 9. 實際 footprint

- Phase A：153 行 Python 新增 + 25 行 runner 改動 + 一支 parity test（5 tests）。
- Phase B：~600 行 C++ 新增（`resolve_cpp` + `TrackletLifecycleMergerCpp` + `IdentityResolverCpp` + pybind bindings）+ runner 15 行 + 一支 parity test（8 tests）。
- Phase C：~100 行刪除 + ~60 行測試更新。

---

## 10. 後續方向

- **M4-c**：評估是否值得消除 `post_count` / result-count 單點回讀（目前仍有一次 D2H sync）。
- **長線**：runner 熱路徑的 Python overhead 已大幅收斂；下一個可量測瓶頸應透過 profiling 確認再決定方向。
