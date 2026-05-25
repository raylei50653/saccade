# Saccade vs Ultralytics: 640 / 960 Operating Points

日期：2026-05-13

## 目的

把目前常用的四個 tracking operating point 放到同一張表，回答兩個問題：

1. `640` 是否能有效壓低 `FP`
2. 壓低 `FP` 的代價是否只是把 `FN` / `Recall` / `IDF1` 一起打壞

比較對象：

- `Saccade speed preset 960`
- `Saccade speed preset 640`
- `Ultralytics yolo26s.pt + BoT-SORT 960`
- `Ultralytics yolo26s.pt + BoT-SORT 640`

資料集：

- `MOT17 train`
- `SDP` 7 sequences：`02 / 04 / 05 / 09 / 10 / 11 / 13`

---

## 配置

### Saccade 960

```bash
uv run python scripts/eval/mot17.py \
  --preset speed \
  --detector SDP \
  --output results/MOT17_preset_speed_verify
```

### Saccade 640

```bash
uv run python scripts/eval/mot17.py \
  --preset speed \
  --detector SDP \
  --tiling native_640 \
  --output results/MOT17_preset_speed_verify_640
```

備註：

- `native_640` 走 Saccade 原本的 detection / tracking pipeline
- engine 會從 `yolo26s_960_batch1.engine` 自動切到 `yolo26s_batch4.engine`

### Ultralytics 960

```bash
uv run python scripts/eval/ultralytics_official_mot17.py \
  --model models/yolo/yolo26s.pt \
  --output results/MOT17_ultralytics_eval_960_conf025 \
  --imgsz 960 \
  --conf 0.25 \
  --detector SDP
```

### Ultralytics 640

```bash
uv run python scripts/eval/ultralytics_official_mot17.py \
  --model models/yolo/yolo26s.pt \
  --output results/MOT17_ultralytics_eval_640_conf025 \
  --imgsz 640 \
  --conf 0.25 \
  --detector SDP
```

---

## Overall Results

| Method | IDF1 | MOTA | Recall | Precision | IDs | FP | FN | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Saccade 960 | 51.2% | 40.8% | 53.0% | 81.9% | 539 | 13,139 | 52,758 | 110.01 |
| Saccade 640 | 43.4% | 34.8% | 42.2% | 85.7% | 423 | 7,877 | 64,942 | 132.07 |
| Ultralytics 960 | 50.3% | 38.8% | 48.6% | 83.9% | 521 | 10,437 | 57,719 | ~27-32 |
| Ultralytics 640 | 44.1% | 32.4% | 38.4% | 87.1% | 366 | 6,365 | 69,203 | ~30-33 |

---

## Readout

### 1. `640` 的確能壓 `FP`

兩條系統都一樣：

- `Saccade`: `13,139 -> 7,877`（`-5,262` FP）
- `Ultralytics`: `10,437 -> 6,365`（`-4,072` FP）

這不是偶然，方向很一致。

### 2. `640` 不是免費收益

兩條系統都用相同方式付款：

- `Recall` 明顯下降
- `FN` 明顯上升
- `IDF1 / MOTA` 明顯變差

具體數字：

- `Saccade`: `Recall 53.0% -> 42.2%`, `IDF1 51.2% -> 43.4%`, `MOTA 40.8% -> 34.8%`
- `Ultralytics`: `Recall 48.6% -> 38.4%`, `IDF1 50.3% -> 44.1%`, `MOTA 38.8% -> 32.4%`

### 3. `Saccade 960` 仍是整體最強點

若目標是整體 tracking 品質，而不是單純壓低誤報：

- `Saccade 960` 是目前最佳 operating point
- `Ultralytics 960` 是最接近的保守 baseline
- `640` 版本比較像「precision / throughput 模式」，不適合作為主預設

### 4. `Saccade 640` 仍優於 `Ultralytics 640`

即使兩者都切成更保守的 `640`：

- `Saccade 640`: `IDF1 43.4%`, `MOTA 34.8%`
- `Ultralytics 640`: `IDF1 44.1%`, `MOTA 32.4%`

兩者接近，但 `Saccade 640` 仍保留較好的 `MOTA`，而 `Ultralytics 640` 仍是更保守的高-precision 端點。

### 5. `Saccade 640` 的價值在於 profile，不在於 default

它有兩個明確用途：

- 當作「低 FP / 高 FPS」的備選 profile
- 幫助驗證目前 `FP` 問題是否主要來自 detector operating point，而不是 tracker 後段

但它不應升格為 default，因為代價太大：

- `IDF1 -7.8pp`
- `MOTA -6.1pp`
- `Recall -10.8pp`

---

## 結論

這輪結果支持一個很穩定的判斷：

> `FP` 可以靠更保守的 detector operating point 顯著下降，但目前看到的每一條路徑，都是在用 `FN / Recall / IDF1 / MOTA` 付款。

因此，若目標是改善 Saccade 的 `FP` 問題，正確方向不是直接把主預設改成 `640`，而是：

- 以 `Saccade 960` 為主線
- 以 `Ultralytics 960` 當保守 baseline
- 嘗試縮小 `Precision / FP` 差距，但不能接受 `Recall / IDF1 / MOTA` 大幅退化

換句話說：

- `640` 是有用的診斷點
- 不是目前該採用的 default 點
