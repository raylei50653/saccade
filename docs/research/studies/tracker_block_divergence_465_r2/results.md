# Tracker block divergence r2：結果（探索級）

<!-- evidence-tier: exploratory -->

> **不可引用**：exploratory，任何 formal chain 都不得引用（§20.11.2）。要據此做設計決定，必須另寫 formal 宣告並重新執行。
> 採用的 attempt：[attempts/001](attempts/001/attempt.json)（`valid`）。freeze commit `3cb8f74d`，tag `freeze/tracker_block_divergence_465_r2/1`（object `cbe17eaf`）。raw record `results/tracker_block_divergence_465_r2/20261002T133546Z`（`MANIFEST.json` SHA-256 `cd679d6df7d6bf2bbe2f5226b46f767da21569acce54dc88517712ed749a0cbe`）。完整數字見 [report.md](attempts/001/report.md) 與 `result.json`。

## Validity

依宣告順序，`V_RECORD` → `V_REPLAY` → `V_REPEAT` 全部成立；runner 在任一條件失敗時會停止並記為 invalid，本 attempt 沒有停止。canonical form 只用在 `V_REPEAT`；terminal 與以下所有數字讀的是 raw dump（`CND` 列保留寫出時的順序）。

## Terminal

**`FIRST_DIVERGENCE_ASSOCIATION`**：`R_T` 的 7 個序列，第一個結構分歧全部在 A（7/7，門檻 5/7）。

| 序列 | 第一個分歧 (frame, block) | A 第一個 frame（分歧 frame 數；未定義） | P 第一個 frame（數） | E 第一個 frame（數） | r2 stage f\* | 第一個分歧到 f\* |
|:--|:--|:--|:--|:--|--:|--:|
| 02 | 222, A | 222（6；309） | 283（310） | 283（307） | 283 | 61 |
| 04 | 2, A | 2（14；1035） | 15（1036） | 3（1048） | 3 | 1 |
| 05 | 15, A | 15（7；36） | 15（38） | 15（12） | 15 | 0 |
| 09 | 189, A | 189（3；208） | 189（209） | 319（207） | 319 | 130 |
| 10 | 3, A | 3（1；651） | 3（652） | 6（648） | 6 | 3 |
| 11 | 3, A | 3（9；64） | 64（66） | 49（35） | 49 | 46 |
| 13 | 7, A | 7（1；743） | 7（744） | 14（736） | 14 | 7 |

- 05、09、10、13 的第一個分歧 frame 上，A 與 P（05 另含 E）同時分歧；依宣告 block 順序記為 A。
- E 的第一個分歧 7/7 等於 r2 stage study 的 f\*（`V_REPLAY` 與同一個比較器的推論成立）。

terminal 只陳述「在凍結的結構定義下，第一個結構分歧出現在哪個 block」，不陳述成因，也不陳述這個分歧導致了 f\* 或 IDs +21。

## 只報告

- **第一個分歧當下的 `tracker_input`**：04、05、09、10、13 兩邊結構相同、位元不同；02、11 結構已不同。列數兩邊都一致。之前的 `tracker_input` 結構分歧 frame 數：02=10、09=2、13=1，其餘 0。這只描述分歧邊界上的輸入，不判斷上游機制。
- **第一個分歧當下的 A**：05、10、13 的分歧 track 兩邊 candidate 集合結構相同（同樣的 candidate、不同的指派）；02、04、09、11 的 candidate 集合不同。逐 track 的 id／state／age／`trk_to_det`／candidate 與 cost／predict 後的框見 `result.json`。
- **`R_E`（容差內基準）**：A 在 02 f469、09 f189（同 frame 另有 P 分歧）、13 f646 各分歧 1 個 frame；E 在 7 個序列都沒有分歧。依宣告 §5，這表示單一 frame 的 A 分歧不是 T 特有的；本文件不對這些分歧做成因解讀。
- **`GMC`**：所有 arm、所有序列，兩邊 `GMC` 列不同的 frame 數都是 0。

## 與 r1 的關係

r1 維持 **`INVALID_V_REPEAT / NO_TERMINAL`** 永久結案（[r1 results](../tracker_block_divergence_465/results.md)）。r1 attempt 001 不依 r2 的 repeat 規則重判，它的任何觀測都不是 r2 的 evidence；r1 attempt 002／003 沒有執行。
