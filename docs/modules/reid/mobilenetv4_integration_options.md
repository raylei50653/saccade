# MobileNetV4 ReID Integration Options

> 狀態：規劃文檔，尚未進行模型接線、ONNX 匯出、TensorRT engine 建置或 evaluator 整合。

## 背景

MobileNetV4 是 Google 2024 年提出的新一代 MobileNet。相較 MobileNetV3，V4 的核心變化是：

- **UIB (Universal Inverted Bottleneck)**：把 inverted bottleneck、ConvNeXt-like block、FFN-like block、ExtraDW 變體納入同一個搜尋空間。
- **Mobile MQA**：Hybrid 版本使用針對行動/邊緣裝置設計的 multi-query attention。
- **跨裝置效率目標**：論文主張在 CPU、GPU、DSP、EdgeTPU 等 mobile ecosystem 上取得更通用的 latency/accuracy trade-off。

對本專案而言，MobileNetV4 的定位不是直接替換 detector，而是作為 **appearance / ReID 特徵候選 backbone**，用來驗證「MOT/crowd 域訓練的輕量 embedding」是否能突破既有 ReID NO-GO 上限。

## 外部資源

| 類別 | 資源 | 備註 |
|---|---|---|
| 論文 | <https://arxiv.org/abs/2404.10518> | MobileNetV4: Universal Models for the Mobile Ecosystem |
| 官方架構 | <https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py> | TensorFlow Model Garden，含 `MNV4*` block specs |
| PyTorch 實作 | <https://github.com/huggingface/pytorch-image-models> | `timm` 已提供 MobileNetV4 variants |
| timm 說明 | <https://huggingface.co/blog/rwightman/mobilenetv4> | UIB / Mobile MQA / PyTorch implementation 說明 |
| 權重集合 | <https://huggingface.co/collections/timm/mobilenetv4-pretrained-weights> | 實務上最方便取得的 pretrained weights |

## 本地現況

`pyproject.toml` 已包含 `timm>=1.0.26`，本地 `uv` 環境可列出 MobileNetV4 variants。

已下載並保存三個候選權重到 `models/mobilenetv4/`：

| 本地檔案 | resolved timm/HF 權重 | Params | Input | Normalize |
|---|---:|---:|---:|---|
| `models/mobilenetv4/mobilenetv4_conv_small.pth` | `timm/mobilenetv4_conv_small.e2400_r224_in1k` | 2,493,024 | 224x224 | ImageNet mean/std |
| `models/mobilenetv4/mobilenetv4_conv_small_050.pth` | `timm/mobilenetv4_conv_small_050.e3000_r224_in1k` | 957,952 | 224x224 | mean/std = 0.5/0.5 |
| `models/mobilenetv4/mobilenetv4_conv_medium.pth` | `timm/mobilenetv4_conv_medium.e500_r256_in1k` | 8,434,512 | 256x256 | ImageNet mean/std |

驗證狀態：

- 已用 `timm.create_model(..., pretrained=False, num_classes=0)` strict load 三份 `.pth`。
- 三者皆 `missing=0`、`unexpected=0`。
- 尚未匯出 ONNX。
- 尚未建 TensorRT engine。
- 尚未接入 `TRTFeatureExtractor` / evaluator。
- 尚未做 ReID 任務微調或 MOT-domain benchmark。

## 與現有 ReID 狀態的關係

現有 ReID 模組目前不是缺一個更小的 classification backbone，而是卡在 **MOT17 appearance embedding 可分性上限**：

- 既有調查顯示現成 appearance 模型、ROI FPN embedding、OSNet hard pool、Mamba detection features 等路線都未能提供足夠 ID discriminability。
- ReID 方向的解鎖條件是取得或訓練 **MOT/crowd 域、小框、遮擋魯棒** 的 ReID 特徵。
- 因此 MobileNetV4 的價值在於「可訓練、可部署、輕量」的候選 backbone，而不是直接套 ImageNet embedding 後期待改善 tracking。

## 整合方案 A：新 crop-based ReID engine

把 MobileNetV4 當作現有 `osnet` / `fastreid` / `siglip2_reid` 類似的獨立 crop embedding extractor。

預期路徑：

1. 建 MobileNetV4 ReID wrapper：`backbone -> pooling -> BNNeck/projection -> L2 normalize`。
2. 使用 `conv_small_050`、`conv_small`、`conv_medium` 做三個容量點。
3. 訓練或微調在 Market1501 + MOT/crowd crops。
4. 匯出 ONNX。
5. 建 TensorRT engine。
6. 在 evaluator 中新增 `reid_model=mobilenetv4_*`。

優點：

- 架構邊界清楚，和現有 external ReID model 一致。
- 容易與 `osnet`、`fastreid`、`siglip2_reid` 做公平比較。
- Conv-only V4 比 Hybrid V4 更適合第一輪 TensorRT 驗證。

風險：

- 會走 crop/resize/extract 路徑，runtime 未必打得過 `fpn_trained`。
- ImageNet 權重不等於 ReID 權重，必須做 domain training。
- 需要確認 `conv_small_050` 的 0.5 normalization 是否會影響既有 crop preprocessing 慣例。

適合用途：

- 第一個可部署 ablation。
- 對照 `osnet` / `fastreid` 的速度與 IDF1/HOTA。

## 整合方案 B：MobileNetV4 作為 ReID teacher

MobileNetV4 不直接進 runtime，而是作為 teacher，蒸餾現有 FPN/JDE embedding head。

預期路徑：

1. 用 MobileNetV4 ReID wrapper 在 crop 上產生 teacher embedding。
2. 訓練現有 FPN projection / JDE head 對齊 teacher。
3. runtime 繼續使用 detector FPN feature，不新增 crop backbone。

優點：

- 保留現有 `fpn_trained` 的低 latency 優勢。
- 避免 runtime crop-based extractor 增加同步與 batching 成本。
- 如果 teacher 有足夠 ReID discriminability，可能把收益轉移到現有 pipeline。

風險：

- 如果 teacher 本身未過 `reid_id_benchmark.py`，蒸餾只會複製弱訊號。
- FPN feature 的個體可分性曾經被驗證偏弱，蒸餾可能仍受 backbone feature ceiling 限制。

適合用途：

- MobileNetV4 微調後，作為第二階段降 latency 方案。
- 不適合第一步，因為會混淆「teacher 是否有效」與「student 是否能學到」兩個問題。

## 整合方案 C：offline benchmark / relink diagnostic only

先不接 online evaluator，只用 MobileNetV4 embeddings 做離線身份可分性評估。

預期路徑：

1. 從 MOT17 / DanceTrack / SportsMOT 裁出同 ID、不同時間 gap 的 crops。
2. 用三個 V4 權重各自抽 embedding。
3. 評估 rank-1、positive/negative cosine gap、gap 31+ / gap 80+ 分層。
4. 只在明顯超過既有 appearance ceiling 時，才進入方案 A 或 B。

優點：

- 成本最低，避免早早改 evaluator。
- 能直接回答「MobileNetV4 是否有 ReID headroom」。
- 符合現有 ReID NO-GO 的解鎖條件。

風險：

- 只能評估特徵可分性，不能直接代表 online tracking 效果。
- 如果只用 ImageNet 權重，預期結果大概率不足；應把它視為 sanity check，不是最終結論。

適合用途：

- 第一個應執行的驗證步驟。
- 也是是否值得投入訓練/部署的 gate。

## 整合方案 D：Hybrid V4 延後評估

`mobilenetv4_hybrid_*` 引入 Mobile MQA。這類模型可能有更好 accuracy/capacity，但第一輪不建議納入。

原因：

- Attention / MQA 對 ONNX/TensorRT export 風險較高。
- ReID runtime 需要穩定的 crop batching latency，第一輪應優先排除部署變因。
- 如果 conv-only V4 已經無法通過 identity benchmark，Hybrid 版本也不應直接跳進 online pipeline。

可以重啟的條件：

- `conv_small` 或 `conv_medium` 經 domain fine-tune 後已證明有效。
- 需要更高 capacity，且可以接受較高 export / TensorRT 驗證成本。

## 建議順序

1. **C：offline identity benchmark**  
   先用現有 `.pth` 跑特徵可分性 sanity check。若 ImageNet 權重不過關，這是預期結果。

2. **A：crop-based ReID engine prototype**  
   僅在 domain fine-tuned V4 明顯改善 rank-1 / cosine gap 後再做。第一個部署候選用 `mobilenetv4_conv_small`。

3. **B：distill 到 FPN/JDE head**  
   當 V4 teacher 有效後，再嘗試把 runtime 成本降回 detector FPN 路線。

4. **D：Hybrid V4**  
   保留為後續高 capacity 選項，不納入第一輪。

## 驗收指標

進入 online tracking 前，至少需要通過：

- `reid_id_benchmark.py`：gap 31+ rank-1 明顯高於既有 ceiling。
- Positive/negative cosine gap：不能只靠閾值微調產生表面收益。
- 小框/遮擋分層：不能只在 200px+ 清晰框有效。
- Latency：crop + preprocess + TRT enqueue + normalize 必須與現有 `osnet` / `fpn_trained` 比較。
- Tracking：MOT17 SDP/DPM/FRCNN 分 detector source 比較 IDF1、HOTA、IDs、FPS。

## 當前決策

**不建議直接把 MobileNetV4 接入 online evaluator。**

合理的下一步是把它視為「MOT-domain ReID backbone 候選」，先做 offline identity benchmark；只有在可分性超過現有 appearance ceiling 後，才值得投入 ONNX/TensorRT 與 evaluator 接線。
