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
- **已完成方案 C offline identity benchmark（2026-07-02，見下）— ImageNet 權重即超過既有 ceiling，gate 通過。**
- 尚未匯出 ONNX。
- 尚未建 TensorRT engine。
- 尚未接入 `TRTFeatureExtractor` / evaluator（benchmark 走 `reid_id_benchmark.py` 的 timm eager 路徑）。
- 尚未做 ReID 任務微調。

## 方案 C 結果：offline identity benchmark（2026-07-02）

`reid_id_benchmark.py` 新增 timm eager 路徑（`--model-type mobilenetv4_*`，讀 manifest 的
mean/std 與 input size，`--resize bicubic` 對齊 pretrained cfg）。MOT17 train 7×SDP GT
crops，per-id 20。同日同協定重跑 osnet 作 control：gap 31-60 54.6% / 61-120 30.3% /
121+ 10.2%，與 appearance_ceiling_mot17.md 記載（54/36/10）一致 → 協定可比。

| 模型 | gap 31-60 | gap 61-120 | gap 121+ | h 0-50px | d' 範圍 | AUC 範圍 |
|---|---:|---:|---:|---:|---|---|
| osnet（control，同日） | 54.6% | 30.3% | 10.2% | 48.0% | 0.64–0.97 | 72.7–82.5% |
| transreid（既有 ceiling，文檔值） | 63% | 38% | 13% | — | ~1.0 | — |
| **mobilenetv4_conv_small** | **71.8%** | **52.0%** | **22.9%** | **77.4%** | 1.00–1.95 | 76.2–90.8% |
| mobilenetv4_conv_small_050 | 70.0% | 51.7% | 23.2% | 76.7% | 1.02–1.91 | 76.1–90.2% |
| mobilenetv4_conv_medium | 69.7% | 44.5% | 20.8% | 74.6% | 0.90–1.97 | 74.6–91.1% |

判讀：

- **gate 通過**：純 ImageNet 權重（原本預期不過關的 sanity check）即全面超過既有
  appearance ceiling——gap 31+ 全分層明顯高於 transreid/osnet。
- **小框分層是最大驚喜**：h 0-50px rank-1 77.4% vs osnet 48.0%，正是「小框、遮擋」
  這個解鎖條件的族群。
- 容量非單調：conv_small > conv_medium（8.4M 參數反而略差），conv_small_050（0.96M）
  幾乎不掉——首選部署候選仍是 `conv_small`，`_050` 是 latency 備胎。
- 尚未達「好 ReID」絕對標準（d'>2、AUC>95%）；gap 121+ 仍僅 ~23%，長 gap relink
  的 base-rate 牆是否可破仍待 domain fine-tune 與 tracking A/B 驗證。
- intra−inter cosine gap 0.15–0.22（既有模型 ~0.03–0.19），分佈重疊明顯縮小。

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

## Domain fine-tune 結果（2026-07-02）

`scripts/train/finetune_mobilenetv4_reid.py`：full-backbone BoT 配方（BNNeck + CE ls0.1 +
batch-hard triplet m0.3，PK 24×4，AdamW bb 1e-4 / head 3.5e-4，cosine+warmup，bf16，
224×224 stretch 對齊 eval 幾何）。訓練身份與 MOT17 不相交（leak-free，同
`reid_domain_probe.py` 協定）；eval = 同一 `reid_id_benchmark` 協定，直接可比。

兩臂對照（MOT17 rank-1）：

| arm | 資料 | gap 31-60 | gap 61-120 | gap 121+ | gap31+ |
|---|---|---:|---:|---:|---:|
| ImageNet init（參考） | — | 71.8% | 52.0% | 22.9% | — |
| mixed（60ep, final=best） | Market1501 + MOT20/DanceTrack/SportsMOT crops（64.6k crops / 4,002 ids） | 87.2% | 72.9% | 46.0% | 73.3% |
| **visclean（60ep, best=ep45）** | mixed + 去汙染（vis≥0.3 + occ-cov 0.4；62.7k / 3,903 ids） | **89.0%** | **75.1%** | 43.8%（峰值 ep30 46.8%） | **74.1%** |
| Market-only（240ep） | Market1501（12.9k / 751 ids） | 61.4% | 34.6% | 8.9% | 27.5%（峰值 ep120 42.5% 後單調退化） |

判讀：

- **增益全部來自 MOT-domain crops**。Market-only 不只無效、還把 ImageNet 特徵的泛化性
  洗掉（長 gap 121+ 22.9→8.9%），重演 siglip2_reid「Market 81% mAP → MOT17 不轉移」。
- mixed 臂長 gap 121+ 46.0% = ImageNet init 的 2 倍、舊 ceiling（13%）的 3.5 倍——
  正是過去 birth-relink 全滅（look-alike 誤接）的區段。
- **去汙染消融小勝且方向確認**（`--vis-min 0.3 --occ-cov 0.4`：MOT20 標註 vis 過濾
  + 幾何 front-box coverage 過濾，丟 278.9k vis<0.3 + 339.4k 幾何被遮框，後者覆蓋
  無 vis 標註的 DanceTrack/SportsMOT；per-id 取樣改取乾淨幀故池幾乎沒縮）：
  gap31+ 74.1% vs 混合 73.3%，增益集中中 gap（31-60 +1.8 / 61-120 +2.2），長 gap
  持平。標籤更乾淨、收斂略慢（ep15 落後、ep30 反超）。**部署首選 checkpoint**。
- checkpoints：`runs/reid_mnv4_ft_visclean/best.ckpt`（ep45，首選）、
  `runs/reid_mnv4_ft/best.ckpt`（mixed）；用
  `reid_id_benchmark.py --ft-checkpoint <ckpt>` 可獨立重跑。
- Runtime 側設計（整合時）——核心是 **birth-time 汙染**（User 定調）：新軌常誕生在
  「從遮擋者身後走出」的瞬間，此時 crop 像素大多是前面那個人，birth embedding 一進
  bank 就把身份汙染定死，後續 relink 全跟著錯。規則：
  1. **birth gate**：新軌誕生時若幾何判定被遮（存在 foot 更低的框 coverage > τ，
     與訓練側 `--occ-cov` 同一判定式），**不 seed 外觀特徵**——直到第一個乾淨幀才
     建立 reference；在那之前該軌只走幾何關聯，不作 relink query 也不作 match target。
  2. **occlusion freeze**：既有軌被遮期間凍結 bank 更新（不 EMA、不 inject）。
  3. 判定源用 tracker 既有幾何 occ_state（IoU + foot-gap,occ_event 92-100% 準）；
     mamba head x_cls probe（AUC 0.836,弱於幾何）僅作輔助。
  4. 訓練側與 runtime 用**同一個 coverage 判定**：模型沒學過髒 crop,部署也永遠
     不餵髒 crop——train/deploy 分佈一致。
  - 現成接口：Phase-3 bank inject gate / 高品質 tier、need_reid 觸發器
    （birth_death_lost_min）可直接掛。

## 當前決策

**方案 C 已完成且 gate 通過（2026-07-02）**：ImageNet 權重的 conv_small 即超過既有
appearance ceiling（gap 31+ 全分層、小框分層尤其顯著）。

下一步依原順序進入方案 A 評估，但有兩個前置判斷：

1. 先決定是否直接用 ImageNet 權重做一次 tracking A/B（成本低、回答「offline 可分性
   增益能否轉移到 relink/association」），或先做 Market1501+MOT crops fine-tune 再接線。
2. 長 gap 121+ 仍僅 ~23%，過往 birth-relink 失敗主因（長 gap look-alike 誤接）未必
   解除；tracking A/B 應優先看 relink 作用區（gap 31+）而非整體 IDF1。
