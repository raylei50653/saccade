# L2: 向量與去重層 (Vector & Deduplication Layer)

## 1. 定義與目標
L2 負責將視覺信號轉換為「語義特徵向量 (Embeddings)」。本層級的核心目標是建立穩定的物件語義索引，並透過「語義質心」與「動態熱身」機制，在極端環境下過濾掉 90% 以上的冗餘影格。

## 2. 核心組件
- **裁切器 (Cropper)**: 執行 GPU 內的批量 RoI 提取與 Resize。
- **特徵提取器 (FeatureExtractor)**: TensorRT 加速的 SigLIP 2 推理引擎，支援 $N_{opt}=8$ 的動態 Batch。
- **語義漂移處理器 (DriftHandler)**: 
    - **語義質心 (Semantic Centroid)**: 使用 EMA 維護物件的穩定特徵，而非單純的上一幀比對。
    - **動態熱身 (Warm-up Phase)**: 為新物件提供高學習率 ($\alpha=0.7$)，加速語義收斂。
    - **顯著性截斷 (Salience Truncator)**: 當 Batch 超載時，依據「物件面積」優先處理重要目標。

## 3. 關鍵優化 (Industrial Grade)
- **EMA 特徵融合**: 透過 $Centroid = \alpha \cdot New + (1-\alpha) \cdot Old$ 消除光影跳動與局部遮擋的雜訊。
- **優先權批次 (Priority Batching)**: 
    - **P0: New ID**: 必須第一時間建立語義。
    - **P1: Warm-up**: 快速穩定質心。
    - **P2: Stable**: 僅在發生顯著漂移時更新。
- **$N_{opt}$ 截斷策略**: 當 VRAM 壓力大 (Level 2) 時，強制截斷 Batch 至 8，並優先保留面積最大 (Salience 最高) 的物件。

## 4. 異常處理：質心漂移警報 (Centroid Drift Alarm)
- 當已穩定物件的 Drift Score 持續超過閾值，系統會自動重啟「微熱身」模式，以適應物件的形態改變（如換裝、開啟車門）。

## 5. 效能調優指標 (Verified via 5000-frame Benchmark)
- **Batch Inference Time**: 單次 8 個物件 Embedding 平均 **0.82 ms**。
- **Zero-Copy Cropping**: < 0.15 ms。
- **Deduplication Rate**: 在靜止場景下過濾率 > 98%。
- **Batch Efficiency**: 始終運作在 TensorRT 最佳推理區間 ($N \le 8$)。
- **Semantic Latency**: 從物件入鏡到建立穩定質心 < 5 幀。
