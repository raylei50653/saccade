# **視覺變換器中懶惰聚合行為的頻域診斷與 LaSt-ViT 技術路線在 ADR 016 架構下的整合分析報告**

## **1\. 視覺變換器的架構缺陷與偽影起源：從 DINO 到 DINOv2 的演進挑戰**

視覺變換器（Vision Transformer, ViT）自問世以來，憑藉其在大規模數據集上的卓越擴展性與通用特徵表示能力，已成為當前計算機視覺領域的基礎架構核心。然而，隨著模型規模與訓練數據量的激增，研究人員在多種監督範式下均觀察到特徵圖中存在顯著的「偽影」（Artifacts）現象 1。這些偽影通常表現為注意力圖中出現極高範數（High-norm）的標記，且這些標記往往出現在影像中語義信息量極低的背景區域，如天空、平滑的牆面或地面 3。

在早期的 DINO 框架中，這種現象尚不明顯，研究者能夠觀察到清晰且具有生物學解釋性的注意力圖，其焦點緊密圍繞在前景物體上，這使得 DINO 在無監督物體發現任務（如 LOST）中表現優異 6。然而，隨後的改進版本如 DINOv2，雖然在 ImageNet 分類準確度等宏觀指標上取得了顯著提升，卻伴隨著嚴重的高範數標記偽影，這些異常激活點干擾了特徵圖的空間解釋力，導致其在需要密集特徵（Dense Features）的任務（如深度估計與密集分割）中表現不穩定 4。

針對這一問題，Meta AI 團隊在 DINOv2 之後提出了暫存器標記（Registers）機制。其核心邏輯是在輸入序列中添加 ![][image1] 個額外的、不參與損失函數計算的可學習標記，旨在為變換器提供一個「垃圾場」（Junkyard），吸納那些與空間位置無關但對內部全局運算至關重要的信息 5。雖然暫存器在視覺上成功地消除了注意力圖中的亮點，並在某些指標上恢復了空間特徵的清晰度，但 LaSt-ViT (arXiv:2602.22394) 的最新研究指出，這種修補式的方法並未觸及偽影產生的底層根源，即視覺變換器在聚合全局信息時的「懶惰行為」1。

在 ADR 016 系統架構的語境下，特徵的穩定性與抗噪能力是決定 A2 實驗成敗的關鍵。ADR 016 要求模型在極端場景（如低光照、高動態範圍）下依然能保持對目標對象的精準捕捉。若模型本質上依賴背景捷徑來推斷全局語義，則在 Phase 3 的複雜場景篩選中，將面臨巨大的特徵漂移風險。因此，深入分析 LaSt-ViT 與 Registers 之間的效能差異，不僅是學術上的探討，更是決定 A2 技術路線選擇的緊迫需求 3。

## **2\. 懶惰聚合假設與 LaSt-ViT 的理論框架：深入診斷特徵偏差**

LaSt-ViT 提出了一個極具洞察力的假設：視覺變換器中的偽影起源於一種被稱為「懶惰聚合」（Lazy Aggregation）的捷徑行為 3。在缺乏細粒度空間引導的粗粒度語義監督下（例如僅提供影像級別標籤的分類任務，或基於對比學習的文本-影像對齊），模型往往會尋找最簡單的信號路徑來最小化損失函數 3。由於全局注意力機制的存在，ViT 發現利用語義無關的背景補丁（Patches）作為傳遞全局語義的媒介，比精確定位複雜的前景對象更為「省力」3。

為了量化這一現象，LaSt-ViT 引入了兩個核心指標：補丁得分（Patch Score）與盒內點位評分（Point-in-Box, PiB）。補丁得分定義為 標記與各個補丁標記之間的相似度。實驗發現，前景補丁通常集中在補丁得分較低的區域，而背景補丁則主導了得分的分佈尾部，這意味著 標記實際上被大量的背景信息所污染 3。PiB 評分則反映了模型注意力落在真實物體邊界框（Bounding Box）內的程度。

### **表 1：視覺變換器與 ResNet 在 PiB 指標上的對比數據分析**

| 模型架構與監督範式 | 是否存在高範數偽影 | PiB 評分 (反映前景聚焦度) | 特徵對齊質量評估 |
| :---- | :---- | :---- | :---- |
| ResNet (Supervised) 2 | 否 | 68.4 | 高 (受限於感受野局域性) |
| ViT (Supervised) 2 | 是 | 42.7 | 低 (背景主導) |
| ViT \+ Register (n=4) 3 | 否 | 41.5 | 極低 (Registers 未改善對齊) |
| DINO-ResNet 3 | 否 | 71.1 | 優 |
| DINO-ViT (v1) 3 | 否 | 45.3 | 中 |
| OpenCLIP-ViT 3 | 是 | 39.8 | 低 |
| OpenCLIP \+ Register 3 | 否 | 37.6 | 惡化 (背景偏差加劇) |
| **LaSt-ViT (LazyStrike)** 3 | **否** | **大幅提升** | **顯著對齊前景** |

數據分析表明，僅僅添加暫存器雖然消除了高範數標記，但 PiB 評分反而可能下降。例如，在 OpenCLIP-ViT 中添加暫存器後，PiB 評分從 39.8 下降到 37.6，這證明了暫存器僅僅是將背景補丁中的「能量」轉移到了暫存器標記中，而沒有從根本上糾正模型將 標記與背景關聯的錯誤傾向 3。

LaSt-ViT 的研究進一步通過「掩碼探針」（Masking Probe）實驗證實了這一依賴性。如果在預訓練好的 ViT 中移除補丁得分最高的前 50% 標記，ImageNet 的分類準確度不僅沒有下降，反而略微上升了 1.2% 3。這一反直覺的結果揭示了 ViT 在決策過程中對背景信息的過度利用，這種「懶惰」使得模型在面對複雜的下游密集任務（如物體發現與分割）時，由於特徵圖與語義邊界的不匹配而性能崩潰 1。

## **3\. 頻域特徵穩定性分析：1D-FFT 通道維度聚合機制**

為了對抗懶惰聚合，LaSt-ViT 提出了一種基於頻域分析的選擇性聚合方案。與大多數在空間維度（Spatial Dimension）進行操作的方法不同，LaSt-ViT 創新地將快速傅立葉變換（FFT）應用於通道維度（Channel Dimension）1。這一設計的物理直覺在於：在深度學習網絡的最後幾層，前景標記代表了具有一致語義結構的實體，因此其特徵向量在通道之間表現出更強的平滑性與穩定性；而背景標記則包含隨機的環境雜波，其通道頻譜分佈更為散亂且高頻波動劇烈 1。

### **通道維度頻域操作的數學實現**

給定變換器最後一層輸出的補丁表示 ![][image2]，其中 ![][image3] 為補丁數量，![][image4] 為通道維度。LaSt-ViT 的實施步驟如下 1：

1. **實數一維變換**：對每個補丁的 ![][image4] 維特徵應用 ![][image5]：![][image6]。  
2. **低通濾波**：定義一個高斯權重向量 ![][image7] 作為低通濾波器，對頻域信號進行點對點乘法：![][image8]。  
3. **逆變換與實部提取**：通過逆變換回到特徵空間：![][image9]。  
4. **穩定性評分計算**：計算原始特徵與低通濾波後特徵的差異率：![][image10]。

這一評分機制本質上是在尋找那些在低通濾波下「最穩定」的特徵組分。實驗證實，前景物體的特徵在這種變換下保留度最高，而背景偽影則因其高頻特徵被濾除而產生巨大的變動量 1。最終，LaSt-ViT 通過通道向的 ![][image11] 池化（Channel-wise Top-K Pooling），選擇性地將這些穩定的前景特徵聚合到最終的 標記中，從而實現了 與前景對象的精準錨定 1。

這種機制的獨特性在於其「無監督」的屬性。它不需要物體邊界框的標註，僅憑藉特徵空間內部的頻譜規律，就能自發地在預訓練階段壓制背景雜波的傳遞。對於 ADR 016 所要求的 Phase 3 篩選機制，這種頻域過濾器可以作為一個極強的正則化項，防止模型在推理時被複雜的動態背景（如移動的樹影、閃爍的霓虹燈）所誤導 9。

## **4\. LaSt-ViT 與 DINOv2 Registers 的對比效能調查**

在 12 項基準測試的對比中，LaSt-ViT 展示了相對於 DINOv2 Registers 的顯著優勢。特別是在密集對齊任務（Dense Alignment Tasks）中，暫存器機制幾乎無法解決 CLIP 等文本監督模型的對齊失敗問題，而 LaSt-ViT 則能將性能提升數倍 3。

### **表 2：LaSt-ViT 與 DINOv2 在密集預測任務中的性能對比 (ViT-B/16 骨幹)**

| 數據集與指標 | CLIP (Baseline) | CLIP \+ Register | CLIP \+ LaSt-ViT | 增益百分比 |
| :---- | :---- | :---- | :---- | :---- |
| COCO-Object (mIoU) | 8.8 | 8.9 | **13.3** | \+51% |
| ADE20K (mIoU) | 3.1 | 3.2 | **8.3** | \+167% |
| Cityscapes (mIoU) | 6.5 | 6.7 | **12.1** | \+86% |
| VOC20 (mIoU) | 49.0 | 50.1 | **75.0** | \+53% |
| Context59 (mIoU) | 11.2 | 11.3 | **15.2** | \+36% |
| COCO-Stuff (mIoU) | 7.2 | 7.3 | **11.8** | \+64% |

值得注意的是，在更強大的 ViT-L/14 骨幹上，LaSt-ViT 的提升效果更為驚人。在 VOC20 數據集上，LaSt-ViT 將基準 CLIP 的 17.1% mIoU 提升到了 72.4%，增加了 55.3 個百分點 3。這種數量級的提升在傳統的架構優化中極為罕見。分析認為，大規模模型（如 ViT-L 或 ViT-G）在訓練過程中更容易陷入懶惰聚合的局部最優解，而 LaSt-ViT 通過強力的頻域引導，成功地將模型從背景捷徑中「拉回」到了對前景語義的關注上 3。

相比之下，DINOv2 Registers 在 ImageNet 分類準確度上僅有約 0.05% 的邊際提升，且在深度估計任務中改善輕微（RMSE 降低約 0.1），而在物體發現任務（LOST）中，雖然相較於無暫存器的 DINOv2 有所改善，但仍未達到 DINO v1 的解釋力水平 4。這進一步印證了 LaSt-ViT 論文的標題——「視覺變換器需要的不僅僅是暫存器」。

## **5\. 行人重識別（ReID）與多目標追踪（MOT）中的應用潛力**

對於 A2 實驗的核心任務——行人重識別，LaSt-ViT 的前景錨定機制具有極高的應用價值。行人 ReID 任務的長期挑戰在於「視角偏差」與「環境干擾」。由於行人影像通常是從不同角度、不同解析度的攝像頭捕捉到的，背景中往往包含極其相似的雜亂元素 15。

### **處理解析度與局部細節**

行人 ReID 需要提取高度區分性的局部細微特徵（如鈕扣、鞋子、領口紋理），而傳統 ViT 擅長捕捉全局上下文，這常導致其忽略這些關鍵的局部細節 17。LaSt-ViT 的通道向頻域篩選本質上是在評估每個空間補丁的語義純度。在處理低解析度行人影像時，該機制可以優先聚合那些受噪聲干擾較小、特徵結構穩定的補丁，從而增強模型對影像質量下降的魯棒性 18。

### **跨模態與多場景適應性**

在 VIS-IR（可見光-紅外）跨模態 ReID 任務中，不同模態間的幅度（Amplitude）組分差異是導致模態鴻溝的主要因素 14。頻域分析顯示，影像的相位（Phase）組分保留了行人形狀信息，而幅度組分則攜帶了顏色與光照信息 14。LaSt-ViT 雖然主要在通道維度操作，但其底層的 FFT 濾波思想與跨模態頻域對齊（如 FDMNet 採用的實例自適應幅度過濾）具有異曲同工之妙。在 ADR 016 的多場景檢索框架下，結合 LaSt-ViT 的穩定性評分，可以有效地過濾掉受照明變化影響劇烈的非穩定通道，從而提升跨時段（白天與黑夜）檢索的成功率 14。

### **表 3：不同 ReID 增強方案在 A2 實驗中的預期表現評估**

| 技術方案 | 針對痛點 | 預期 Rank-1 準確率 (Market-1501) | 對 Phase 3 篩選的價值 |
| :---- | :---- | :---- | :---- |
| 基準 ViT-B/16 19 | 全局信息過載 | 91.5% | 中 (易受背景干擾) |
| PASS (Part-Aware) 19 | 局部細節缺失 | 92.2% | 高 (細粒度對齊) |
| **LaSt-ViT (Foreground-Anchor)** | **背景懶惰聚合** | **\~94% (預估)** | **極高 (淨化特徵空間)** |
| Registers (DINOv2) 7 | 注意力偽影 | 91.8% | 中 (僅視覺清晰) |
| PersonViT (MIM-based) 18 | 標註樣本不足 | 93.5% | 高 (自監督特徵優化) |

在 Phase 3 的篩選機制中，建議將 LaSt-ViT 的穩定性得分作為「特徵質量門控」（Feature Quality Gating）。當檢測到的候選行人區域其通道穩定性得分 ![][image12] 低於預設閾值時，系統應降低該目標在重識別池中的權重，這能顯著降低由於背景誤檢導致的偽陽性（False Positives）案例 9。

## **6\. 推理期 FFT 的優化路徑、系統兼容性與運算成本**

技術決策的一個重要維度是運算成本。雖然 LaSt-ViT 引入了 FFT 運算，但其對整體推理延遲的貢獻與傳統的自注意力運算相比是可以忽略不計的。

### **運算複雜度分析**

假設輸入特徵為 ![][image13]：

* **自注意力（Self-Attention）**：複雜度為 ![][image14]，隨著序列長度 ![][image3]（補丁數）的增加呈二次方增長 26。  
* **LaSt-ViT FFT 聚合**：複雜度為 ![][image15]。由於 ![][image4]（通常為 768 或 1024）是固定的，且運算發生在序列長度 ![][image3] 的線性空間內，其計算開銷極低 28。

### **靜態優化潛力與硬體加速**

1. **靜態濾波器融合**：由於高斯卷積核 ![][image7] 是預定義的且不隨輸入改變，頻域中的乘法運算可以轉化為特徵空間中的一種特定權重投影。雖然動態的 ![][image11] 選擇保留了非線性，但在靜態場景下，可以通過裁剪（Pruning）低分佈通道來實現進一步提速 28。  
2. **硬體協同設計**：SNPP 等前沿研究顯示，在 FPGA 上實現 1-bit 量化的頻域處理，其延遲可比 GPU 基準降低 6.4 倍 28。對於 A2 實驗的邊緣側部署，這意味著 LaSt-ViT 可以無縫集成到現有的神經處理單元（NPU）中。  
3. **ONNX 與 TensorRT 兼容性**：LaSt-ViT 使用的是標準的 ![][image5] 算子，目前主流推論框架均已提供深度優化後的實作方案。與需要特殊暫存器標記邏輯的 Registers 方法相比，LaSt-ViT 的純代數實現更易於導出到生產環境 1。

### **微調成本（Fine-tuning Costs）**

LaSt-ViT 展示了極強的「零樣本」轉移能力。在許多情況下，直接將預訓練好的 ViT 骨架與 LaSt-ViT 聚合層結合，不經過任何下游微調即可獲得顯著提升 3。若需進一步優化，僅需針對最後的聚合層進行「適應性微調」（APLA），這種方法僅需更新不到 5% 的參數，GPU 訓練時間可縮短 43%，且不增加額外的推理成本 32。這使得 ADR 016 在 Phase 3 的迭代中，能夠快速適應新的數據分佈，而無需昂貴的全參數重訓。

## **7\. 輕量化替代方案與 A2 實驗技術路線的 ROI 評估**

在決定 A2 技術路線時，必須評估 LaSt-ViT 相對於其他輕量化替代方案（如 Early Exiting, MoE, 或輕量級卷積混合架構）的投資回報率（ROI）。

### **與 Early Exiting (LGViT) 的對比**

LGViT 通過引入異構退出頭（Local Perception Head & Global Aggregation Head）來加速推理，雖然能實現 1.8 倍的提速，但在複雜的密集預測任務中存在較大的準確度損失 27。相比之下，LaSt-ViT 並不追求極端的延遲縮減，而是追求「特徵純度」。對於 ADR 016 這種安全等級較高的系統，特徵的精準度（防範懶惰行為導致的誤判）比單純的推理速度更為重要 3。

### **與混合專家模型 (MoE) 的對比**

AdaMV-MoE 等多任務混合專家模型雖然能提升特徵的表達能力，但其複雜的路由機制增加了訓練的難度與參數存儲開銷 33。LaSt-ViT 通過一個簡單的頻譜過濾算子，在不增加參數量的基礎上，達到了與 MoE 相當甚至更優的密集對齊性能，其 ROI 明顯更高 3。

### **表 4：A2 實驗候選技術路線 ROI 綜合評估**

| 技術路線 | 準確度增益 (Dense Tasks) | 推理成本增加 | 微調靈活性 | 總體 ROI 評級 |
| :---- | :---- | :---- | :---- | :---- |
| DINOv2 \+ Registers | 低 (+0.5%) | 極低 (\<0.1%) | 低 (需全參數重訓) | 3/10 |
| LGViT (Early Exit) | 負增益 (加速為主) | **負 (提速 1.8x)** | 中 | 5/10 |
| **LaSt-ViT (arXiv:2602.22394)** | **極高 (+40% \~ 150%)** | **低 (\<2%)** | **極高 (插拔式應用)** | **9.5/10** |
| MoE (AdaMV) | 高 (+10%) | 中 (+15%) | 低 | 6/10 |

從 ROI 矩陣中可以清楚地看出，LaSt-ViT 提供了一種「非對稱」的優勢：以微小的運算代價，換取了特徵解釋力與穩定性的跨越式提升。這對於資源受限但要求高性能的 A2 實驗而言，是技術路徑的最優解 3。

## **8\. Phase 3 篩選機制的優化方向與結論**

綜合 LaSt-ViT 的理論深度、實測性能以及系統維度相容性，本報告為 A2 實驗與 Phase 3 篩選機制提出以下優化方向：

### **A2 實驗技術路線決策**

建議全面轉向基於 LaSt-ViT 的頻域增強型變換器架構。具體實施上，應採用「凍結骨幹 \+ 頻域聚合微調」的策略。利用 CLIP 或 DINOv2 作為強大的基礎編碼器，通過 LaSt-ViT 的通道向 1D-FFT 模塊取代原有的 輸出。這將確保模型輸出的每一個特徵標量都經過了語義穩定性的「審計」，從源頭切斷懶惰聚合行為對系統穩定性的干擾 3。

### **Phase 3 篩選機制優化建議**

1. **特徵穩定性門控 (Stability Gating)**：在 Phase 3 的多目標提取環節，系統應計算每個候選區域標記的頻譜差異率。對於背景特徵過重的候選框，系統應將其穩定性得分 ![][image12] 直接映射為特徵的可信度權重。這能有效減少行人 ReID 中由於攝像頭邊緣噪聲導致的匹配錯誤 1。  
2. **頻域感知的負樣本挖掘**：利用 LaSt-ViT 識別出的「高不穩定」背景標記作為訓練中的硬負樣本（Hard Negatives）。通過強制模型在頻域區分穩定的前景語義與不穩定的背景噪聲，進一步純化模型的特徵空間 15。  
3. **自適應通道過濾**：針對不同的下游任務（如分割 vs. 識別），動態調整 LaSt-ViT 的高斯核寬度（![][image16]）。在需要精確輪廓的分割任務中，使用較寬的頻譜保留；在追求語義純淨的 ReID 任務中，則採取更嚴格的低通濾波，以過濾掉變動劇烈的環境標籤 1。

總結而言，LaSt-ViT (arXiv:2602.22394) 為視覺變換器的設計與應用提供了一個里程碑式的轉向——從單純追求模型規模與分類精準度，轉向追求特徵的「結構穩定性」與「語義對齊質量」。在 ADR 016 的技術演進路徑中，LaSt-ViT 展現出的理論完備性與實戰效能，使其成為主導 A2 實驗成功的技術基石。這不僅能大幅提升 Phase 3 篩選機制的準確性，更為未來大規模視覺系統的穩定部署奠定了可靠的技術支柱 1。

## **9\. Phase 1 實作狀態與 Phase 2 驗證計畫 (2026-05-02 增補)**

本節記錄目前 (Sonnet 工作分支) 已落地的 Phase 1 CUDA + C++ 整合，以及與論文原版的差異對應。詳細演算法總結請見同目錄 `last_vit_method_summary.md`。

### **9.1 Phase 0 結論 (Python prototype)**

`scripts/eval/validate_last_vit_phase0.py` 強制 `HAS_CPP_EXT=False` 走 Python TRT path，從 `output_buffers["last_hidden_state"]` 直接讀取 [B, 196, 768] 張量並執行 dual-sigma pipeline。在 MOT17-04-SDP 100 幀 GT crops 上：

* **Cosine gap (same vs different identity)**：dual-sigma (`σ_embed=0.015, σ_gate=0.040`) 達 +0.0823，相對 baseline image_embeds 的 +0.0674 提升 **+0.0149**。
* **FG/BG stability discriminability**：Mann-Whitney U-test **p=0.022**（前景 patch 的 stability 顯著高於背景 patch）。
* 結論：post-hoc 後處理路線可行，量級遠低於論文宣稱的 mIoU 增益（合理，因論文是訓練改造）。

### **9.2 Phase 1 落地內容 (CUDA + C++ + PyBind11)**

已完成檔案修改（commit pending）：

| 檔案 | 變更摘要 |
|---|---|
| `CMakeLists.txt` | `saccade_perception` target 加入 `CUDA::cufft` |
| `src/perception/preprocessor_gpu.cu` | 新增 5 kernel：`apply_gauss_kernel`、`stability_score_kernel`、`topk_threshold_kernel`、`topk_aggregate_kernel`、`mean_stability_kernel` + host 入口 `launch_last_vit_refinement` |
| `include/perception/feature_extractor.hpp` | 新增 `extract_with_stability(...)` 方法宣告 + `lhs_name_/lhs_N_/lhs_C_` 成員 |
| `src/perception/feature_extractor.cpp` | 建構子自動偵測 `last_hidden_state` 維度；實作 `extract_with_stability` (chunked over `max_batch`) |
| `src/perception/perception_python.cpp` | PyBind11 binding `extract_with_stability(input_ptr, num_images, embedding_ptr, stability_ptr, stream_ptr, σ_embed=0.015, σ_gate=0.040, top_k_ratio=0.5)` |

Build 通過：`cmake --build build` 全部 7 targets 編譯成功，符號 `saccade::FeatureExtractor::extract_with_stability` 與 `saccade::launch_last_vit_refinement` 已暴露於 `libsaccade_perception.so` 與 Python 模組。

### **9.3 與論文流程的差異追蹤表**

| 步驟 | 論文 (`last_vit_method_summary.md` §3) | 現行 Phase 1 實作 | 風險等級 |
|---|---|---|---|
| 特徵來源 | 訓練期取代 ViT [CLS]，骨幹被反向傳播校正 | 推論期後處理 SigLIP2 預訓練 `last_hidden_state` | **🔴 不可消除** — 量級上限受限 |
| Stability score | 公式 `s = 1 - ‖x − x̃‖² / ‖x‖²`（per-patch 解讀） | per-patch（通道求和） | 🟡 與論文步驟 6 「各通道找最高分」存在歧義 |
| Token 聚合 | Channel-wise Top-1 voting + mean pool | Patch-level Top-K threshold + mean pool | 🟡 V1 變體；Phase 2B 將補測 V2 嚴格版 |
| σ 數量 | 單 σ | Dual σ (`σ_embed`/`σ_gate`)，Phase 0 已驗證優於單 σ | 🟢 我方優化 |
| FFT shift | `fft + fftshift`，Gaussian 置中 | `cufftExecR2C`（DC 在 index 0），Gaussian 自然以 DC 為中心 | ✅ 數學等價 |
| L2 normalize | 未明訂 | 末段 `launch_l2_normalize` | ✅ ReID 必要 |

### **9.4 Phase 2 驗證計畫**

| 階段 | 內容 | 工時 | 通過門檻 |
|---|---|---|---|
| **2A** C++ kernel 數值正確性 | `tests/test_last_vit_cpp_vs_python.py`：同 batch 比對 Python prototype vs C++ kernel（embedding cosine sim、stability abs diff） | 0.5 天 | embed cos sim > 0.9999；stab abs diff < 1e-3；batch ∈ {1, 8, 16, 30} 全通過 |
| **2B** 演算法變體比較 | 在 `validate_last_vit_phase0.py` 增補 V1/V2/V3/V4 切換選項 | 1 天 | 選最佳變體；下限不得劣於 V1 baseline（cos gap +0.0149） |
| **2C** Tracker 整合 + MOT17 A/B | `TrackAppearanceBank` 接入 `extract_with_stability`：R0 baseline → R1 純 embedding → R2 加 stability gate → R3 gate sweep ∈ {0.3, 0.4, 0.5, 0.6} | 2-3 天 | **Go/No-Go**：R1 IDF1 ≥ 46.29% + 1.0%（即 47.3%）；否則停止整合並產出限制報告 |
| **2D** 延遲剖析 | `nsys profile` 分析 cufftPlan 開銷、Top-K kernel occupancy；若超標則把 plan 移到 ctor 快取 | 0.5 天 | 整體 pipeline 增加延遲 ≤ 5%（論文宣稱 < 2%） |
| **2E** 超參 sweep | `σ_embed × σ_gate × top_k_ratio × gate_thr` 共 192 組合，先在 MOT17-04 fast eval 後選 top-5 全 7 序列 | 1 天 | 僅當 2C 顯著增益時執行 |

#### **演算法變體規格 (Phase 2B)**

* **V1**（現行 Phase 1）：stability score [B, N]（通道求和），patch-level Top-K threshold + mean。
* **V2**（論文嚴格版）：stability score [B, N, C]（不求和），對每個 channel c 取 argmax patch → vote，累計票數選 top-vote patches → mean pool。
* **V3**：stability score [B, N]，但用 score 加權 mean（軟版 Top-K）。
* **V4**：stability score [B, N, C]，每個 channel 獨立取 Top-K mean（無投票）。

#### **MOT17 A/B 矩陣 (Phase 2C)**

| Run | Embedding | Bank gate | 對比 |
|---|---|---|---|
| **R0** | image_embeds（baseline） | 無 | IDF1 = 46.29% (現行) |
| **R1** | LaSt-ViT (V*ᵢ winner) | 無 | 純 embedding 增益 |
| **R2** | LaSt-ViT | `stab >= 0.5` | + stability gate |
| **R3** | LaSt-ViT | sweep `stab >= τ` | gate τ ∈ {0.3, 0.4, 0.5, 0.6} |

### **9.5 風險與回撤策略**

| 風險 | 機率 | 影響 | 緩解 |
|---|---|---|---|
| 後處理增益 < +0.5% IDF1 | 中 | 高 (Phase 1 工作白費) | 2C R1 早期判斷；保留 V2 voting 變體作備援；產生「為何後處理上限有限」分析 |
| cuFFT plan 反覆建立拖慢 pipeline | 高 | 中 | 2D 證實後立刻把 plan 移到 ctor (1 天工作量) |
| C++/Python 數值不一致 | 低 | 中 | 2A 阻擋；Phase 0 prototype 為 ground truth |
| LaSt-ViT 對 ReID 特徵分布不適 (mIoU 任務 vs ReID) | 中 | 低 | §5 Table 3 已預估 ~+2.5% Rank-1，遠低於論文 mIoU 增益是預期的 |

### **9.6 Go/No-Go 決策流程**

```
2A correctness ──┐
                 ├─→ 不通過 stop，修 kernel
2B algo variants ┘
        ↓ (選最佳變體)
2C R1 (純 embedding 替換)
        ↓
   IDF1 ≥ R0 + 1.0%?
   ├─ Yes → 2C R2/R3 → 2D 延遲 → 2E sweep
   └─ No  → 結束 LaSt-ViT 路線；產出限制分析
```

最快 go/no-go 時間：完成 2A + 2B + 2C R1 約需 **3-4 個工作天**。

## **10\. 最終結論與封檔 (2026-05-02)**

### **10.1 Phase 2 執行摘要**

| Phase | 執行內容 | 關鍵數字 | 結論 |
|---|---|---|---|
| **2A** C++ 正確性 | `tests/test_last_vit_cpp_vs_python.py`，18 tests | 18/18 passed；發現並修正 cuFFT C2R unnormalized bug（÷C） | ✅ |
| **2B** 演算法變體 | V1/V2/V3/V4 on MOT17-04-SDP 15 frames | V1 gap=+0.0887 (Δ+0.0235)；V2(論文) gap=+0.0708 最差；V2/V4 p 值反向 | V1 最佳 |
| **2C** Tracker 整合 | MOT17-04-SDP 全 1050 幀 R0→R1→R2 | R0=44.85% → R1=44.94% (+0.09pp)；gate=0.30 全清零亦 +0.00pp | ❌ No-Go |
| **BG mask** 背景預處理 | 7 種 mask (Gaussian/mean-fill/vstrip) | 所有 mask 使 base_gap 退步；best delta=+0.0204 仍是 none | 無助益 |

### **10.2 根本限制（確認）**

Phase 0 至 Phase 2C 的完整實驗確認了一個根本限制：

> **LaSt-ViT 的增益來自訓練期骨幹校正，非推論公式本身。**

SigLIP2 的 `last_hidden_state` 所有 patch 的穩定分數皆均勻分布在 ~0.12（無論前景或背景），FFT 低通濾波無法在這個已高度語義化的特徵空間中分離前景/背景。論文宣稱的 +40%~+167% mIoU 增益，是因為訓練期的 LaSt-ViT 聚合層強迫骨幹只依賴穩定前景特徵計算損失，改變了骨幹本身的學習方向。

### **10.3 留存產出（可供未來參考）**

以下程式碼已整合進 codebase，即使本次 go/no-go 為 No，仍有技術價值：

- **CUDA kernels**：`src/perception/preprocessor_gpu.cu` — 5 個 kernel + `launch_last_vit_refinement`（cuFFT, Gaussian 低通, Top-K 聚合）
- **C++ API**：`FeatureExtractor::extract_with_stability()`，可供未來支援 LaSt-ViT 訓練模型時直接使用
- **Python wrapper**：`TRTFeatureExtractor.extract_with_stability()`，支援 C++ 與 Python TRT 兩個路徑
- **驗證工具**：`scripts/eval/validate_last_vit_phase0.py`（支援 `--variant-compare`、`--bg-mask-sweep`、`--sigma-sweep`）
- **單元測試**：`tests/test_last_vit_cpp_vs_python.py`（18 tests，含 batch/sigma/top_k 覆蓋）
- **C++ 修正**：`GPUByteTracker::set_unified_score_params()` stub 補實作（修復 undefined symbol）

### **10.4 未來路徑（若要真正使用 LaSt-ViT）**

唯一有效路徑：以 LaSt-ViT 聚合層替換 SigLIP2 的 attention pooling，從頭或 fine-tune 訓練。可從官方 CLIP 權重 (`openai_b_16.pt`) 出發，但需評估與現有 SigLIP2 ReID 品質的對比。工作量估計 5-7 天（ONNX export + TRT build + MOT17 eval）。

#### **引用的著作**

1. Vision Transformers Need More Than Registers \- arXiv, 檢索日期：5月 1, 2026， [https://arxiv.org/abs/2602.22394](https://arxiv.org/abs/2602.22394)  
2. (PDF) Vision Transformers Need More Than Registers \- ResearchGate, 檢索日期：5月 1, 2026， [https://www.researchgate.net/publication/401280024\_Vision\_Transformers\_Need\_More\_Than\_Registers](https://www.researchgate.net/publication/401280024_Vision_Transformers_Need_More_Than_Registers)  
3. Vision Transformers Need More Than Registers \- arXiv, 檢索日期：5月 1, 2026， [https://arxiv.org/html/2602.22394v2](https://arxiv.org/html/2602.22394v2)  
4. Glitches in the Attention Matrix | Towards Data Science, 檢索日期：5月 1, 2026， [https://towardsdatascience.com/glitches-in-the-attention-matrix-a-history-of-transformer-artifacts-and-the-latest-research-on-how-to-fix-them/](https://towardsdatascience.com/glitches-in-the-attention-matrix-a-history-of-transformer-artifacts-and-the-latest-research-on-how-to-fix-them/)  
5. Paper Review — Vision Transformers Need Registers | by Arjun Rao | Toward Humanoids, 檢索日期：5月 1, 2026， [https://medium.com/correll-lab/paper-review-vision-transformers-need-registers-0edb16e05079](https://medium.com/correll-lab/paper-review-vision-transformers-need-registers-0edb16e05079)  
6. Do All Vision Transformers Need Registers? A Cross-Architectural Reassessment \- arXiv, 檢索日期：5月 1, 2026， [https://arxiv.org/html/2603.25803v1](https://arxiv.org/html/2603.25803v1)  
7. Vision Transformers Need Registers | OpenReview, 檢索日期：5月 1, 2026， [https://openreview.net/forum?id=2dnO3LLiJ1](https://openreview.net/forum?id=2dnO3LLiJ1)  
8. Vision Transformers Need More Than Registers \- arXiv, 檢索日期：5月 1, 2026， [https://arxiv.org/pdf/2602.22394](https://arxiv.org/pdf/2602.22394)  
9. Vision Transformers Need More Than Registers \- arXiv, 檢索日期：5月 1, 2026， [https://arxiv.org/html/2602.22394v1](https://arxiv.org/html/2602.22394v1)  
10. Vision Transformers Need More Than Registers \- WisPaper, 檢索日期：5月 1, 2026， [https://www.wispaper.ai/en/blog/vision-transformers-need-more-than-registers-20260228/eng](https://www.wispaper.ai/en/blog/vision-transformers-need-more-than-registers-20260228/eng)  
11. Toward Robust Gait Identification: A Frequency Domain Approach in Varied Surveillance Environments \- IEEE Xplore, 檢索日期：5月 1, 2026， [https://ieeexplore.ieee.org/iel8/6287639/11323511/11348061.pdf](https://ieeexplore.ieee.org/iel8/6287639/11323511/11348061.pdf)  
12. A time-frequency feature fusion-based deep learning network for SSVEP frequency recognition \- PMC, 檢索日期：5月 1, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC12515880/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12515880/)  
13. GitHub \- ChengShiest/LAST-ViT: \[CVPR 2026\] The official PyTorch ..., 檢索日期：5月 1, 2026， [https://github.com/ChengShiest/LAST-ViT](https://github.com/ChengShiest/LAST-ViT)  
14. Frequency Domain Modality-invariant Feature Learning for Visible-infrared Person Re-Identification \- arXiv, 檢索日期：5月 1, 2026， [https://arxiv.org/html/2401.01839v2](https://arxiv.org/html/2401.01839v2)  
15. A Large Scale Benchmark of Person Re-Identification \- MDPI, 檢索日期：5月 1, 2026， [https://www.mdpi.com/2504-446X/8/7/279](https://www.mdpi.com/2504-446X/8/7/279)  
16. Performance comparison of our method with baselines on the Market1501, DukeMTMC-reID and MSMT17 dataset. \- Public Library of Science, 檢索日期：5月 1, 2026， [https://plos.figshare.com/articles/dataset/Performance\_comparison\_of\_our\_method\_with\_baselines\_on\_the\_Market1501\_DukeMTMC-reID\_and\_MSMT17\_dataset\_/23610114](https://plos.figshare.com/articles/dataset/Performance_comparison_of_our_method_with_baselines_on_the_Market1501_DukeMTMC-reID_and_MSMT17_dataset_/23610114)  
17. PersonViT: Large-scale Self-supervised Vision Transformer for Person Re-Identification, 檢索日期：5月 1, 2026， [https://arxiv.org/html/2408.05398v1](https://arxiv.org/html/2408.05398v1)  
18. PersonViT: Large-scale Self-supervised Vision Transformer for Person Re-Identification \- arXiv, 檢索日期：5月 1, 2026， [https://arxiv.org/pdf/2408.05398](https://arxiv.org/pdf/2408.05398)  
19. arXiv:2203.03931v3 \[cs.CV\] 20 Jul 2022, 檢索日期：5月 1, 2026， [https://arxiv.org/pdf/2203.03931](https://arxiv.org/pdf/2203.03931)  
20. Person Re-Identification with Improved Performance by Incorporating Focal Tversky Loss in AGW Baseline \- PMC, 檢索日期：5月 1, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC9784096/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9784096/)  
21. A local-global transformer-based model for person re-identification | PLOS One, 檢索日期：5月 1, 2026， [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335848](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335848)  
22. Frequency Domain Nuances Mining for Visible-Infrared Person Re-identification \- arXiv, 檢索日期：5月 1, 2026， [https://arxiv.org/html/2401.02162v1](https://arxiv.org/html/2401.02162v1)  
23. Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification \- arXiv, 檢索日期：5月 1, 2026， [https://arxiv.org/html/2509.16635v1](https://arxiv.org/html/2509.16635v1)  
24. MSFF-Net: Multi-Sensor Frequency-Domain Feature Fusion Network with Lightweight 1D CNN for Bearing Fault Diagnosis \- MDPI, 檢索日期：5月 1, 2026， [https://www.mdpi.com/1424-8220/25/14/4348](https://www.mdpi.com/1424-8220/25/14/4348)  
25. MM FD ConvFormer multimodal frequency aware deformable CNN transformer network for robust brain tumor classification \- PMC, 檢索日期：5月 1, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC13087178/](https://pmc.ncbi.nlm.nih.gov/articles/PMC13087178/)  
26. POINTS-Long: Adaptive Dual-Mode Visual Reasoning in MLLMs \- arXiv, 檢索日期：5月 1, 2026， [https://arxiv.org/html/2604.11627v1](https://arxiv.org/html/2604.11627v1)  
27. LGViT: Dynamic Early Exiting for Accelerating Vision Transformer \- arXiv, 檢索日期：5月 1, 2026， [https://arxiv.org/pdf/2308.00255](https://arxiv.org/pdf/2308.00255)  
28. FUSAR-Ship: building a high-resolution SAR-AIS matchup dataset of Gaofen-3 for ship detection and recognition | Request PDF \- ResearchGate, 檢索日期：5月 1, 2026， [https://www.researchgate.net/publication/339951986\_FUSAR-Ship\_building\_a\_high-resolution\_SAR-AIS\_matchup\_dataset\_of\_Gaofen-3\_for\_ship\_detection\_and\_recognition](https://www.researchgate.net/publication/339951986_FUSAR-Ship_building_a_high-resolution_SAR-AIS_matchup_dataset_of_Gaofen-3_for_ship_detection_and_recognition)  
29. Resolution-robust Large Mask Inpainting with Fourier Convolutions \- CVF Open Access, 檢索日期：5月 1, 2026， [https://openaccess.thecvf.com/content/WACV2022/papers/Suvorov\_Resolution-Robust\_Large\_Mask\_Inpainting\_With\_Fourier\_Convolutions\_WACV\_2022\_paper.pdf](https://openaccess.thecvf.com/content/WACV2022/papers/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.pdf)  
30. FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection, 檢索日期：5月 1, 2026， [https://arxiv.org/html/2603.20403v1](https://arxiv.org/html/2603.20403v1)  
31. (PDF) No-Reference Video Quality Assessment Using Transformers and Attention Recurrent Networks \- ResearchGate, 檢索日期：5月 1, 2026， [https://www.researchgate.net/publication/374004520\_No-Reference\_Video\_Quality\_Assessment\_Using\_Transformers\_and\_Attention\_Recurrent\_Networks](https://www.researchgate.net/publication/374004520_No-Reference_Video_Quality_Assessment_Using_Transformers_and_Attention_Recurrent_Networks)  
32. APLA: A Simple Adaptation Method for Vision Transformers \- arXiv, 檢索日期：5月 1, 2026， [https://arxiv.org/html/2503.11335v1](https://arxiv.org/html/2503.11335v1)  
33. AdaMV-MoE: Adaptive Multi-Task Vision Mixture-of-Experts \- CVF Open Access, 檢索日期：5月 1, 2026， [https://openaccess.thecvf.com/content/ICCV2023/papers/Chen\_AdaMV-MoE\_Adaptive\_Multi-Task\_Vision\_Mixture-of-Experts\_ICCV\_2023\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_AdaMV-MoE_Adaptive_Multi-Task_Vision_Mixture-of-Experts_ICCV_2023_paper.pdf)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAYCAYAAADOMhxqAAAAu0lEQVR4XmNgGAWDCSwH4rlAPA2IeYB4NRBvA+IrQHwCSR0YMAJxOhCzAPF/IJ4IxMZQcQsgfgDE/DDFIGAKxFxALArEr4E4EE3uARALI4kxsEFpfSDeC8RSSHKxDBBbOZDEwIAZiDuAOANN/BgQX0ITAwOQc0CetEUTB5k+nwHiHz5kiUioJMgvMKAKxGeBWAOIQ4C4GUkO7ByQBmSgAMTbGSAe3gDEWsiSIkAshiwABaA4QQmhUUATAAB+2hmKkKKjRwAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHYAAAAZCAYAAADkBdqeAAAEjElEQVR4Xu2Z34tVVRTHV1lZmqmZ5YTBLX+gpZiKUWgxPVj6IESJKOKDFGpI4A8QVFJQiZRAaErEFylCA60XoRxIEMUHCSEievFloIce+iNsfVh7zd13zTnDvTPduDPsD3w556xz7rnnnrXXj72vSKFQaIvVqgdJU5NtuuqrZPtH9Xiyd8Imad73puqO6lfVkeyaQpc5rPpDtSzYP5ems8fCOdWN7PhF1SXVvMxWqOEh1ZwaPa16pHlpLfvErh9UPZ9sU1SLhq8wnkqKPCP2HJEh1Z5gm6a6opoV7IXATLFU95fqetLfql+kmQp5uTiujvfSlms3p31e/Iy077ygOhNsDJztUu/YNcGGYxlAc4O9UMPxtPVodV4Sc1Kf6u3M7uA8j56TYs4Ad3YVA2Lp9EMxp1aB486qHgv2JapvZeSgmfQQAXk6ZYS3Q51j+byn1K9VT2bngHMeba+LOZa6OlqTs1isftZFKhCpMQ0DA+ZANE5WcMpvYqkujvB22ZG20bGwLm1PSOs5nHIwO4YNqquqvcGe84mYU4nIqkaI38A5r9fOm6o/pX4wDLNUmg8NK1WvZccTAVLWF6pn44kOqXMsL5n0B19Kawpk3yPdoeGi1r4b7E5eUxtiDozwm6ijcZBeE+vAa6FWMKqAL7gv1lbzoKekOgX0IutV30fjGKlyLI0VUfq+6idp7ZBJld5crc3sEAeAs1q1JRqVXapH0/5Gsbkv9/Vmjvnrp+n8qHyjejU75ibnVW+I3dR/ZK/DaPcudLzkjqUj5oX6AsN+1fx0vqfJGwrCfkhGttX/B0y4fxZLMaSwTsEBRBTpqU51KTESI5YGh/o3W/WKXzSRwKGk4Viox0tDWrNCHQwyVmiY0HdKNx1LiTok9lwXxNLyhIIfVDVfGi+05Y1orKAhY0/930l739EO0bHAei+dKA7enWw9Deuap8WilGiltjqsd/qqxsOqFWmL6JifSOf8fCPYaDBIsT+KvaA87TN4Vknr6GdO1ifWoT+X2dvhv2yequaxRC0OZfuyVE9NegZWSWgKWG/kxdAceDuPU+i+fJ7EPwwfqY6Jpcud0voiL6o+Fmu8SIuwUJr/bJAKGfHAvS+LRTKRRt3C6XyWLhL7PelsusU9P0vb8VLlWKDGUlJ4J1ULFD0DD0jEDoo5ilHIdIdjXlK+SP2BWPT6SgctPMtZ/C11VJp1eZvqdtoH6h7XOf1i3+EwjyPKcfrvmZ3swb06hcHIlIDlv3ZXmhwc59OWKIdUnNsXZOd6Cn58PiqZ3C/PjnNIxR51DTEHkmrZesNDNBN5DtkgX/aihuO0CNcwP3RwMt/XKQxWnokBG50zkF1XSOA4nOZRwCL3D2KrVaRwh2WuftXWdHxXbA31HdVbYtMZT3VA6mRCjrO9aSH6SfussxJFhS5CGiaivK7cEnMgKc9TL+eojdRpnwuThknbpDDOk7bzv6to3IjMPEIZLOx7PSt0EaIJBxKxVfM4IovOmAjM//il+41Rh7NI//m0KtZD7sH9Cl0m1s7CJIAlNV+9iZFVKBQKhTHzLxBTu5FPp3iOAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAABA0lEQVR4XmNgGDGADYiFkTATqjSYL4gmhhUUAfF/JByCKs2gCsQnoXIwPA1FBRrYCMTzgfgBEFugSoHBDCCORRdEB8wMEEU8DBAbj6FKg8FaIBZCF0QHAkBsCmXDnI8OOhggFuIFAQwIRSkMEINcEdIMokCsj8THCkAGgGyDASUGiNcmATE7VAzkWpCr8QKQTavRxECBDXJVCQMkefSjSmMHIG8huwgEYIG+E4ilgHgTqjR2AIoNWEAjA5AhIMPqgLgJTQ4rAHkLFJjoAOQtkEGPGSCuxgtAgZmALggFoEhoZoBYhDOgQfnHCYinAPEOIPaCiqEDUKCDvMWILjEKRgG1AACwnCgFS8N1bwAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAZCAYAAADXPsWXAAAA5ElEQVR4Xu2TMQ4BQRiFf4kQCQqi0agkSo1OoXMBLZXCHcQZhMIB9A7gEFQOsL1D8L/9dzaTl5klStkv+ZLNvjezM7uzIn9NVe2o3YBfs1FfH9ypAzegiLvYgBAr9amOOWAwQcI3M+rqMRPXQVpik+w58JiKdWZ0P2coVlhy4OEmiXa26lXtceCBTnQl2MpZbCs1yhyuc1NHlKXMxZ4w4cBjIdZZc+DAGUChaCv4Kuj0OXBgmSjEtgISsXMSpeh8tNWTelCblOVUxCbBl2GQXcRW0KAsBT8XBsd8iL0H/JwlJT/zBv/lNLRSv8hfAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFkAAAAZCAYAAABaU4LDAAACyUlEQVR4Xu2Xz6tNURTHv8iP5FdeSiZORIQwMJAfPQMZmUkZkIFipmQiSRkpAzPUC1EmfowUI+VHxMTAQCZyZ2/A/8D6tt5+1l1773P2HdxzJ/tT3947a62z9z7rrL3OvkClUqmMldWiqYQW2qBKngWivd7o+C7626IHogPz0f3TIF5TTmRJwp7Tn4QtpbeiizDw4qPop2hWdNo6MyyHDjbjHcIu0SfReehLmwSLRDdFb0QbnI9wbV+d7Rj0mfjXs0f0Q3QGwzuWc/AePwcLcWicHaIjovWi1yhLcgMd/JKzB05AE73JO3pireil6Da0Uj27oc9quSr6AH02z0rRO2irtLyC5sHP8Rw6RwT7ammSLyC/IMKxOPkN7+iJU9D59xkbd9W6uf/3i+4a3xbRF8RFwwIkrORf1jEH5xh4o3Af+mIiSpO8THRHdA/aNlKEJDNmEqS28Rr8r66topPGl2sVLCZyWPTEOqAtifewJXkuQ/0RpUlmFfwWTTu7hTGTrOTwcQ4nn+Oiz8g8uPAYGr8RGs/KfYS411r4wnjPWWdvpTTJfLscvHF2S4jpGmtccO6UcrBV+Fgqt1MJn22A4ZbUSWmSuT3aFsyPHT96D0UrnM/CrXmlUEPHoQ7Yd7k+trTAQehHKgV7NeOZ6ABzkYsnnIP+3Ic1S2mS2SraksyTBf1HvcMxriSHbRz6KeFc7NMp+IFiPFtGgG3CXntYvQOM2CpIaZK5IJ4scryAxiz1jp5IbeNGtNlcW7ZB12tPFqvQ/qOMc/CekVoFKUlyg3hBAR7SeVh/L9rufH0x6jZmq7iG9uNoitz5uJOSJB+CDj7t7OQ6tJU0zt4nYRvbVtEGj3XPoK0hea7NMEB7y4wIpe/FhDPxhAn0/qBZ0VPEv4b65BzidVGpHUd2ir4hjmfCmXjPYtEtxPFBzXxkpVKpVCqVSqVS6eQffV/TUisR6UAAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAARgAAAAZCAYAAADqpTGdAAAH50lEQVR4Xu2c2YskRRCHwwPv9dZVQRlvXRWVRVnRlRW8UFcUD3a9QPBYnxTxQGRdvFdfRFFZVFgVD1x9UAc8EF8UBR8UFBFBhH7zwT9C45vMmM6OzuyqobunZ6bzg2CqsqqrqjMjfxmRWT0ilUqlUqlUKpVKpVKprCgOUTssY3umJ7Vgjdq3vnAKoK6OULtY7UB3rNLMY2qv+8JKlx1q/7UwzttP7aLMsZJxLuDE/ljJflY7LXysh93UrvCFypfSf43UvlG7e/7sMp+q3esLp4BSvZ8voc0XKtQrlVsk+L/nALU3pNZTI9ZREQMPlfi92kxSdruE8/mb4y+1B+M2jnuf2u7dw/PC5u/Xka4wwQ8Szvsn/i1hz++dgHtulPA8N7pjBp2pdGwaoH28wMxKqM+c2E8DJ6rtUvtX7U8J/uV91dhb7TO11f5ApcsggQHSh9OT/SaBeVrtqbjN52a6h+YoCcy7apuT/UvUjpfu85UoCQwQ/eAov/kDyh4SnvUYf2CKyAkMbcYAQd1NI6SLpI0HSfDxQQID+Nc0D1KN5AQGZba8HLFIO35JYM6If3HQ9+P2BrW94rZREpgtao+7MhhGYMCe97JC+TSTE5hKlzYCs13yA9g8dIg0ND9XQug8LeQEhsk/c7r1ahckx0oCsy3+NeWH6+PflJLAcO79rgyGFRienVGGaIWoxXhJ7Zdk30NujW/wfWwf39h3/ozlCb7NCE0KmRMYm/y1AYZIhjo4Vu2cWGZzcofHfThawnVL7bAcaSMwDL5Z/yQE/yRuU4nk6qQDq9SekTCiLiVoZL5oW2vb0NZBT5bwuXVqH0h5VDOBYWKU83E8tk1gmigJTIlhBYYUiHb9ScJ3NNh/M9lPwTdm4vZWCb5xuQTfQKyWmm+0gclsvofNh50kYb4lFRhElLbsJGXUK21MHTM3wbwW18Af31F7QO2teC5wvYEjemRc/jxK2giMiXTf81E5pshABeL8NuL5EXrSvKj21QLsnvCxRqyDemsSGG9LVWC4D+f4kboj+WVGRm98w0BM7Pr4BttLzTeaIOXluf1q2Vrpr5dcVAPWzuncDHXTUbs0KbP2bWJc/jxK2giMDWCH+gOpQ+JUHbXzkjLPrPR3KjMexEKlnLEi4isstUH3HTe5FInUwDuYYQKTdjKcbluyP4gd0n+/QQwrMNZhaL/UCfhM7pm5TnqtLyT4xiRgkpvn5NmHASGgE/gJ7ZyY5MqA56A8BR+gPdP6svadNHdJeFeljW2Kn/G0ERgbwAadM9fBcw3gIe+ksVK4+HVxm4r212GZ9lcJOSodkYnMTjxGqIkDp6s0i01OYNj2DmbkBAZynTXHYguMRaTbpXcOpiQwno6M50U8Vs3SCLoEaRzPPgx8z1wnyIlJrgyoL9LKlCowLQWGCzHp51c8PIgLDpvCkp5NEiMshN37dA/POdGPyT6OhagYT0qIoCZFTmDSVSRPSWBsFamJxRYY2ozjfhWprcBwHr4xavz7RSU4z9f1QhmVwHCNlKUsMKNgKIH5Xe0F6eZQqXCcKf2d3s4z8XhYwjlMCNpKQypAREUo401q18Yy8J2T6KfNOwfjmhTLCcwgSgLTlsUUGOqV6IXJTQ+diGfx0B74BiJLm/P51Dc+kq5v2EqM/fWrj5RTlq48kaaR+myV/jpggGOwMn8CBiOiX1ZoViflHp5xf18YWaf2h9rZrjwnJrkyGLXAjMufR0kbgbH66nm+gyVUwscSRjacyCqUmXQa33d6GpDzjGelN+JBeIheLD1iWTQ3t5Jr6EmyUgWGzn2zBHG52h2DWenvMIBPcD18BN9IO5utsphvbJQQkT4h4V53qh0Vjx2ntlPtKgnfeXMsf1Tt1WiE58atah+qXaP2XizjO32u9oqEa/s5EMPmarhfDpvkfSgp4zvcJv1iMkhg/P2HEZjlQBuB8bowB5XLKPW1dJ0CR2T/OcmnB1QcEQzQYF481kvvjZ6X/pSLRtsuvXMBk8IcwVtJOEq/RWpqAMAB/efMcs4M/jwzu5cJS8nelvxvmAwTSg+/PcE3SGvxDQYNfIP5B3zD4JXyGyREGPbTiFUSzieSwJdssEFcZuI2vsfLi5xrbJBuCkeqidgAPrU2blsEnYOfdHTU/nblKUdKiNhtwYHvcqF064v29W1M21g9pebPw06V/jbJRYhLHfzLfw+zXN+gP3vhnQOnTzsGDXBWsu+hca3CEA6fQm2RXoHZlGwbOHzuISuLD2kw72vkxB7foNMCkRC+YfseroEQwIwE8SCisIEEY9tGeSIjoiQDX2KOJ7fAgHCZn3GPQe+XmHA1sUbthLhN6kYfSFOyysJggBmJkPpc3IMAMXqVYMQinPbCVJkczI8M8zsSxMNSBPttE+3LCG++QjpMWoygMegwx2OrMby8x5wM6VoKqZiPWIiIWBnhxTb+VYWHxYTXfGFlrJDJ7JJ8eywIHIjGzo0yBgJEFFPCctvciFmZDHTkYRwEMdkpIbqhg38Xy4kQLKph0KHdSbdIq4lKGGhIyx+R4A+8umA/+yeiIJXxEYvN3d0h/fODgLhc6QsrY4UBKn3JsFLpg5ShNLfRBOnujIQByHd6yky4EA9SLYO0KN0HW1UxOCedPOUapFcllvtvpJYbTNAPylgqlXlOibYQ/NxKZbp4WfKLQZXKSGAEw5gXqSJTqVQqlUqlUlnG/A8VMjGsKxgFAQAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAYCAYAAADDLGwtAAAApklEQVR4XmNgGAW0AIZAvBaIu4H4ABC7AfFyZAUgwAjEt4G4E8rnAeL/UAwH4kC8EYhDkAUZIIoWIQukQAUVkMRANoDEqpHEGA5DBZGBBhCfhdJwABJAVxjAALGWF1lwEhC/RuK7MEA85o4kBgZmUAlBIBYC4hsMEBtUkRVhA7UMmE7BClYzYFEI8j6y7ySA+DkQh8NVQAETEJcyQKIKpMkJVXroAABxEB/Ft3y1MwAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIIAAAAZCAYAAAD9ovZ9AAAEK0lEQVR4Xu2Y26tNURSHh0vI/RKJ6Ehyzy0ilxApTxSFECUpeSCUJOVOpBSJJBIPKKXclRci8SQvXk558OCPYHxn7nn22GOvvdZ2OPs4zK9+7TXnXGedOcccY8yxlkgikUgkEolEIpHo9HRVzVTNUfVwY4n/hJ2qj6oXqieqb6orqi72psS/y0bVK99pOKK66Tv/BBNVC0x7hoRU9K/BGieY9iLVaNP+W3ipmuQ7Dd1VB1Q9/UABIyWsuSqjLFHdK10z+EXCJPqpjql2lMYaBXMYoBryC+rd8pf5XFRtK13jCK9Vl1TzVG9VI0pjfwOrpP4Nvqbq6zsz2KX6KuXnPlD9kJAAWrihmh4bEgYvSzDQd9UmM9YI+qjOSDgP69X2lr/M57iEKAI2nbUtlOAMrHloaayjoRg85zslODsB4rmlmuY7HQQX67XP3SJh3QNjh40mjNGsmm36PFclFCw8hF/rRNBLwvnFOHqq+qS6rZpl7mskVN0ognOfl/wKPM7f67GELIRh/RiisCP7eGeNIstio1qQre64PrJzfD4Batkj+cHaTXVUdVfMpksIdp6XCQ7AsVCUJpnMXsk4ZwzvVKdMe7eEY6ejYfNxgqIjjwDBWNYWpFVr0MNSGY04G0fOMyk7GXa4ULomhRMkeXb7047A/NnTQ1L5f+n7bNoV1BMpgBFJq3kw6fWmTZFGX9Gz2xuy3iPJz3oQDeij96QEJ2FzqNoHVw7LQ9Vm02bNdqO2mOssyDZkTw9BxLN8PYBjUVPUgnU2q1a4fjIadmiFtH1aygu3HjdFqs/OeF9egcYYBrFGIj3xz4toj2KRe1gnRsGJMWjc4Jg6Pdxn57uu9BvfpDCsTa0UY8ydSj4eQzgLxrY2bDLXtVgj9ReLfFPwzmEhYxH5NrNhrwoHJcXRQbpbLmHh8dWKwsqnE4gelkeThGMBI0eYzH7TbiREDOvk93rpOjJZwvo9FJE4PLApPoMclHJqZeOzonKcVNuhHgZJeJsrgoBZ7TsdBON9CXMB5krGYu6txxqbTKRQ0G1VDZeQgmifUPWPNxo4OqKBIrYQA86tpVKO2Kxqt5GwLs5uvs6NUa007flS7exAUMQ6IuvT7gcp10BNUlmIAc8kkIoq+lpskGDnrLkBWbzeD0o847mEAFgroV6rclDSBJsVGaaaatoeJke0WOaa61hk+SOloyHK7FE1XjXKtD3x9RJ8NgCbWpukesNiDdFWO/C8ZaqzqrGmnzXw1bFWoBYR98fWb23CGghIYRQskcWS81rSSeA8xdl9oRhhk30N5PE1xO+AU/CVkYLbZ6YiyPg2c1B/ZBWdvwTeREaIhQdn53sJ9UWE18pm0+6M4Oh5r5fUUVWp1cGxUPP1rEHgyGy6fTt4I8E5Ev8ZZGwKW75N7JPy19VEIpFIJBKJXH4CnQXGztVnYlMAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,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>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMkAAAAfCAYAAABNoek3AAACxElEQVR4Xu3czYtNYRwH8B8Jobw1eUm6ifJSZEEUGkVsiKJQFoqysfGyUCblZW2nyUYppWyEvBQbSlmwsrOZnYU/gu/X73ny3J9z7zD3ztw5zvdT307zu2dmNufXec5znueaiYiIiMgAzEBWI6PIdWRpqp9FFueTRJrsLvINOYScQ74g+8ybZl5xnkjjLEDuIasq6veR4VAXaZyjyPdYTM4jrVgUaZqXyI9YTC7GgkgT5SZphTpxyCXSeHuRr+aNksOf55QnTYLZ5jNofxvOsM389ZsiU4xTv5zFKpuEuYrMKs7rt93Iq3/IQ/MpapGBm4/cMG+Uk0WdzbQnHbvhhXwiFkXqiHeJzbGYtMyb5HRRW4d8NB8mdcP3Le9jUaSOONx5EYvJSvMLvRXq/aThlkx7nN59G4vJQeSytQ+tdpoPxbpZiCyLxQ704C7T3qj5kCo+YyxHnqRjNoxcQR4VtYjPKzeRd8j28JlILT1F1piv0eKU723z4ReXqKwoztuAHDZfosIm6GRjOvJvbSs/EKmrtenIF4a7zKd8T9mfd5bsM7I/Fiu8RoZiUeR/t8T8GYbPEXnZPI9xposP/FzvJdI4O5AtyCb7fafhosg4/OIwS0MtaaxF1j67xLvIpeLnuebvSOLdRaSxOLS6YP7ugkMsTgJ8aDtDpOHyEhW+P+HU8HGb/EWRIiIiIiLmExxcStPLprX8N/QlHNI3D5BjsTggVRc3n+GY8XBldt7Hs978OwhEesZ1ZgeQW9Z5xcBUqmoSbjO4E2oRX9o+M59uJzWJ1BYbkdueO90ZqppkItQk0hccmmxNR47juTKglyX1/F0u+e+GDfAceRw/SGKTcPvAeNsMiNPp5f9Wk0jP+F4mf6neJ/OvZB1DRvIJE8AXo50u/hL34hyJxaRskmvmF/4ba99+UOLWgjPmS3rGirqaZEB+ApdIYB0CZA8NAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEMAAAAZCAYAAABq35PiAAAClUlEQVR4Xu2Xz6tNURTHl9/yI08Sz+hKiucVEiPJ6yUDKQMDClGKyUukV5KJkYmUkZIBU5SBnmeglJEBA0nK5M4M/BGsT2vvd/ZZZ99zDe47dXrnU986e+197tl77bXW3leko6Ojo8Q61XbVUdUa17fAadUf1d8hYsxHe6V1rFA9kGItY+Xugk+qS0k7vvhBtSOxH1L9SNpt5IWYMwbyXLUpaW9VzakeqVYn9i2qV0m7jXyRmg3tSXn34aKY9w47O+1nztYmlomti6jPwgJJi5SYW95JB1S3nS1CZB1TjTs7Ezii2hDay8XGbVsY0RzUCdZ1JrHtVR1M2hUIo9q8CqxUXVHdV60Ptt2q12KpNimWglRvfu+tmGPglOqmmLOaAie8FHMK36cMnFP9EjtpsjDxvjdmuCB2wsRdB6KM90m1W6qrqpPBNpGM26N6L9XoWyxw+j2xqGcT2UDmzeHB3NYWQwvYUTrxWh38EOMeO3vMyydiqUNa5Cr4cbEI3O/sERx45z91I7xTB86neLL4u65vINQQJn7Z2T0swucfbAz2tEgxga9JG66rPoulVY5ROyNG50+xaJ4W26haCO++VE8Sz3nJ7yw7ENMkQptcjZCfRA4Rg/OagBRhHlwVqFs8z5ZGOAbdL3LgjKdSLjykBQXzmhSFkX4fQWeDje81BVGYRifffxeemVulkBMNfRmeIsBJgePSBeUKak+qzngjVsGbxEdnrGv8R3kYjatCR07kei8OzDCj+q2aV32XfB5yovAxihbjvql2lkYsPpwUbNKJxEaK9MVONO4aI4E04MhMoyFCPaAuxDDkOp8b1wTjUk4FnjfL8HIwMmIFH/gPcSkRK7i/7i859qmmEu0qd3d0dHS0m38QyIs2OgQKHwAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAYCAYAAADpnJ2CAAABe0lEQVR4Xu2VvytGURjHHxLCQFJIuvkxKGyUQpQyGRVKFmSxoGRS/AE2yWJkIZMfK2U0yGY0GPwRfL+ec/N4XjflnvfN4FOfbud7T/fcc+45zxX5p0SUwXZ4ALdhY8gXYUPaKSb78AVOwSX4CCdEX6DG9MtNHTyEbd/kR3DM5bnhDN58GJiXyLMjl5I94JoPYsBl44Cv8AwmsNx2iM04fBIdNJXtKtupGCRwQXSnctAH2Gs7FItauCM66Ky71w8rXObhWZ7xIUngsQ8DPPQccNJkLAx9pp1FJ+z2IRmBFz4MtMJb0ZeKRnr++OaWZngerik9cBlemcwzCnfhDRx09z7gDJ5F19t+l1N4Z9rcrZuiK3Jvcs+G6HP4XBaMArrClSVsGG7BOSmcccq1aCH/CfZr8uFv4Kw7fOiohHvhmgvuPO7YFvksCKxM/GaWgWAU6uXrt+Ys1k27WvQXl3t2WfDorIoe9BXR5babLTo8BtxgQ/AETksJ6u/f5B0+WjYF4Qdj6QAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFQAAAAZCAYAAACvrQlzAAADFElEQVR4Xu2Yy6tNcRTHl0febxGi7oSQZyK3JAaiTCgZeCQxuBMppCTJM48kj2QgXcLEIxnILSaKkkwkE5M7M/BHsD6t/btnnd/Z+559nHtvm36f+nb3Xvt3fvfstdfrbJFEooAxqh+q36q1zr4+s6HVzl4W9v0m9nn271H9Ur1WjXTr/ksWqt6pzqlGOPsN1Sp33irzVZ+ktudw1WbVCam4U6eqpheIa804LOZUoumgsx91xzBMNTuywWTV2NioHFF9j43KF7FrleWx6oPqreqNWIqhz2JOeqHa1be6kdOqKWJp+dzZiSYPkbZPzLGe66pZkQ3uqp7GRjGHno2NVWOc1Grd1ew8cEU1Sszxi509sC37u0T1VTVazMET+1bUc0i1W9Uh5swiqJlhbw8PuSs2DiZx2pYBB67LjmOH3lFNU22R/Mgg3QFH3hRz7PLa5Qaofzi1KDIDpHu8Dw+2V/6u0ZVmpthN75H+v2AzihzKOU6boTrj7IDdpzD18KOq29liFqhuqQ6IRWoeHapLUt/kOD4u+VlSxyKp3QysVK1x581glOAftUuRQ4kkIp2xiE7uieskkJIPY6MDZ/LgiVScGtdU2CCN6b5VrH4WslH1LDtmUxoB4we157yUqxN0VVJtIMhz6Byx9KSZ7JDazTMn/hRz3jWpfwDHJN/RfIa9YnaqlmXHpHRohjRKmiR1maAZn60p5IFqhTtnEzpbp1hB3uuu5UHhfxIb2yA49L1Yxw6D9Ssxh1Ye/1SpT73SWrHFAdw8g25/KkscoaT6BNU9yZ8fKw2OJN1JsbIMtkNPis2f6EJY9K9AehMR1JCy0Mzux8Y2iB1KVHZnxzSSykONuiwWlURnp7vGLEcZaMZANSWcFuZJ35ToxNuz41ACKgkNhYLPT6tNYk0o3BA3cUryx4kY5j7mzzJr+6PIofBINVdaLyFDCg4gQntU+8XmsvCa6qJqUm1pU5gRGezpxPOia2VgEOfhxgqv43DubWd/KeVemgw5fFH/05BfPEvdeSvwaouUDO8mvRKJRCKRSCQSCfgDZbyL5Z1Ef/8AAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFYAAAAZCAYAAACrWNlOAAADnElEQVR4Xu2Yz6tNURTHlx8hyc+UX4ObUn5GiciProGMFGVCSFKUDEhK0itKSihKioiBAZFSZKBEmZlIJibKwMAfwfq8ffa7+6y7zrnn3Xfui/fup769+9baZ9991957rb2PSJ9xzRbVOdUO1VTj69Mlk1XHVItVn1VP8+7/jwmqA6prqjuqo3m3y3kJbS30NUs1L9P0vHsQbLSzHFJdyD7PV71VzW25a8GOL2qOamLSzrJA9Vz8cbusl/DAbdVu1UnVe9UV1cxWsxx0zo9eYh1KQ/VR9ScTfdvgELzoR78z+wrVnuwzP/ZN9rdO1qq+Sf77U/1SPRA/yKskpKhSCApb7bD4nTyS8EV2hsh716U9WJarqlsS+uCvZavqtTUmXFR9ssYauSdhbN6OWiPBd9w6lA/S2lUuLyWsFBu4yD4JnS819tWqL8bm8URC8OjDa39CQvCLeKfaYI01EndVEfi8iS2ccALJTHxXbTS+lOUSCoidHVbZY2PzuKmaIiEVMMhFebe8kLAtLStV97PPk1QzEl+dMCaCWwRpyNux/A4mvW1cMWBsz7LjTGw3YOw/VJeMzUJAKETAXwbIDkhhRc82NmAVUygAv7dVR8o0CWO6ax0JMbD2+3mWor3M2AeLjreCLEcktDtj7NiaxmZhJVLVgRk/K2FCNmU2VnIsUilxElL1gs3S+XeQJr0VC4wzLpwh4gNEvgy2Mu12JTYCwvMMrAyCxqqNkJfp63L2P0H30sBoQX4nDTSMPaVsYomJTZGlD6TEdmmA2BakB9JEEWxfThSWOKHAD0v7HQkURvrtdEqJxBxpd2IKBZs+H1pHBkV5wBqrBJYA0sZW8yqBZSU+s0ZpFbGGhN1QF8MN7DYJk9w09hTqAW12WkdG14FtSmhz2tirBHa/+IHjqkqfrBROBHWxXcK7haqwW+IEezBBLALO6kXF3Q1sXDlFD3FVxe/dqgAfHRfB+dg7f7L1442r06mil5SdX7kosVK5BJThFq+9Eh72zrAEGx/BL4JBtXWa8EpaJwJLLGLeiWC0SHN9Cu8PWIX89kbe1QYLxF1cvEX6KuFezAql4U8Jd2TveJFSdEGIKSaqKD9xjfbOr73Gji8VceAlFMHtRCx+9nw7BEcntiyBJe9V6RQ4MtmiNp6Ixa92qr6EGauQf21Rr411EnL1eIM0SQ1ZaB11UvSie6wy7Bfd3UIBPCX13aD+dW6oDlpjnz59+mT8BeZEyPZfRpbmAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAH8AAAAZCAYAAAAYEPFUAAAFD0lEQVR4Xu2az6tVVRSAlyUlkWWGWRJlQqEYJIVSoXEFowiCAidGxUMEBZ0oIYhIUCBBVFAQYaDkwEFlBEHlQJAaNdCBiBMnFxo06I+o9b191nvrrLf3vuc97+ENPB8s3r1r77PPOXv92Gvv+0QGBgYGBgZmWaHyjsqnKl+r7G83Z6HfiaiUNNaDKg83cl+7eRZ09OuL+2X+/l5W+k4V7PrdKltDW9/wjPG5EZ6pBHP5rsqjsWESL6hcVPlK5U2VwypXVE6rPDDfrQUP+J0kI0c2qvyp8l8jjL3Wd1BONm0m/7abb4sNKpelPX6Un1TesAsyfCPzfT8MbX1zSBY+r5cbKk/O9Z4HmxCQNSeZ43GV71XeV7krtAHG5WYxQh+SdN29QR/5ROVLSWPwN7JT5deonCIfSbr3utgg6d1/luTgtWywHMaHLSrXVGaCHrAVwXI9NihHVa5GZQ5enkGicY29kl5+U9C/qvJ30OW4IMnAjJF7UDwcB+mLHyTd+57Y0PCiyrj5W2K5jP+WpHtvjw0NFlQxADdLMn7JprMNB1VuqewIbR4biBTtQdfFaF9ImnjSPg9KOvaQep8LumnCPcdR6aDuoM+3scGxXMY/L3XHtaAaBT0QVARoFjMq3hM9x2P94stz05mgi9yt8l7zmb9cQybxkBnWBN004Z61ZcWM/1tscJSMv15SzUCEPhbaDNLzKyrPN98p2iiQ0U3iL0n3LmHGtzn2vCQLA3aOS5KPxMiMpH7HnG6VpLWINakGEW1rLZnmA2mnWDyaieuLp2Wyk1qfxUQ+1TT1zlNOx2d0vtJmmWNZBd6fcU5JqjE+t04V6I+dSljBPAp6wMlwaP4ugHWeCzFkDdI2/V5zOiIVr2TiamBYot94VtJYHzffcYw+Uz7PPJbymgn0MaOUiMb/rNFF0GFYIJtGx2POcnVPjtWSrmf+S9iyQHaOcH3RRlyUe4GI9fNGrHpVAw7CTiFiTgesS37cGuckXUc66wIvz+RYzZGDrSe1CNFVy4DR+HzPLRMsj/ZujM1nn5It23ahi+PWbMhyxvPkHKN6oWHrYfTWLsYnoqm0I1b4bZS6V0cWa3yrVWaC3mOV/oGgj3Q1PjqbU5/mjZtST+MG13IdfXNbVLDMUhrvto0/ktSHfaOni/H3Sd64TDRjUkNQ6XflCUnX1PbjHkvntcixrRJnFjWWYnz4UVLgcODC87Oz2u3aS3TJWjguWfTt2NBQNb5FYKnSpyKlnYOQSHXgBgqd3MST5q1Q4QCmL2r7eyJrj6RTslG7KUs0PgYdu+/GWNpZ0l+zGCbt7znBo712ilcNUDwGz8nt8XEI2nCQEtycrUaJX6Scsqzw67PS9+tvhOzzj6Qj7S5E41tg+EMUS/PHne6KTN4R5SDlM1acP+7xjCTb1ApUoNDjeDr3m8ospFC8n4nghYhITu3OSuV0qIHJzR3y8NBeSgcNbIv62N/H+3th0tjzsz/vgj/bR/xkEiBsd0nlzCGfYxZlWYvPgLAjyBnFlo2SsEzVAs5DMb0rKiOkRdILxmdNzf1Ik+OclIuNOwUOcKgXkPi7CBF6Rtq7CAw+kuSEpaCYBtwHR42ZY2p0/WHnToUsWjrDoOovnr5NAQr0P6Jy2rwu+ZphQGSbpMiPBdcjkopC2vsCw/Mrbe+U/pljIC0F/BMIS+TvKkea731Bnbakf+ZYKmw37EhzYHl5WdL5wMDAwMDAAPwPvjRAkKIACRMAAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAYCAYAAADOMhxqAAAAiElEQVR4XmNgGAWDETgD8Tog/o8Fr0BSBwbCUAls+DUQtyGUMjBUQiVYkMREgfgcEGsiicHBbSDeiCbGywAxxAZNnEGIAWKlJZq4KRDvBWIpNHEGVSA+DMQKaOIZQNwMxMxo4gw8DJgmFTJAbGVEEkMBIMlEKJsdiB8g8XECNiBWZ4DYOAroCwAuGRy0CM57rAAAAABJRU5ErkJggg==>