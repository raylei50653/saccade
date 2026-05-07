# **技術報告：LaSt-ViT (LazyStrike ViT) 架構與演算法分析**

## **1\. 執行摘要 (Executive Summary)**

視覺變換器（Vision Transformers, ViT）在大規模預訓練中常出現特徵偽影（Artifacts）與高範數標記（High-norm tokens）現象 1。本報告基於最新研究指出，這些現象的根本原因並非模型容量不足，而是一種被稱為「惰性聚合」（Lazy aggregation）的最佳化捷徑行為 1。為此，研究提出了 LaSt-ViT 架構，透過「頻率感知選擇性聚合」機制，主動將分類標記錨定於前景特徵，成功在 12 項密集預測基準測試中取得顯著提升，並消除特徵錯位問題 1。

## **2\. 問題背景與病理分析**

* **惰性聚合假說 (Lazy Aggregation)**：在粗粒度語義監督（如單一圖像標籤或簡短文本）與全局注意力機制的驅動下，ViT 傾向利用語義無關的廣大背景圖像塊作為代表全局語義的「捷徑」 1。  
* **高範數標記的本質**：先前的研究（如引入暫存器 Register tokens）將高範數特徵視為需要被隔離的雜訊。然而，深度分析顯示，高範數僅是惰性聚合在深層網絡中的「病徵」（Manifestation），單純轉移這些標記並無法解決下游任務中前景特徵對齊失敗的問題 2。  
* **空間失真量化**：透過框內點比例（Point-in-Box, PiB）指標探測，標準 ViT 的高分分類標記通常落在真實物體邊界外，顯示其嚴重的空間定位失能 4。

## **3\. 核心技術：頻率感知選擇性聚合機制**

LaSt-ViT 摒棄了標準的 CLS Token 全局無差別注意力機制，轉而利用信號處理的先驗知識（前景同質為低頻、背景異質為高頻）來過濾並提取高純度前景特徵 4。

**關鍵運算流程與設計原因：**

1. **特徵分離與梯度阻斷 (Detachment)**  
   * *流程*：將編碼器輸出的空間圖像塊序列特徵 ![][image1] 從計算圖中分離（x\[:, 1:\].detach()） 5。  
   * *原因*：防止後續的頻域濾波操作對編碼器主幹梯度的反向傳播產生寄生干擾，確保濾波僅作為前景探測機制 5。  
2. **空間至頻率域轉換 (1D FFT)**  
   * *流程*：執行一維快速傅立葉變換 ![][image2]，並透過 torch.fft.fftshift 將低頻平移至中心 5。  
   * *原因*：為了量化特徵通道上的平滑度（前景）與劇烈變化（背景），必須將空間信號轉換至頻譜域 5。  
3. **高斯低通濾波調變 (Gaussian Low-Pass Filtering)**  
   * *流程*：利用一維高斯核 ![][image3] 執行哈達瑪乘積：![][image4] 6。  
   * *原因*：大幅衰減高頻分量，物理意義上等同於抹除異質性的背景雜訊，保留平滑的前景語義骨架 5。  
4. **空間特徵重構 (1D IFFT)**  
   * *流程*：執行逆向平移後，透過一維逆變換並取實部 ![][image5] 6。  
   * *原因*：將乾淨的信號還原至空間域，以便與原始特徵進行數值對比 5。  
5. **穩定性分數計算 (Stability Score)**  
   * *流程*：計算公式為 ![][image6] 1。  
   * *原因*：定量評估特徵的受擾動程度。低頻前景受濾波影響小（分母趨近零），獲得高穩定性分數；高頻背景則因變化大而獲得低分 1。  
6. **標記選擇與動態聚合 (Token Selection & Aggregation)**  
   * *流程*：在各通道找出最高分圖像塊（Top-1），統計投票數後提取原始特徵，並透過平均池化（Mean pooling）生成最終分類標記 ![][image7] 6。  
   * *原因*：強制優化器在計算損失時只能依賴高純度的前景特徵，徹底切斷背景捷徑，迫使模型精細對齊前景空間特徵 4。

## **4\. 實驗硬體與優化器配置**

LaSt-ViT 針對標籤監督、文本監督（CLIP）與自監督（DINO）三種範式進行了驗證 5。其核心預訓練配置如下：

* **硬體環境**：使用 8 張 GPUs (--num-gpus 8\) 進行分布式訓練 5。  
* **優化器**：採用 AdamW 7。  
* **學習率排程**：採用餘弦衰減（Cosine decay），搭配 500 至 2000 步的線性預熱（Linear warmup） 1。  
* **精度與正則化**：啟用自動混合精度（AMP，bfloat16 格式），梯度裁剪範數限制設為 1.0 1。

## **5\. 核心實驗成果 (Key Results)**

1. **偽影消除與精確定位**：成功根除了高範數標記異常，使 Patch-BBox 命中率（PiB 分數）大幅攀升，達到甚至超越具備局部偏置的 ConvNet 表現 5。  
2. **全監督下的湧現分割特性**：在未提供像素級邊界標註的全監督（標籤監督）環境下，LaSt-ViT 促使模型自主展現出類似 DINO 的「湧現語義分割」（Emergent semantic segmentation）能力 4。  
3. **零樣本與開放詞彙檢測提升**：在 CLIP 框架的弱監督下，高度純化的前景特徵使得模型在 12 項基準測試中全面獲勝，大幅增強了對未知物體（Novel objects）的區域提議與檢測精度 6。

#### **引用的著作**

1. (PDF) Vision Transformers Need More Than Registers \- ResearchGate, 檢索日期：5月 2, 2026， [https://www.researchgate.net/publication/401280024\_Vision\_Transformers\_Need\_More\_Than\_Registers](https://www.researchgate.net/publication/401280024_Vision_Transformers_Need_More_Than_Registers)  
2. Vision Transformers Need More Than Registers \- arXiv, 檢索日期：5月 2, 2026， [https://arxiv.org/pdf/2602.22394](https://arxiv.org/pdf/2602.22394)  
3. Vision Transformers Need More Than Registers \- arXiv, 檢索日期：5月 2, 2026， [https://arxiv.org/html/2602.22394v2](https://arxiv.org/html/2602.22394v2)  
4. Vision Transformers Need More Than Registers \- 每日论文, 檢索日期：5月 2, 2026， [https://paper.dou.ac/p/2602.22394v1](https://paper.dou.ac/p/2602.22394v1)  
5. GitHub \- ChengShiest/LAST-ViT: \[CVPR 2026\] The official PyTorch ..., 檢索日期：5月 2, 2026， [https://github.com/ChengShiest/LAST-ViT](https://github.com/ChengShiest/LAST-ViT)  
6. Vision Transformers Need More Than Registers \- arXiv, 檢索日期：5月 2, 2026， [https://arxiv.org/html/2602.22394v1](https://arxiv.org/html/2602.22394v1)  
7. Downstream Task Guided Masking Learning in Masked Autoencoders Using Multi-Level Optimization \- PMC, 檢索日期：5月 2, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC12356090/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12356090/)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAYCAYAAACFms+HAAABvklEQVR4Xu2VPS9EQRSGj+9ExGfER4FKhEIioVKQSCQqEh0KhYRK6FRahY7CbyBEQqHR6igU6u39CM6zc07u7CTLRnZZyX2SN3fmzN6Zd8+cmSuSk5OTk5PzR/SrFhNBZxKbsXjdMK/6SAQTSezC4nXHoGQmO1STqid71j0YxfiZ6k6ysvltelStafArGlSbEsy/JmPVYDwNlOFNQglXTGwcUT7VpFIzz1L5b4tQKreS1Tfmj6Jxtq9PNaVqlFBKI9E4MQ41f56n06baV61IeJ8EOQOqHdVcFHPjxL79A+lhxBTG3yVbCDOXqgcJt8yqtTnMcGD9MdVpFN9QnZtIhNfvtupeNSph3naLY3xXtaQ6lDIXRHodblmcZxxHwOSY9kXI5qNqWIIhN0tmj60NrBNnj/evVN1R38E4SYF0nh+TGgcO07Rk5XOjOpGvjWMonceJa7xmxsnwtapXVVCtW9wXnLV+bJySa5Jspxy+1lAz49zxQ9anDtesXZCs1DhYLLhsfXaENobdLGdqz9pdEu5veFEtWLuqxj3jZLm5dLhY574bLVJ6g7A7fgYcxjFMmdWUtFT+BXz5uM5cf8YnP5JS3Uf4wx4AAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAL8AAAAYCAYAAACr8yxQAAAFj0lEQVR4Xu2by6t2UxzHf+6XXF6k5NI5ueWWa0SkQ6SYEBFe5R2IMhBGQqcYMFDuEuX1GhB6pQwoGYlMGEgmJu/MwB/B+vRbv3N+5/estZ+9n0f7Oc8561Pfzt5r7cvaa3/XWr+19nNEGo1Go9FoNBqNRqOx27gwa5k5kHRrTGyUeSTp3x76NR9/WiGvJmO9kNclylTi/KQjQtq3Mnm+1/dJj20cXefrpMdj4pJyW9IlMbExyfFJFyR9ImqWR0UNjq5OuiPpn5wHhyednvRATvvKHX9O0s1Jv+Q8rg0v5f19SbdkvZbTyLO0J3Pa83raBoclPZX0h2xe0zhF9N40Ts5dyfuIcn6a0zEE1ylxZNIXotfaCRwjOgKcHDMaZdZFTXJTzEickLQ/pF0kevz7Id24XdR88GHS3S4PbMSJ93si6T23T6OgEVr5ovkNGwFK+ZieBvx7zBAdSV5OOjNmLDm8l4MxsVGmy/yAKT3TzI+ZOAY+T7rY5UHN/GtSvuY85gcabyn/3KSfZTKcWnaoXxp8EXolG25NcFJIo9fZDZTMT0hi8fdZSUe7vJL5MThGB3pbmzy+IZMhRc3854keH5nX/DeImoFe3hv99aTf3L4RvXFpTies8+nbtdGcmvRN/jsBlU5leYG9VFOpF9qJlMzPPKA2+SyZn/kBJuxDzfw15jU/5WVe8GXSHpeOQUpljt7g/mBzC1PtfovmWNHw0UbfImfI5oMQ2zJLZhjcrrPlh5K+G6B39bSpmLmippk/qmSkEmObnwkwx2Beb4hDsnWOEWEViOtyDP7AG2gZIFR9MCZGeBge8G3RnsBCoN2EmetOUaPQKRAmTDP/R7K5ukJj267mX036USbNzzlcu4Z1huaPv2Qx/mCVLYaO06COa+9vA+LTvaIPWFoRMIgNiYMNYloai0Gejx+PE107tlWPD5LOzttU+FrevjHpp7y9KMxc3oyk1SqvFPaQ9pnb72Js81vM/6psjdOnmR/why33Xhny5oVy9fmwdk3SnzFxCoPNj+j1StBrxIq63m2TZx+DDNZcrae51qVzrH/xDFGLpGR+Jnmrbt9TMj+LBde5/S7GNj/1Sz5LsJ6h5icMqvljFp6RfnVgc5Yh9DI/wxoP5Ye457YcoXjzU8mIFRB6eIjmt4e6XHQC4oetaH7fMLrgnhZm9FHfDx0l83dRMv8QxjS/mZeQJcL76noG8wSGZ5t7ID6MGXiAVUHrLAiLuKdByHKvaAfLtkHHSLkINXlP/hy+i/BlmmPAzI/X6GD6vNfosQns4fgLFJyHo1C+MODNj6FjRXvzU+iuG08t2MjsVPNjtvtFjX9XyAPC1q55Spzgch3ug5HNH9TF36JzHu6FcV8RnSBzzNOi/lpNejOn02AeFr3WO3mbNBrVPtGQeCXn8Uxm/hdFv1YTJptnS9DZVld7qHRubLLhwV6Kl2Hm54GuksmKJs/MTwvterEc25U/FqXnRTVTM5rEY2M91ajdy1QaouPSosnqzkxf08eiS7A1rEyReJ1aupWD+vJ19lbSD3kbUxuUd93t+2sA12A51rBzY9gTrxO5QnSOEDvwmfE9/4nSbX7oMjfHduU3xuEy0UWOeT9WRfPzfm2CyuizX/R3UNyry/yYutTxDDU/jbr6hXcWvPlLRPN30cy/fSCGvi8mDiSaH7MfFJ3LHXLpZlqb45n5nxU1OF+gbcSAFdGFhKHmp+HdExPngWXMUqs07NeLfaDgazGxsRD+j1914gvmDwZzA8wXzU+67/gwKWEZ8wJ+D2VzUJsYvyBaLn464pfRu8zPfJNldeYWjcZUMJ3vcYdiPT+GK/2ehrmShclH+Qwpm5TVnFka47zP0dil8LGpzwenEjHsWRQHpP0nV2NEaDB8F0KzNp7R+A8bebVmuhY6+AAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAaCAYAAABhJqYYAAAA30lEQVR4Xu2SvQ4BURCFx09DpVFohMJbSESnFZVEoqGi1IhCyTsoRKUQNeVGQkchOlErPARn7s7dO7sRnUZ8yZfsPfcUs7NL9Od3iMMWHIo5mIBTOFM9KsA1fEAP7uX5BG/wCTPSNQc2bQOwkawK6yr/WG6qzHCUC57TcpYsqzJDRy48OIYTOW9VJ2AFR3BJfoGtwZQuWXiMO7m1sV1YhjHVM8zJvWTUBcy7KlH7TUm7s8WSBD0bRLCbMtjyILgOcyBVZvifuEjIn5q3cZUzz1x0VZ8kbJDbRh9WQo2v8QJ/dEA5cHsuigAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIQAAAAYCAYAAAA74FWfAAADhElEQVR4Xu2Yy6tNcRTHl0deyTPPqEuRRyFlRMrAo6SIlKLcMpIBBkoGSimvGUledWeGygBFiSiJgQyUoYGBP4L1ae+fu3zP3vuc67r3nnvv71Pf9tnrt/fZZ+39/a3128csk8lkMplMJpPJZDKZzDhmoqvP9dP1zPXC9c7VGw/KjA/mu2649rlmhvj6Mr7VNSHE/wsLXDtEMEtim8v4aGOqteZHbJLEtqcTugQMcN41WQcC61wvNVgDxjloxXceda11LXWdcp10TU8HbnP9EsEaid0p46MNZpnmR2yGxD6mE7qEA/Z3VaiD42ZrUFjiemBFnm9dP8rPbL+Wn0/8ObpkcTmA+CG4j17FdjhZaEWvHIg6gZmW8iMncqQvM1u6jS2uuxp05rqmaNC5YkXFqyNOhMTFMnbVtd9ChYhgAA666Xpi/e1jrHDbivzIjRzbleSR4pjrgsQwML/9i8ShzzVHg4EmQzRWfvoMM4YDP8tY5JM1fxHjuDZxyaoTGQmoDukGkW8Vz133wj69lhmbjqf/kmPaZ7vadbzcx3As+ICWTP4cs8yqZ77Cw8IUkd3W/7unyRiGiA9beW3FeZvKfVoM51AhaTm1REMg2kgV9NsmQzBOUoke1xvXvBAbKaIh6irgU2vNb6UV6w5gfVW15kiz+myIYYh4LzaGz3UMtEI8tOYKcdiKczHxZdercr/X2lRIWsZj618/cBJlVdEHrug4M4zVcE+I1cF7N24fiDqFlvHNipub2gf5KtEQTJJ0jWRoNQSrdMyyy4pezBogoYZgHaAzXOE6jzRYA9fdqUHhveuaBtuhi0hmDzeMsqKlVR+4ouN7rPp7hhNdRFL9UqXQ1Xw0xHJrNZ0a4rr1Vw9FDdEpnb5lcG8xWRMYgtyZ3ElUMVpfS4XQ187Uu9jGOEroA4e4+mWcMpZmcMtFhxGuH3NID1pfq2N+0RA91mwITD4UhgCqRN25XJc2fEgHKtjr+m6t+SadsUFO1ipDHAmf260xup1oCF7nmgwBp21oDMF1q/6p3OC6ZZ39U8k41UErBPpghSFo57T1f0YNQc+7H/bHkiGqUEM0MRhDRKjAfBctvZ0JIj1WPPSqNQSVmzgtZZWMDQh9rcTF3MREu9fSbofXzpiPkl4766pCJL52jgRUOAxBhVgkY+fKeLf+H5MZQnh7o/VgAgzAdoVlI2QymY75DXok4Cjx8CMZAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALsAAAAfCAYAAAC/HbySAAACo0lEQVR4Xu3cu2sUURQG8GMUEx/gC/GByBIUfIBWioJKBAULI1oIRrAQItjYJFpYCYp/gE0INlaCYCMKPgobBcFCKzsbO4v8Efp9e+7g3ePs5rG7mczy/eBjmJNJqpPhzp2zayYiIiIyUFYhe5FZ5AGyLdUnkS3FRSKDYAb5jYwjt5AfyDnz5l+fXSdSWxuRp8iekvozZCzURWrrCjIXi8ltpBGLInX1DvkTi8lULIjUWdHsjVAnLmVEBsZZ5Kd5wxfh+XB+UR+sNd/xWWi4IzTU/E2RJeKWI3dd8mZn7iNrsut67TTyfhF5br41KtIzG5CH5g0/kdX5T3EmHTthQ16LRZEq8a59JBaThnmz38hq+5Gv5suPTrhf/zkWRarEZcTbWEx2mzdsI9R7ScsYWTbcVvwYi8kF5K61LllOmi9xOtmE7IjFNvSAKstm1nypEtfgO5FX6VgYQ+4hL7JaxPX8I+QTcjz8TKRSr5FR8xkYbjU+Nl/WcHRgV3bdQeSS+egAm7mdQ+nIv3Us/4FI1falI18cnTLfarxu/9/pC9+R87FY4gOyPRZF6mKr+Rqf6+xi3JfHuDPDB1vO04jU1gnkKHLY/t35OTwWlzVcvmgJI7W32Vp3Q3hXn87OR8z32OPdXqT2uGS5Y773zaULH3a/tFwhMiCK0QHuv3NL8qr1f3hMRERERLrGh3Vuv3bzARcO43FbVw/1sqJxXodfOZJbZwsbVuO06cXsnC/1RFassmbn/P/lUCvzBnmZnavZpVLcbeJHFtspa/alUrNLcyuTDcfw7e3q7Lzf8zdPrP1XjFBs9gPmb5nnwxFojkLn1OzSxAdAvpllOHbMYzcPhcQPjsyHzfstFjN5s3OkmbP+HJvuNHN/0/zdxC9rbXA1e5/8BcGEXbpXEMdZAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAXCAYAAACS5bYWAAACM0lEQVR4Xu2WwUsVURTGvywSk7SUKNtktNEQCtGdQa2CIDAQI+kPaCXUIgiJoDBwG5HiTheCLSJwoS6CR4toowsJN22Elv0R9X2ducyd8+7MBMF7Qv7gY97cc2fmm3PPPfOAI46opYMapOapDeou1R1POCx0Ug3ql9NX6lg+rb3IyDdqwQfIMszwe+qsi7UcGZ2hnlEnXExcpj5TP6kbLtZyVJc7ftAxDsvuCx9oJcqqTLzyAccQDoHZSeoTddEHHMGs5rcNZWqT6vcBh2pVpSLTbeEUtURtU+dczPOEegNrbZ5h6lZ2FKlNWobKcAC2iXXdJep4YUbEHGx5X/tAxD1YJ/Am3sG6RIw6yo/o/CXs/ilkVM8N970AS56SmOQm7GYyE75SungMeaa+UB+z3wHN2aOuu3H1YfXjgMqszKxW87YbW0SFWaEeu4vc9CPqPGx5VqmRfOoflJHn1AM3HlCbC8hsWVs8A4uphAIqh7/6UvYin6jM6RMrs57TsPEJH8iIy6XKrFBylCTdb9rFmtCfFi2zLghqUA+jOQHV3zrqzcakzCohcfbkQS1xjVqBJS3JHRSNSh/Q3E+1QxV7jLwM/JzAleh3ymwfrC673LhW5ADFMqpEtdpA8wt8R7FtlW0wvdRsdJ4yO5Ud1bNlPKZBXXNjlYTe9xTWivR/NrU0KoUtN6aVuh+dq3T2o3Pt/mBeZtU6QxL03Leo6Qb/ylXYjh71gRp6suNJ2PVt+zr+n/wGLx9irPpqSqoAAAAASUVORK5CYII=>