# Saccade 實驗紀錄 (Experiments)

此目錄集中存放 Saccade 專案針對模型、追蹤演算法與管線效能的各項實驗結果與分析報告。
為了保持結構清晰，實驗紀錄依照功能模組進行了分類：

## 📁 目錄結構

* **[pipeline/](./pipeline/)** - 管線與系統效能優化
  * [解析度與 Zero-Copy 實驗 (`resolution_and_zerocopy.md`)](./pipeline/resolution_and_zerocopy.md)
* **[tracking/](./tracking/)** - 追蹤器與生命週期關聯
  * [FP/FN 恢復與 GMC 實驗 (`fp_fn_recovery_and_gmc.md`)](./tracking/fp_fn_recovery_and_gmc.md)
* **[reid/](./reid/)** - 語義特徵重識別與裁切
  * [Semantic Relink 與 Crop 實驗 (`semantic_relink_and_crop.md`)](./reid/semantic_relink_and_crop.md)
  * [動態 ReID 觸發機制設計 (`dynamic_trigger.md`)](./reid/dynamic_trigger.md)

## 💡 命名與維護規範

- **新增實驗**：請將新的實驗紀錄放置於對應的子目錄中，檔名請使用能清楚描述實驗主題的英文蛇形命名 (snake_case)。
- **時間戳記**：若有需要，可在文件內文頂部標註實驗日期，不需加在檔案名稱前綴，以保持檔名簡潔。
