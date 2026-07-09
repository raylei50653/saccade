# Geometry — 模組 TODO

> 全局進度矩陣與 Baseline 見 [docs/TODO.md](../../TODO.md)。本檔只列幾何 / GMC / 卡爾曼模組待辦。
>
> **WIP=1 sole active：** GMC Warp 精度驗證（**依賴** detection VGT，非第二獨立目標）。規則見 [DOC_MAINTENANCE.md](../../DOC_MAINTENANCE.md) § Workstream WIP。

## 待辦

- [ ] **GMC Warp 精度驗證**（sole active）：量化 GMC 仿射矩陣在 detection 模組 [VGT-Mamba](../detection/TODO.md) 時序特徵對齊中的重構與對齊誤差，確認 warp 後 FPN 特徵的對齊品質。
