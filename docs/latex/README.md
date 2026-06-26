# Saccade 數學模型 — LaTeX 文件

單一原始碼,雙版本輸出。學術主幹(L1–L2)+ 可剝離的 L3 實作補充。

## 建置

```bash
make full     # 含 L3 實作補充       -> build/main-full.pdf
make paper    # 乾淨學術版(剝掉 L3) -> build/main-paper.pdf
make clean
```

需要 XeLaTeX 與 Noto CJK TC 字型。Arch:
`sudo pacman -S texlive-xetex texlive-langchinese noto-fonts-cjk`。
`preamble.tex` 已固定使用 `Noto Serif CJK TC` / `Noto Sans CJK TC`;若改回
ctex/fandol fallback,需同步修改 `preamble.tex` 的 `\setCJK*font`。

## 結構(資訊架構)

```
main.tex              骨架,看一眼懂全書
preamble.tex          套件 + \archmap 迷你地圖 + contract/implsupp 框 + \IMPL 開關
chapters/
  00-overview.tex     Part I 主架構:只放地圖(架構圖/模組表/dataflow/baseline)
  05-gmc.tex          Part II 細節章「樣板」(完整範例)
  A-symbols.tex       附錄:符號↔config 大表(reference,不擋正文)
supp/
  05-gmc-impl.tex     L3 實作補充(kernel/env/line number),paper 版剝掉
```

## 寫作公約(怎麼抓範圍與深度)

- **深度階梯**:L0 一句話 / L1 概念+骨架式 / L2 完整可重現 / L3 綁程式錨點。
  學術本體寫到 **L1–L2 且全 pipeline 等深**;L3 一律進 `supp/`,用
  `\suppinput{...}` 掛在章末,`make paper` 自動剝離。
- **每個細節章固定四件套**:`\archmap{焦點}`(全局「你在這裡」)→ `contract`
  合約框 → **章內演算法傳遞圖**(該模組\*內部\*步驟 + 邊上標傳遞的量)→ 數學。
  照 `05-gmc.tex` 複製。傳遞圖用 `step=<C色>` / `flowarr=<C色>` / `flowlbl`
  三個 style(見 preamble `\gmcflow` 範例),顏色用該模組的 `C*`。
- **reference 大表進附錄**,不放正文(避免「沒重點」)。
- 焦點關鍵字:`detect / gmc / track / output`;track 內子章(Kalman/assoc/
  auction/lifecycle/relink)目前共用 `track`,要更細可在 preamble 擴 `\trackmap`。

## 從 math_model.md 搬內容

`docs/reference/math_model.md` 是內容來源(目前整份 L3)。搬法:每節的概念+
公式 → 對應 `chapters/*.tex`(L1–L2);kernel/env/line number → `supp/*-impl.tex`
(L3)。逐章把 `main.tex` 裡註解掉的 `\include` 解開即可。
