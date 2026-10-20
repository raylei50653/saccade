<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-20 -->
<!-- doc-module: cross -->

# #610 第二片：`model_fetch` 模型取得與離線核對

本頁是 [`scripts/model/model_fetch.py`](../../scripts/model/model_fetch.py) 的 CLI 契約、操作說明與測試矩陣。

- **設計依據**：[ADR 028 #610 修訂決策](../decisions/028-model-bundle-runtime-separation.md#decision-610) R4／R6。
- **實作授權**：[#610 第二輪 owner 裁決](https://github.com/raylei50653/saccade/issues/610)，只授權 Draft PR，merge 另外授權。
- **Source baseline**：`main@d6724f11`，也就是 #611 merge 之後。

這個工具在 repo 側，只用 Python 標準函式庫，限 Linux，不進 runtime package。

**它是什麼**：使用者自己選擇上游 URL，提供 expected sha256，說明這個 hash 從哪裡來，並記錄自己接受了上游條款。工具依此取得**一個**檔案，以原子方式發布；之後可以完全離線核對。

**它不是什麼**：
- 不託管、不鏡像、不中繼任何模型，SACCADE 不提供模型 bytes；
- 不是來源或發行者認證，也不是權利或授權許可；
- 不是 model bundle 驗證：不是 Gate A，也不是未來的 `--check-bundle`；
- 不是 identity level（VL1／VL2），也不是 A_L parity 或 benchmark 資格；
- runtime 照舊永遠不下載（[契約第 5 節](../architecture/model_bundle_contract_549.md#fail-closed)）。

## 1. CLI

```
.venv/bin/python scripts/model/model_fetch.py fetch  --source SOURCE.json --dest DIR \
    [--allow-no-expected-hash] [--ca-bundle PEM] [--max-redirects N=5] \
    [--timeout S=60] [--max-bytes N=64GiB]
.venv/bin/python scripts/model/model_fetch.py verify --source SOURCE.json --dest DIR
```

`DIR` 必須已經存在，而且是目錄，不可以是 symlink。

**輸出契約**：下表每一種 exit 都會在 stdout 印出**一份 JSON**，帶 `exit_code`。
- `fetch` 印 `saccade.model_fetch_result/v1`，`verify` 印 `saccade.model_verify_report/v1`；
- 參數錯誤（缺參數、未知參數、型別錯誤、沒有子命令）、檔案系統錯誤和內部錯誤，也各印一份 `{"status": ..., "error": ..., "exit_code": ...}`；參數錯誤時 stderr 另外印 usage；
- 失敗時 stderr 另外印一行 `model_fetch: <原因>`；
- **唯一例外**是 `--help`／`-h`：stdout 印說明文字，exit 0，不印 JSON。

| Exit | 意義 |
|:--|:--|
| 0 | `fetch`：`fetched`、`already_present` 或 `record_recovered`；`verify`：`verified` |
| 1 | 核對失敗：sha256 或 size 不符、截斷、bytes 比預期多。`verify` 沒有通過時一律是 1 |
| 2 | 用法錯誤（含參數錯誤、讀不到或格式不對的 `--ca-bundle`）、source 不合法、政策拒絕（沒有 hash、未接受條款、非 https、redirect 降級、迴圈、超過次數）、不安全或衝突的目標、lock 被佔用、檔案系統錯誤（例如 `DIR` 不可寫） |
| 3 | 傳輸錯誤：網路、TLS（含憑證不受信任）、HTTP status、`Content-Encoding` |
| 4 | 內部錯誤，也就是工具本身的 bug：`status: internal_error`，stderr 印 traceback |
| 128+N | 被 signal N 中斷（SIGINT 130、SIGTERM 143、SIGHUP 129） |

## 2. Source 檔：`saccade.model_source/v1`

由使用者撰寫。key 必須完全一致，不能多也不能少；重複的 key、NaN 都會被拒絕；大小上限 1 MiB。

```json
{
  "schema": "saccade.model_source/v1",
  "name": "yolo26s.pt",
  "url": "https://<上游位址>",
  "expected": {"sha256": "<64 個小寫 hex 或 null>", "bytes": null},
  "hash_reference": {"kind": "upstream_published", "citation": "<上游公布 hash 的位置>"},
  "terms": {"url": "<上游條款>", "accepted": true, "accepted_utc": "2026-10-18T00:00:00Z"}
}
```

| 欄位 | 規則 |
|:--|:--|
| `name` | 單一路徑段，只允許 `[A-Za-z0-9._-]`，開頭不可以是 `.`，結尾不可以是 `.fetch.json`，不可以含 `.partial-` |
| `url` | 只允許 `https`，不可以帶帳號密碼。可以是簽名 URL：query 和 fragment 不會被寫進紀錄或輸出（見第 3 節） |
| `expected.sha256`／`bytes` | `null` 或 64 個小寫 hex／`null` 或正整數。`bytes` 是選填的 size 檢查 |
| `hash_reference.kind` | `upstream_published`：必須附 `citation`。`user_supplied`：使用者自己提供的 hash。`none`：只在 `sha256` 是 `null` 時使用，反之亦然 |
| `terms.accepted` | 必須是 `true`，否則不送出任何請求 |

## 3. 取得語義（`fetch`）

1. **政策**：source 沒有 expected sha256、又沒加 `--allow-no-expected-hash` 時，在送出任何請求之前就 exit 2。明確放行時，紀錄寫 `integrity: local_only`。
2. **Lock**：對 `DIR` 取 `flock`，被另一個 `model_fetch` 佔用時 exit 2。
3. **既有檔案**：`<name>` 和 `<name>.fetch.json` 必須是一般檔案，symlink、目錄都拒絕；工具絕不覆寫或修改既有檔案。
   - 已發布的一對，且和 source 一致：離線核對後回 `already_present`，**不送出請求**。
   - 已發布的一對，但和 source 不一致：exit 2，不替換。
   - 只有紀錄，沒有檔案：exit 2，請使用者移除過時的紀錄。
   - 只有檔案，沒有紀錄（中斷殘留，見下）：如果 source 有 expected sha，先比對既有檔案，不符就 exit 2，不送請求；符合或沒有 sha 時重新下載，只有既有 bytes 和新驗證的 bytes **完全相同**才補寫紀錄，回 `record_recovered`，否則 exit 2。
4. **傳輸**：
   - redirect 不自動跟隨，每一跳都要符合：目標是 https、沒有帳號密碼、沒有迴圈、最多 `--max-redirects` 次；
   - 只接受 HTTP 200，`Content-Encoding` 只接受 identity，送出 `Accept-Encoding: identity`；
   - proxy 依環境變數設定；憑證用系統 CA 或 `--ca-bundle`。
5. **寫入**：寫進同一目錄的隱藏檔 `.<name>.partial-<rand>`（`O_EXCL|O_NOFOLLOW`，0600），邊寫邊算 sha256。
   - `Content-Length` 和 expected bytes 不符：讀 body 之前就 exit 1；
   - 超過 expected bytes 或 `--max-bytes`：立即停止；
   - 實際長度和 `Content-Length` 不同：截斷，exit 1；
   - sha 或 size 不符：exit 1。
6. **發布**：
   1. fsync 後，先把紀錄寫進 staging；
   2. 以 hard link 把檔案發布成 `<name>`：已存在就失敗，不會取代；檔案系統不支援 hard link 時 exit 2；
   3. **最後**以同樣方式發布 `<name>.fetch.json`；
   4. 刪除 staging 檔，fsync 目錄。

   紀錄是唯一的發布標記：沒有紀錄的檔案就是未發布。
7. **中斷**：SIGINT、SIGTERM、SIGHUP，以及任何失敗，都會刪除 staging 檔，正式路徑不變。SIGKILL 會留下隱藏的 `.partial-` 檔：它永遠不會被當成正式檔，rerun 會在 `stale_partials` 列出它，但不會自動刪除。第 6 步 ii 和 iv 之間被殺掉，會留下沒有紀錄的檔案，依第 3 步處理。

**紀錄 `saccade.model_fetch_record/v1`** 包含：
- `name`、`bytes`、`sha256`；
- `integrity`：`expected_sha256_matched` 或 `local_only`；
- 固定的 `source_authentication: "not_performed"` 和 `reading`；
- `source`：`url`（**去掉 query 與 fragment**）、`expected`、`hash_reference`、`terms`，以及 `descriptor_sha256`，也就是 source 檔 bytes 本身的 sha256；
- `transport`：`final_url` 和每一跳 redirect，同樣**去掉 query 與 fragment**；另有 `content_length`；
- `fetched_utc`；
- `tool`：路徑與工具檔的 sha256。

**簽名 URL**：presigned URL 會把 token 放在 query 或 fragment。
- 工具寫進紀錄、印到 stdout／stderr 的每個 URL 都去掉這兩段，包括 source url、redirect、`final_url` 和錯誤訊息；
- 拒絕非 identity 的 `Content-Encoding` 或無法解析的 `Content-Length` 時，只印固定診斷，不回顯原始標頭值，避免伺服器透過標頭把簽名 URL 或 token 帶進輸出；
- 紀錄仍然透過 `descriptor_sha256` 綁定**完整的** source 檔，所以去掉 token 不會放鬆配對：只差 token 的兩份 source 會被當成不同來源；
- 紀錄以 0600 建立，作為額外防護。要讓別的帳號 `verify`，請自行調整權限；
- **沒有涵蓋**：把 token 放在 URL path 裡的方案認不出來；`terms.url`、`hash_reference.citation` 原樣寫進紀錄，不要在那裡放秘密；source 檔本身含完整 URL，由使用者自行保管。

## 4. 離線核對（`verify`）

不連任何網路。以下都要成立才回 `verified`（exit 0），任何一項不成立就回 `failed`（exit 1），報告列出所有原因：

- `<name>` 和紀錄都存在，而且都是一般檔案；
- 紀錄的 schema、`name`、`integrity`、`source_authentication` 都和這份 source 一致；
- 紀錄的 `source` key 恰好是 `url`、`expected`、`hash_reference`、`terms`、`descriptor_sha256`，每一個都等於從這份 source 算出的值。`url` 比的是去掉 query 與 fragment 的 URL；完整 URL 由 `descriptor_sha256` 綁定。因此 source 檔只要改了任何 byte，包括只重新排版，`verify` 就會失敗：請保存當初 `fetch` 用的那份檔案；
- `<name>` 和紀錄都讀得到。讀不到（例如權限不足）時列為原因，exit 1；
- 檔案的實際 bytes 和 sha256 等於紀錄的值，也等於 source 的 expected 值。

`local_only` 的檔案也可以 `verified`，但報告的 `integrity` 照實寫 `local_only`。核對通過**不**代表官方來源認證。即使比對的是使用者提供的 hash，結果也只是內容一致。

## 5. 操作說明

1. **選來源**：從上游自己的發布頁選檔案的 URL，閱讀並接受上游條款，把條款位置和時間寫進 `terms`。
2. **取得 expected sha256**：上游有公布 hash 時，填 `kind: upstream_published`，`citation` 寫公布 hash 的頁面。上游沒有公布時，可以用你從其他可信管道取得的值，填 `kind: user_supplied`。都沒有時，只能用 `--allow-no-expected-hash`，結果是 `local_only`，不得當成來源已核對。
3. **執行**：`fetch` → `verify`。把 source 檔和 `.fetch.json` 和模型放在一起保存；之後每次使用前都可以離線 `verify`。
4. **不要做的事**：
   - 不要把取得的權重提交進 repo，也不要公開散佈。權利由 [#547](https://github.com/raylei50653/saccade/issues/547) 決定，目前是 `local-only`。
   - 取得成功不會自動得到 VL2、A_L parity 或發行資格。

## 6. 測試矩陣

[`tests/unit/test_model_fetch.py`](../../tests/unit/test_model_fetch.py)：72 個 case，CPU。全部對 127.0.0.1 上的本機 https server 進行：測試 CA 和 leaf 憑證在測試內產生，透過 `--ca-bundle` 傳給工具。body 是約 2.2 MiB 的合成資料。不連外部網路，也不下載真實模型。

| 類別 | Case | 預期 |
|:--|:--|:--|
| 正控制 | 發布一對檔案後 `verify` | exit 0；目錄只有 `<name>` 和紀錄；`expected_sha256_matched`；`not_performed` |
| 正控制 | 302 加 307 的 https redirect | 記錄兩跳；`final_url` 和 hops 不含 query token |
| 正控制 | 重複執行 | `already_present`；請求數不變；mtime 不變 |
| 政策 | 沒有 hash、沒加旗標 | exit 2；0 個請求 |
| 政策 | 沒有 hash、加了旗標 | `local_only`；`verify` 0 且回報 `local_only` |
| 負控制（7） | sha 不符、`Content-Length` 和 size 不符、無 `Content-Length` 但超過 expected bytes、宣告長度大於實際、連線中途關閉、gzip、404 | exit 1／1／1／1／1／3／3；目錄是空的 |
| 負控制 | redirect 轉到 http | exit 2；http 目標沒有被請求 |
| 負控制 | redirect 迴圈；`--max-redirects 2` 遇到 3 跳 | exit 2；目錄是空的 |
| 負控制 | 不信任的憑證（沒有 `--ca-bundle`） | exit 3（`CERTIFICATE_VERIFY_FAILED`） |
| 負控制（15） | 不合法的 source：http、帳密、`../x`、`a/b`、`.hidden`、`*.fetch.json`、`*.partial-*`、`kind none` 卻有 sha、`upstream_published` 沒有 citation、`accepted` 為 false 或 `"yes"`、`bytes` 為 true、大寫 hex、多出的 key、重複的 key | exit 2；0 個請求 |
| 負控制 | `DIR` 是 symlink；`<name>` 是 symlink（目標不變）；`<name>` 是目錄 | exit 2 |
| 衝突 | 已發布的一對遇到另一份 source | exit 2；0 個新請求；bytes 不變 |
| 衝突 | 只有紀錄沒有檔案 | exit 2 |
| 復原 | 沒有紀錄但 bytes 相同的檔案 | `record_recovered`；inode 不變；`verify` 0 |
| 復原 | 沒有紀錄、和 expected 不符的檔案 | exit 2；0 個請求；檔案不變 |
| 復原 | 沒有紀錄、沒有 hash、和下載內容不同的檔案 | exit 2；檔案不變 |
| Lock | `DIR` 的 flock 已被佔用 | exit 2；0 個請求 |
| 離線 verify（12） | 翻轉一個 byte、截斷、刪除紀錄、竄改紀錄的 sha、竄改紀錄的 integrity、source 換 hash、source 換 URL（bytes 相同）、只竄改紀錄的 `terms.accepted_utc`、只竄改紀錄的 `descriptor_sha256`、紀錄的 `source` 多一個 key、source 換 `accepted_utc`、source 只重新排版 | exit 1，而且 server 已關閉；後五項的報告點名對應的欄位，只竄改 terms 或多一個 key 時恰好一個原因 |
| 簽名 URL | source url 的 query 和 fragment 帶 token | `fetch`、`verify`、rerun 都 exit 0；整份紀錄、stdout、stderr 都沒有 token；紀錄 `source.url` 是去掉 query 的 URL、`descriptor_sha256` 等於 source 檔的 sha256、權限 0600 |
| 簽名 URL | 同一個去 token URL、另一個 token 的 source | `verify` 1，原因只有 `descriptor_sha256` 而沒有 `source.url`；`fetch` 2，0 個新請求；輸出沒有 token |
| 簽名 URL（2） | 帶 token 的 source 遇到 404；redirect 到帶 token 的 http | exit 3／2；輸出沒有 token；目錄是空的 |
| 標頭回顯（6） | 帶 query token 的 source；`Content-Encoding`／無法解析的 `Content-Length` 分別回顯完整簽名 URL、單獨的 query token 或另一個 token | 真實 CLI subprocess exit 3；stdout 一份 JSON，stderr 同一個固定診斷；兩者都不含原始標頭值或 token；只有一次請求，目錄是空的 |
| 簽名 URL | `scrub()` 單元測試 | URL、query、fragment 都被移除 |
| CLI（6） | 沒有參數、未知子命令、缺 `--source`／`--dest`、`--max-redirects x`、未知旗標 | exit 2；stdout 一份 JSON |
| CLI | 真實 subprocess 的參數錯誤 | exit 2；stdout 是 JSON；stderr 有 usage |
| CLI（2） | `--ca-bundle` 不存在；內容不是合法 PEM | exit 2；0 個請求 |
| CLI | `DIR` 不可寫；`<name>` 不可讀 | `fetch` 2（`file system`）且目錄是空的；`verify` 1 並列出 `Permission denied`。root 執行時略過 |
| CLI | 內部錯誤（monkeypatch 讓 `cmd_verify` 拋出例外） | exit 4；`internal_error` |
| 中斷 | 下載途中 SIGINT、SIGTERM（subprocess） | 130／143；`interrupted`；目錄是空的 |
| 中斷 | 下載途中 SIGKILL，然後 rerun | 殘留的是隱藏 `.partial-` 檔，沒有 `<name>`，`verify` 1；rerun `fetched` 並在 `stale_partials` 列出殘留；`verify` 0 |

本機 mutation 檢查（只在本機，結果沒有提交）：分別拿掉 redirect 的 https 檢查、sha 比對、staging 清理、紀錄和 source 的欄位比對，每一項都會讓至少一個 case 失敗。第二輪（#612 owner review 之後）又加了九項：紀錄寫入未去 token 的 URL、不比對 `terms`、不檢查 `source` key 集合、紀錄改回 0644、argparse 預設的 `error()`、拿掉 `OSError` 處理、拿掉 `--ca-bundle` 處理、拿掉內部錯誤處理、`verify` 不捕捉讀檔錯誤，各自至少讓一個 case 失敗。第十項是把 `scrub()` 改成不做事：端對端 case 抓不到（傳輸錯誤原因通常不含 URL），因此另加單元測試，加入後這項也會失敗。

## 7. 尚未實作或整合（仍然 OPEN）

- **Ultralytics 的隱性下載路徑仍然存在**：`YOLO()` 在檔案不存在時會呼叫 `attempt_download_asset`。本片沒有改任何 exporter，所以**不得宣稱這個路徑已修復**。exporter 改成只接受已存在、並經過 `verify` 的檔案，屬於第三片（v2 schema／exporter／bundler），需要另外授權。
- exporter 和 bundler 都還不讀 `.fetch.json`，原始權重到 ONNX、engine、TorchScript 的配對紀錄也還沒有（第三片）。
- `--check-bundle`、runtime 無聯網的負控制、package README（S2-3）。
- 沒有續傳；沒有自動清理 SIGKILL 殘留；只支援 Linux。
- 沒有對真實上游做過驗證：真實上游的 redirect 鏈、proxy 和憑證行為都 **UNVERIFIED**。
- V5 `UNRESOLVED`、installed package `UNVERIFIED`、#547 `local-only` 都沒有改變。
