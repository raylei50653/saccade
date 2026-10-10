<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-10 -->
<!-- doc-module: cross -->

# #549 S0：模型／runtime 依賴盤點與 unchanged-model freeze

本文件是 [#549 S0](https://github.com/raylei50653/saccade/issues/549) 的文件盤點，
source baseline 為 `b56ad9f4cd4d982fc3f0b293206a32f7c8675332`（#569 合併後）。
逐檔 bytes／完整 SHA-256、實際 availability、producer／consumer 與證據來源保存在
[inventory JSON](model_runtime_inventory_549.json)。這是 S0 snapshot，**不是 S1 bundle schema**。
跨 Issue 排序只在 [#550](https://github.com/raylei50653/saccade/issues/550)；本文件不批准
S1 設計、S2 實作、模型替換、支持範圍或公開發行。#547 的 `local-only`／
`owner_confirmation=null` 仍由 [license audit](../../shipping/license_audit.json) 擁有。

## 1. 覆蓋範圍與證據種類

盤點以現有文件、實際 load sites、checked-in runtime defaults／presets 與本機 bytes 為準：

- **Native required**：installed `saccade_track` 的三個 loaded artifacts，加上 config、
  lineage、attestation；checkpoint／pretrained 欄位是 provenance references，native 不開啟它們。
- **其他 runtime surfaces**：Python eval、選配 ReID、legacy C++ detector、multistream、
  cognition／text 的模型入口。任意 caller path／遠端模型名以 unresolved surface 記錄，
  不假造唯一檔名或 hash；存在與 source-readable 不代表已批准支持或已驗證可執行。
- **Training／eval-only**：ONNX、head redesign、訓練祖先與 caches、外部比較器等保留分類；
  全部歷史 checkpoint genealogy 仍由 [#421 inventory](../research/training/training_lineage_inventory.md)
  擁有，不複製成另一份訓練目錄。

`observed_bytes` 是本次重新讀取檔案的身分；`historical_record` 是 linked evidence 的記錄；
`source_inspected` 是程式判讀。S0 沒有反序列化 checkpoint、重建模型、下載模型或重跑 GPU parity。
未知的來源、授權、shape／版本與 availability 逐項保留 `unresolved`，hash 不證明權利或性能。

## 2. Native unchanged-model baseline

安裝路徑全部是 `PREFIX/share/saccade/<下列 repo-relative path>`，`--model-root` 指向 `PREFIX/share/saccade`。
這個名稱為 model root 的目錄也放 `configs/`、`build/`，不是只有 weights。
[install_model_root.cmake](../../shipping/cmake/install_model_root.cmake) 保留原路徑而不改寫 frozen JSON。

| ID／角色 | 精確 repo-relative path | bytes | SHA-256 |
|:--|:--|--:|:--|
| N01 backbone | `models/yolo/yolo26s_backbone_640_best.engine` | 20298516 | `2ef3d4d40dfb670982cbbb98e6ed7d07e5b1a590cfa126d5ccf7342ee1579ce4` |
| N02 TorchScript head | `models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.pt` | 45656246 | `1663ec97022879e8078fb5b9dc3b65f4d2b76e4f8df6d5848300c2b3c999d879` |
| N03 model-coupled scan operator | `build/libsaccade_scan_torchop.so` | 3517632 | `aa84cccd5c5094b33d63217eeb377452ae19c69cd1382ec7de8ba96d0d8d31c4` |
| N04 resolved config | `configs/shipping/mamba_whole_graph.resolved.json` | 28648 | `6576956f37c1a3f85134febf8ab7be86507beddbe90ac525a5947c91bd55ddf9` |
| N05 frozen lineage | `models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json` | 7539 | `677bc82320657a9ae289e78699c7a6f86e890fc629d1bb8189c08d35b57a277a` |
| N06 realization attestation | `configs/shipping/mamba_head_realization.attestation.json` | 4052 | `33289884c3586cc062a1c6ddbf84d7a45993070514664734329d809c7bae351c` |

N01–N06 在本機 source/build、PR-12 r2 loose tree、#536 loose tree／installed prefix，以及
保留的 C3、C4、#536 tarball 內逐檔相同。舊 C1–C4 loose trees 的部分 binary 已不在本機；
不能把目錄存在寫成完整 tree 可重播。比較只讀取 named model members，沒有重新驗證舊
整包 digest／signature／runtime，詳細 availability 在 JSON 的 `retained_baselines`。

### 2.1 Producer、來源與資料

| 角色 | Producer／來源證據 | 血統／資料與未解項 | Consumer |
|:--|:--|:--|:--|
| N01 | [backbone exporter](../../scripts/model/export_yolo_backbone_ckpt.py)、[TRT builder](../../scripts/model/build_yolo.py) 是現有產製入口；實際 engine build chain **沒有 manifest 綁定** | #421 對 sibling ONNX 的 initializer match 指向 `runs/gated_det_v1/best.ckpt`；engine↔ONNX 是 role claim，不能把該結果升成 engine bytes 的來源證明。YOLO26／Ultralytics 血統、MOT17 training 是文件記錄；完整 pretrained dataset／權利 unresolved | [DetectorHost](../../shipping/src/detector_host.cpp) → `TRTEngine` |
| N02 | [TorchScript exporter](../../scripts/model/export_headline_mamba_head_torchscript.py)，frozen tool commit `37d4d21d0fb5a702e69f2b8c8f7b6aa4683f3864`；N05 綁定 `runs/mamba_gt_v14replica_t3_t1/best.ckpt` | #421 的 T3→T1 Phase B、MOT17、YOLO26s 與 teacher／distillation chain；舊 checkpoint 沒有 recorded base/teacher SHA gate，匯出時改以 inventory ckpt hash 綁定。資料權利及 head derivative obligations unresolved | `torch::jit::load`，單幀 `T=1` head |
| N03 | [scan op](../../src/tracking/mamba_scan_torchop.cpp)、[CUDA scan](../../src/tracking/mamba_scan.cu)；N06 列 source blobs 與 realized build | 專案 operator、LibTorch／CUDA ABI；不是 weights。第三方條款由 #547 擁有；原 lineage 的 op hash 已不存在，現行 realized hash 由 attestation 取代 | `dlopen`，`saccade_native::selective_scan_fwd` |
| N04 | [resolved config exporter](../../scripts/model/export_resolved_shipping_config.py)；preset `mamba_whole_graph.yaml` 的 hash／source claims | 是 native supported subset 的組態，包含 mamba/pretrained/teacher path claims；不載入這些 `.ckpt`／pretrained bytes | [resolved config](../../shipping/src/resolved_config.cpp)、native plans |
| N05 | 與 N02 同次 exporter；`saccade.head_artifact_lineage_torchscript/v1` | checkpoint、head、backbone、runtime flags 與 inventory 的記錄；caller-controlled JSON 不是獨立 authenticated provenance | [plan_detector_files](../../shipping/src/detector_plan.cpp) |
| N06 | #465 PR-8 realization、PR-10 re-attestation；[resolved config §14](native_runtime_resolved_config.md) 保存重現結果 | 只對 named operator realization 與接受的 `A_L` MOT txt 作歷史宣稱；native 只核對記錄內 consistency，不重跑 reproduction | `plan_detector_files`；installed packaging required，直接 CLI optional |

### 2.2 Compatibility 與 I/O

Frozen native evidence 是同一台 WSL2 的 RTX 5070 Ti Laptop／sm_120、Ubuntu 24.04 container／glibc 2.39、
TensorRT `10.16.1.11`、torch `2.11.0+cu130`／CUDA 13.0／cuDNN 91900 的
[#465 A_L](native_runtime_closeout.md)。其他 driver／原生 Linux host 尚未驗證；
不能由 version 字串相同推出支持。
原 head／operator build ABI 需精確配對，N03 的 absolute build RUNPATH 是既有 named exception。

N01 預期 CUDA float32 input `[1,3,640,640]`，三個 FPN tensors 依 ordinal 為
`[1,128,80,80]`、`[1,256,40,40]`、`[1,512,20,20]`。
N02 的 outputs 依 ordinal 是 `cls_p3..p5`（80 classes）、`reg_p3..p5`（4 LTRB channels），
float32、static batch 1；temporal blocks bypass、native scan calls 3。
Runtime 檢查實際 TRT shape／input-output order 與 head output shape/dtype/device，但**沒有查 TRT I/O dtype**。
SM／TRT／torch build metadata 不是完整 runtime compatibility predicate；缺口見 §5。
Native ingest 產生 RGB／0–1，detector 在 backbone 前做 640 stretch resize；
S2 做 sigmoid／class-max、top-k、LTRB decode 與 xyxy scaling。
ReID／TTA／letterbox／TRT head 與其他未支持 config 由 native plans 拒絕，詳見
[detector plan](../../shipping/src/detector_plan.cpp) 與 [S2](../../shipping/src/detector_s2.cu)。

## 3. Native 之外的 runtime 載入面

下表是 consumer 覆蓋；每個 concrete default／preset artifact 的 exact path、bytes、full SHA、
producer、ancestry／data evidence 與 availability 都在 JSON 的 `artifacts`。
空白 caller path、mutable model ID 與 cache 未解析時保留 null bytes／SHA，不能算作已固定模型。

| Surface | 模型／入口 | Runtime consumer／fallback | 與 native freeze 的關係 |
|:--|:--|:--|:--|
| P01 Python detector | YOLO raw s/m/l engines、pose/batched engines；`--engine`／`--pose-engine` | `TRTYoloDetector`／batching／concurrent routes；missing/deserialize behavior 以 consumer 為準 | 非 native model-root；raw CLI default 是 m/960，不能混成 s/640 |
| P02 Mamba eager／whole-graph | `mamba_ckpt`、YOLO pretrained、teacher config／weights、`fpn_backbone_engine`、optional `mamba_head_engine` | TRT backbone path 已可跳過 YOLO construction；無 TRT path 用 PyTorch teacher backbone。`mot17.py` 的 preset／explicit head engine 直接選用；只有 `--mamba-trt` auto branch 檢查 default 檔案後選用或回 PyTorch | s Python head仍讀 `.ckpt`；native N02 才是 frozen TorchScript。m preset 是另一模型組合 |
| P03 legacy C++／multistream | `models/yolo/mamba_head_best.pt`＋backbone、可覆寫 head/engine | C++ batched Mamba、multistream server／eval；server 在 temporal 或缺 head script 時略過 C++ head；import／construction errors 會傳出，`cpp_ptr` 的 detail／channel mismatch 會拒絕 | 這份 legacy head 不是 N02，#421 未記錄其 source checkpoint |
| P04 Python teacher／temporal／gated／JDE | `--teacher-head-ckpt`／backbone、temporal/conditioned ckpt、`fpn_reid_ckpt`／`jde_proj_ckpt` | 依 config 選路；teacher、Mamba head、trained FPN projection 是獨立 runtime inputs | 選配／研究 eval，不 install 到 shipping |
| P05 embedding／ReID | SigLIP2、SigLIP2-ReID、DINOv2、TransReID、OSNet、FastReID、MobileNetV4 的 7 個 `_DEFAULT_ENGINE` | [TRTFeatureExtractor](../../src/saccade/perception/feature_extractor.py)，C++ unavailable 時同檔 TRT Python backend；`--reid-engine-path`／Cheb-GR 可覆寫 | native ReID off；FastReID default 本機 unavailable。MobileNet GPU-decode 是另一路 artifact |
| P06 learned JSON policy | external-FP logistic／cascade JSON，可 caller 指定 | [external FP](../../src/saccade/perception/eval/external_fp_model.py)、evaluator policy loader | model weights 也可能在 JSON；不因 extension 排除。未啟用則不讀 |
| P07 text／cognition | `google/siglip2-base-patch16-224` model＋processor、`BAAI/bge-small-en-v1.5`、Ollama `llama3` | [text encoder](../../src/saccade/perception/text_encoder.py) `from_pretrained`；[orchestrator](../../src/saccade/cognition/orchestrator.py) optional RAG，setup failure 被捕捉 | 未固定 remote revision/digest／resolved cache／service bytes；可能 fetch 或服務端選 tag，不享有 native no-download 邊界 |

ADR 023 是 Proposed、較早的 Python coupling snapshot。它對當時的 import 盤點與 model-lineage
風險仍有用；現行 TRT path 是否載 YOLO 以
[MambaGatedDetector](../../src/saccade/perception/temporal_yolo/mamba_gated_detector.py) 的分支為準。
不得重複實作已存在的 TRT loader，也不得把任何 P01–P07 自動提升成已批准 shipping 支持。

Training ancestry／caches、ONNX、被否決的 TRT head redesign、calibration inputs 與外部 YOLOX／
Ultralytics comparator 屬 training/export/eval-only。外部比較器模型不代表 Saccade runtime 預設；
JSON 另保留已列出 concrete references，任意 CLI overrides 的實際檔案仍須下一次 capture。

## 4. C1–C4、package 與 identity 的已存在機制

| 面 | Source-inspected 已存在 | 證據／限制 |
|:--|:--|:--|
| C1/C2 tree／native boundary | CMake shipping component 複製三個 bound artifacts；lineage 必須符合 attestation hash，operator 使用 `SACCADE_ATTESTED_OP_LIBRARY`；27 vendor objects、launcher/auditor、entrypoint pin、no measurement surface | [phase C](native_runtime_phase_c_scope.md)、[closeout](native_runtime_closeout.md) 是歷史基準；本次 static installed check 13/13 PASS |
| C3 installer／manifest | builder 要求 clean tree／current publication；MANIFEST 列每檔 SHA／size／mode、model-root members、coordinate、entrypoint；installer 做 digest → staging → exact file checks → atomic no-replace install。開發端 [package checker](../../scripts/native/check_shipping_package.py) 拒絕非法 archive path／symlink／hardlink headers；installer 檢查 regular/non-symlink files，沒有獨立 hardlink-count gate | [package builder](../../scripts/native/build_shipping_package.py)、[installer](../../shipping/package/install.sh)。`--verify` 信任 tree 內 MANIFEST，unsigned digest 不認證來源；只確認 ext4 rename 邊界 |
| C4 signature／distribution | minisign release-set opt-in verification、wrong key/tamper controls 已存在；公開發行 policy 要求簽章 | [closeout §6](native_runtime_closeout.md#6-公開散佈的未解項目)、#547 release key／owner decision 尚未完成；S0 不產生或公開任何模型包 |
| Runtime integrity | Gate A CUDA-free checks actual three file hashes；Gate B 重驗 hash 後 dlopen／TorchScript／TRT load；attestation 覆寫 realized op hash並綁定 frozen lineage；shape／graph／runtime flags checks | Gate A 使用同一 DetectorPlan 給 Gate B，避免重新 parse 的不同 plan；hash/check→open TOCTOU 仍在。Native 模型載入沒有替代模型或下載 fallback；nvJPEG decode backend fallback 是另一層 |
| Runtime identity | current [coordinate publication](runtime_identity.generated.json)、entrypoint pin、manifest coordinate 與歷史 archives | #536 publication 的 probe 用 `mamba_whole_graph_m`，不是 native s model identity 的替代；report v3 `identity.level=null`，沒有 portable model-bundle identity integration |

現行 entrypoint 是 [pin](../../shipping/entrypoint_pin.json) `fbab8b79710f52a6f6c668b956c42e1650fe59e3f0d1eeeb525e914e47a07dde`
（8862856 bytes），build source `e0eab2f7caa3c495c74b8cf00e45788498d573cc`。
closeout 的 `92f74ef4…` 是原 PR-C2 歷史 pin；不能當現行 pin。
[#536 package evidence](native_runtime_package_repin_536.md) 保存新 installed Completion／Gate A／
七序列 EXACT 的 bounded evidence；S0 只核對保存的模型 bytes 與 static gate，不重新宣稱該量測。
N06 frozen lineage 原 op hash `cfea782f…`、現行 realized hash `aa84cccd…` 的關係保持原樣。
完整 previous attestation／C3/C4/current identity references 及檔案身分在 JSON。

## 5. Existing-vs-missing：交給 S1 的缺口

這些是盤點輸入，沒有選定修正方案或授權實作。

| ID | 現況 | 缺少／需要 S1 定義 |
|:--|:--|:--|
| G01 expected identity trust | lineage／optional attestation 驗證 actual bytes 與 caller claims consistency；release signature 是 package 邊界的認證機制 | runtime-side trusted release model identity／key／signature authority；協調替換 JSON＋artifact 可形成另一組 consistent claims，現行 hash gate不單獨認證 producer |
| G02 bundle／pairing contract | frozen lineage v1 描述 head／backbone／op，resolved config subset checks 已存在 | independently versioned model bundle、size/pairing／preprocess/decode／migration contract、source of expected values；不能把 S0 inventory JSON 當 loader schema |
| G03 model-root paths | `resolve_model_path` 接受 absolute path；relative 只 concatenate，沒有 `..` 或 symlink containment gate | 模型 relocation／traversal／symlink／artifact-open TOCTOU 規則。Package extraction protection與 loader auditor 的 scan-op pathname restriction 不能代替 head/backbone model-root confinement |
| G04 compatibility | TRT deserialize 與 runtime shapes／head dtype/device／scan count 檢查；frozen version/SM metadata存在 | explicit supported SM/TRT/LibTorch/CUDA ABI／precision matrix與 pairing版本；TRT I/O dtype 檢查缺少，GPU family/version 欄位未作完整 gate；不能以成功 deserialize 等同契約支持 |
| G05 model-free extraction | `--model-root` 已存在；目前 CMake/package 預期六個 model-root files，build/tree/static checks與例子依賴 bundled模型 | model-free runtime 與 private bundle 的獨立 manifests/signatures、binding、install/rollback、CI artifact inclusion contract；移走檔案本身不完成此工作 |
| G06 provenance／rights | #421 記錄 source ancestry；N01 engine source attribution unresolved；#547 M-1/L-4追蹤權利／SDK可能交互限制 | 每個 channel 的 model/data rights evidence與owner/legal decision；inventory不裁決法律 |
| G07 evidence／identity | existing `A_L` frozen outputs、#536 package EXACT、coordinate archive可按原有效條件引用 | substituted model的新evaluation／identity、positive-negative matrix、model coordinates與migration；CLI名字相同不轉移 HOTA/IDF1/MOTA/FPS |

## 6. 權利未知項與 owner 路由

權利的唯一裁決入口是 [#547](https://github.com/raylei50653/saccade/issues/547)，
[license audit M-1／L-4](../../shipping/license_audit.json) 與
[ADR 023](../decisions/023-ultralytics-runtime-decouple.md)。本次沒有重新檢索官方條款或提出法律結論。

| 項目 | 現有記錄 | Remaining unknown／decision owner |
|:--|:--|:--|
| YOLO26／Ultralytics derivatives | ADR／M-1 記錄 AGPL lineage 與 commercial/legal/alternative owner options | exact pretrained/model licence與各 binary義務、商業授權或法律判斷、公開／私有channel grant；#547 release owner |
| Mamba head | 自有 implementation，但 teacher／Ultralytics loss與dataset ancestry存在 | derivative obligations與trained-weight redistribution權利不能由自有程式license推斷；#547 release owner，#421供 ancestry evidence |
| MOT17與其他資料 | checkpoint historical args／training protocols記錄 MOT17；選配ReID另涉多資料集 | exact dataset snapshots、terms及trained-artifact使用／散佈permission未裁決；#547取得並review，producer owner補文件 |
| Embedding／HF／Ollama | JSON列repo／model ID、exporter或training文檔；無revision pin的surfaces保持unresolved | per-version model/licensor/data/terms與private/public grants；不能從HF名稱或engine可載入推斷；#547與各model producer |
| Model licence／SDK interaction | L-4與M-1分別記錄NVIDIA條件與open-source/model疑問 | 是否／如何相容由owner/legal/vendor confirmation裁決；runtime audit不證合規或違規 |

## 7. 本次檢查、重播與 stop point

盤點初次保存的本機檢查：61/61 present artifact的bytes／SHA相符，六列native表與JSON相符，
101個source references存在；installed static／manifest為13/13 PASS。
3038個relative doc links、strict doc structure、`--mode attested` checker與whitespace檢查通過。
Doc structure仍有兩個既有index warnings；identity checker未重新計算host environment、legacy
runtime inputs或probe。Snapshot的raw report path／SHA保存在JSON的`validation`；分類為
**local-only evidence**。模型與raw reports仍在gitignored的`models/`、`runs/`、`build/`、`results/`，
本PR不攜帶這些bytes；這些檢查不能升格為clean-checkout reproducible evidence或CI模型重播。
PR修訂後的文件／CI檢查另記在PR，不改寫原snapshot的raw report紀錄。
重播需保留本交付的inventory JSON與列出的gitignored artifacts；`source_commit` 是所檢視程式的
座標，該舊commit尚不含本盤點文件。若另checkout它，須從交付分支另保留JSON。
unavailable／remote surfaces保持unresolved，不下載或換同名模型。只讀bytes replay如下：

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
d = json.loads(Path('docs/reference/model_runtime_inventory_549.json').read_text())
checked = 0
for a in d['artifacts']:
    if a['availability'] != 'present':
        continue
    p = Path(a['path'])
    assert p.is_file(), a['path']
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    assert p.stat().st_size == a['bytes'], a['path']
    assert h.hexdigest() == a['sha256'], a['path']
    checked += 1
print(f'{checked} present artifact identities match; unresolved surfaces unchanged')
PY

.venv/bin/python scripts/native/check_shipping_bundle.py static \
  --tree results/536_package_repin/full_b24c27f/installed/saccade \
  --manifest --report /tmp/549-static-replay.json
.venv/bin/python scripts/tools/check_doc_links.py
.venv/bin/python scripts/tools/check_doc_structure.py --strict
.venv/bin/python scripts/tools/check_runtime_identity_staleness.py --mode attested
git diff --check
```

S0 stop point：文件／byte inventory完成後交審；未知來源或權利保持OPEN，不需要改模型才能完成盤點。
S1另需自己的設計審查／owner approval；S2須有批准後的bounded PR。#549保持open，
`RUNTIME_PACKAGE_READY`／`MODEL_BUNDLE_READY`／`PUBLIC_DISTRIBUTION_READY` 不由本盤點宣告。
