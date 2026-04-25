import subprocess
import re
import os
import itertools
from pathlib import Path

# --- 參數配置 ---
ENGINE = "models/yolo/yoloe-26m-seg-pf_person_embed_batch4.engine"
DATA_ROOT = "datasets/MOT17"
# 測試序列 (建議先用單一序列如 MOT17-05-SDP 加速迭代)
SEQUENCES = "MOT17-05-SDP" 
OUTPUT_BASE = "results/grid_search"

# 定義要搜索的參數範圍 (擴展版)
search_space = {
    "conf_threshold": [0.2, 0.3, 0.4],
    "semantic_threshold": [0.93, 0.95, 0.97, 0.98],
    "mahalanobis_threshold": [6.25, 9.49, 13.0],
    "semantic_ema": [0.85, 0.9, 0.95],
    "semantic_min_lost_frames": [1, 2, 4],
    "spatial_gate": [0.15, 0.2, 0.25]
}

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        # 這裡不直接 print result.stderr 以免輸出過多，但在解析失敗時提供線索
        return f"ERROR: {result.stderr}\n{result.stdout}"
    return result.stdout

def main():
    best_mota = -1.0
    best_params = {}
    
    # 建立所有組合
    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"🚀 Starting Expanded Grid Search: {len(combinations)} combinations.")
    
    for i, params in enumerate(combinations):
        out_dir = Path(OUTPUT_BASE) / f"run_{i}"
        
        # 1. 執行感知評估
        ld_path = (
            "/home/ray/developer/ai/saccade/build:"
            "/home/ray/developer/ai/saccade:"
            "/home/ray/developer/ai/saccade/.venv/lib/python3.12/site-packages/tensorrt_libs:"
            "/home/ray/developer/ai/saccade/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:"
            "/opt/cuda/lib64"
        )
        eval_cmd = (
            f"PYTHONPATH=/home/ray/developer/ai/saccade/build:/home/ray/developer/ai/saccade "
            f"LD_LIBRARY_PATH={ld_path} "
            f"uv run scripts/eval/mot17.py "
            f"--engine {ENGINE} "
            f"--sequences {SEQUENCES} "
            f"--output {out_dir} "
            f"--conf-threshold {params['conf_threshold']} "
            f"--semantic-relink "
            f"--semantic-threshold {params['semantic_threshold']} "
            f"--semantic-mahalanobis-threshold {params['mahalanobis_threshold']} "
            f"--semantic-ema {params['semantic_ema']} "
            f"--semantic-min-lost-frames {params['semantic_min_lost_frames']} "
            f"--semantic-spatial-gate {params['spatial_gate']} "
            f"--no-reid"
        )
        
        if i % 10 == 0: # 減少日誌噪音
            print(f"[{i+1}/{len(combinations)}] Processing... (Current Best: {best_mota}%)")

        run_command(eval_cmd)
        
        # 2. 執行指標計算
        mota_cmd = f"PYTHONPATH=. uv run scripts/eval/calculate_mota.py --results {out_dir}"
        mota_output = run_command(mota_cmd)
        
        # 3. 解析 MOTA
        try:
            lines = mota_output.splitlines()
            found = False
            for line in lines:
                # 尋找包含序列名稱或 OVERALL 的行
                if SEQUENCES in line or "OVERALL" in line:
                    # 使用正規表示法抓取所有百分比數值
                    # MOTA 通常是該行中第 6 個百分比數值 (IDF1, IDP, IDR, Rcll, Prcn, MOTA)
                    percentages = re.findall(r"([-+]?\d*\.\d+|\d+)%", line)
                    if len(percentages) >= 6:
                        mota_val = float(percentages[5]) # 第 6 個是 MOTA
                        found = True
                        
                        print(f"   -> MOTA: {mota_val}%")
                        
                        if mota_val > best_mota:
                            best_mota = mota_val
                            best_params = params
                            print(f"   🌟 New Best!")
                        break
            
            if not found:
                print(f"   ⚠️ Could not find MOTA in output for {SEQUENCES}")
                if len(lines) < 5:
                    print(f"   Raw output: {mota_output[:200]}")
                    
        except Exception as e:
            print(f"   Failed to parse output: {e}")
            print(f"   Full output for debugging:\n{mota_output}")

    print("\n" + "="*30)
    print(f"✅ Grid Search Completed!")
    print(f"Best MOTA: {best_mota}%")
    print(f"Best Parameters: {best_params}")
    print("="*30)

if __name__ == "__main__":
    main()
