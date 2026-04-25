import optuna
import subprocess
import re
import os
import shutil
from pathlib import Path

# --- 參數配置 ---
ENGINE = "models/yolo/yoloe-26m-seg-pf_person_embed_batch4.engine"
SEQUENCES = "MOT17-05-SDP"
OUTPUT_BASE = "results/optuna_optimization"

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

def objective(trial):
    # 1. 定義搜尋空間 (像梯度一樣自動收斂)
    params = {
        "conf_threshold": trial.suggest_float("conf_threshold", 0.1, 0.5),
        "semantic_threshold": trial.suggest_float("semantic_threshold", 0.9, 0.99),
        "mahalanobis_threshold": trial.suggest_float("mahalanobis_threshold", 4.0, 15.0),
        "semantic_ema": trial.suggest_float("semantic_ema", 0.8, 0.98),
        "semantic_min_lost_frames": trial.suggest_int("semantic_min_lost_frames", 1, 5),
        "spatial_gate": trial.suggest_float("spatial_gate", 0.1, 0.3),
    }

    trial_id = trial.number
    out_dir = Path(OUTPUT_BASE) / f"trial_{trial_id}"
    
    # 2. 執行感知評估
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
    
    run_command(eval_cmd)
    
    # 3. 執行指標計算
    mota_cmd = f"PYTHONPATH=. uv run scripts/eval/calculate_mota.py --results {out_dir}"
    mota_output = run_command(mota_cmd)
    
    # 清理磁碟空間 (選擇性)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # 4. 解析 MOTA
    try:
        lines = mota_output.splitlines()
        for line in lines:
            if SEQUENCES in line:
                percentages = re.findall(r"([-+]?\d*\.\d+|\d+)%", line)
                if len(percentages) >= 6:
                    return float(percentages[5]) # MOTA
    except Exception:
        pass
    
    return 0.0 # 失敗時返回 0

def main():
    # 建立 Optuna 研究物件，使用 SQLite 儲存過程
    # 檔案名稱為 mota_optimization.db
    db_path = "sqlite:///mota_optimization.db"
    study_name = "saccade_mota_optimization"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=db_path,
        load_if_exists=True,
        direction="maximize"
    )
    
    print(f"🚀 Optimization Process is being stored in: {db_path}")
    print(f"📈 Study Name: {study_name}")
    
    # 執行 50 次試驗
    try:
        study.optimize(objective, n_trials=100)
    except KeyboardInterrupt:
        print("\n🛑 Optimization interrupted by user. Progress is saved in DB.")

    print("\n" + "="*30)
    print("✅ Optimization Completed!")
    print(f"Best MOTA: {study.best_value}%")
    print(f"Best Parameters: {study.best_params}")
    print("="*30)

if __name__ == "__main__":
    main()
