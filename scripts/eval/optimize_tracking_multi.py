import optuna
import subprocess
import re
import os
import shutil
from pathlib import Path

# --- 參數配置 ---
ENGINE = "models/yolo/yoloe-26m-seg-pf_person_embed_batch4.engine"
SEQUENCES = "MOT17-05-SDP"
OUTPUT_BASE = "results/optuna_multi_obj"
DB_PATH = "sqlite:///tracking_optimization_multi.db"

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

def objective(trial):
    # 定義搜尋空間
    params = {
        "conf_threshold": trial.suggest_float("conf_threshold", 0.1, 0.5),
        "semantic_threshold": trial.suggest_float("semantic_threshold", 0.85, 0.99),
        "mahalanobis_threshold": trial.suggest_float("mahalanobis_threshold", 4.0, 15.0),
        "semantic_ema": trial.suggest_float("semantic_ema", 0.8, 0.98),
        "semantic_min_lost_frames": trial.suggest_int("semantic_min_lost_frames", 1, 5),
        "spatial_gate": trial.suggest_float("spatial_gate", 0.1, 0.3),
    }

    trial_id = trial.number
    out_dir = Path(OUTPUT_BASE) / f"trial_{trial_id}"
    
    # 執行感知評估
    eval_cmd = (
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
    
    # 執行指標計算
    mota_cmd = f"uv run scripts/eval/calculate_mota.py --results {out_dir}"
    mota_output = run_command(mota_cmd)
    
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # 解析多個指標 (IDF1 是第一個百分比, MOTA 是第六個百分比)
    try:
        lines = mota_output.splitlines()
        for line in lines:
            if SEQUENCES in line:
                percentages = re.findall(r"([-+]?\d*\.\d+|\d+)%", line)
                if len(percentages) >= 6:
                    idf1 = float(percentages[0])
                    mota = float(percentages[5])
                    # 抓取 IDs (通常是該行中較大的整數，我們用 index 定位)
                    # IDF1 IDP IDR Rcll Prcn GT MT PT ML FP FN IDs FM MOTA MOTP
                    parts = line.split()
                    ids = float(parts[11])
                    
                    # 我們回傳 (MOTA, IDF1) 給 Optuna 優化
                    # 同時我們也可以將 IDs 記錄為 User Attribute 方便後續分析
                    trial.set_user_attr("IDs", ids)
                    return mota, idf1
    except Exception as e:
        print(f"Error parsing: {e}")
    
    return 0.0, 0.0

def main():
    # 多目標優化: 同時最大化 MOTA 與 IDF1
    study = optuna.create_study(
        study_name="saccade_multi_obj_v1",
        storage=DB_PATH,
        load_if_exists=True,
        directions=["maximize", "maximize"]
    )
    
    print(f"🚀 Multi-objective Optimization (MOTA & IDF1)")
    print(f"📈 Results stored in: {DB_PATH}")
    
    try:
        study.optimize(objective, n_trials=50)
    except KeyboardInterrupt:
        pass

    print("\n" + "="*30)
    print("✅ Multi-objective Optimization Completed!")
    print(f"Found {len(study.best_trials)} Pareto optimal trials.")
    for i, trial in enumerate(study.best_trials):
        print(f"Trial {trial.number}: MOTA={trial.values[0]}%, IDF1={trial.values[1]}%, Params={trial.params}")
    print("="*30)

if __name__ == "__main__":
    main()
