import sys
import subprocess
import re
import csv
import time

SRC_FILE = "src/tracking/tracker_gpu.cu"
REBUILD_SCRIPT = "bash scripts/native/rebuild.sh"
EVAL_CMD = "uv run python scripts/eval/mot17.py --detector SDP --match-thresh 0.78 --semantic-threshold 0.91 --cross-tile-merge > /dev/null 2>&1"
CALC_CMD = "uv run python scripts/eval/calculate_mota.py --detector SDP"
REPORT_FILE = "results/ablation_a7_quality.csv"

def get_base_content():
    with open(SRC_FILE, "r") as f:
        return f.read()

def patch_and_rebuild(content, patch_code):
    # Regex to find the ADR 017 block and replace it
    pattern = re.compile(
        r"(\s*// ADR 017: Quality-Aware Sinkhorn Prior\n).*?(float p = .*?)\n",
        re.DOTALL
    )
    
    if not pattern.search(content):
        print("Could not find the target block in tracker_gpu.cu")
        sys.exit(1)
        
    new_content = pattern.sub(patch_code + "\n", content)
    
    with open(SRC_FILE, "w") as f:
        f.write(new_content)
        
    print("Rebuilding C++ extension...")
    result = subprocess.run(REBUILD_SCRIPT, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr.decode()}")
        sys.exit(1)

def run_eval():
    print("Running MOT17 eval (SDP)...")
    subprocess.run(EVAL_CMD, shell=True)
    
    print("Calculating MOTA...")
    result = subprocess.run(CALC_CMD, shell=True, capture_output=True, text=True)
    
    # Parse OVERALL metrics
    # OVERALL      44.3% 55.0% 37.3% 50.6% 74.2% 546 118 256 172 19799 55469 744  1714 32.3% 0.211 376 333  85      112297
    metrics = {}
    for line in result.stdout.split('\n'):
        if line.startswith("OVERALL"):
            parts = line.split()
            # Headers: IDF1 IDP IDR Rcll Prcn GT MT PT ML FP FN IDs FM MOTA MOTP IDt IDa IDm num_objects
            try:
                metrics['IDF1'] = parts[1]
                metrics['MOTA'] = parts[14]
                metrics['IDs'] = parts[12]
                metrics['FN'] = parts[11]
                metrics['FP'] = parts[10]
            except IndexError:
                print(f"Failed to parse line: {line}")
    return metrics

VARIANTS = {
    "baseline (No Penalty)": """            // ADR 017: Quality-Aware Sinkhorn Prior
            float p = expf(-lambda * cost);""",
            
    "v1_score_only": """            // ADR 017: Quality-Aware Sinkhorn Prior
            float p = expf(-lambda * cost) * score;""",
            
    "v2_aspect_only_soft": """            // ADR 017: Quality-Aware Sinkhorn Prior
            float aspect_penalty = 1.0f;
            if (det_boxes) {
                const float* b2 = det_boxes + d * 4;
                float aspect = (b2[2] - b2[0]) / (b2[3] - b2[1] + 1e-6f);
                if (aspect > 0.8f) aspect_penalty = fmaxf(0.5f, 1.0f - (aspect - 0.8f));
                else if (aspect < 0.15f) aspect_penalty = fmaxf(0.5f, 1.0f - (0.15f - aspect) * 5.0f);
            }
            float p = expf(-lambda * cost) * aspect_penalty;""",
            
    "v3_score_and_aspect (A7 Initial)": """            // ADR 017: Quality-Aware Sinkhorn Prior
            float aspect_penalty = 1.0f;
            if (det_boxes) {
                const float* b2 = det_boxes + d * 4;
                float aspect = (b2[2] - b2[0]) / (b2[3] - b2[1] + 1e-6f);
                if (aspect > 0.8f) aspect_penalty = fmaxf(0.5f, 1.0f - (aspect - 0.8f));
                else if (aspect < 0.15f) aspect_penalty = fmaxf(0.5f, 1.0f - (0.15f - aspect) * 5.0f);
            }
            float p = expf(-lambda * cost) * score * aspect_penalty;""",
            
    "v4_additive_cost_offset": """            // ADR 017: Quality-Aware Sinkhorn Prior
            float aspect_penalty = 1.0f;
            if (det_boxes) {
                const float* b2 = det_boxes + d * 4;
                float aspect = (b2[2] - b2[0]) / (b2[3] - b2[1] + 1e-6f);
                if (aspect > 0.8f) aspect_penalty = fmaxf(0.0f, 1.0f - (aspect - 0.8f));
                else if (aspect < 0.15f) aspect_penalty = fmaxf(0.0f, 1.0f - (0.15f - aspect) * 5.0f);
            }
            // Additive penalty to cost instead of multiplying probability
            // If score is low or aspect is bad, cost increases by up to +0.2
            float cost_penalty = (1.0f - score) * 0.1f + (1.0f - aspect_penalty) * 0.1f;
            float p = expf(-lambda * (cost + cost_penalty));""",

    "v5_hard_gating": """            // ADR 017: Quality-Aware Sinkhorn Prior
            if (det_boxes) {
                const float* b2 = det_boxes + d * 4;
                float aspect = (b2[2] - b2[0]) / (b2[3] - b2[1] + 1e-6f);
                if (aspect > 1.0f || aspect < 0.10f) continue; // Hard reject
            }
            float p = expf(-lambda * cost);"""
}

def main():
    base_content = get_base_content()
    results = []
    
    print(f"Starting A7 Strategy Sweep (Total Variants: {len(VARIANTS)})")
    
    try:
        for name, patch_code in VARIANTS.items():
            print(f"\n[{time.strftime('%H:%M:%S')}] Testing Variant: {name}")
            patch_and_rebuild(base_content, patch_code)
            metrics = run_eval()
            print(f"  Result: {metrics}")
            
            metrics['Variant'] = name
            results.append(metrics)
            
            with open(REPORT_FILE, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['Variant', 'IDF1', 'MOTA', 'IDs', 'FN', 'FP'])
                writer.writeheader()
                writer.writerows(results)
                
    finally:
        # Restore original state (which was v3_score_and_aspect essentially)
        print("\nRestoring original tracker_gpu.cu state...")
        with open(SRC_FILE, "w") as f:
            f.write(base_content)
        subprocess.run(REBUILD_SCRIPT, shell=True, stdout=subprocess.DEVNULL)
        
    print(f"\nSweep completed. Results saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()
