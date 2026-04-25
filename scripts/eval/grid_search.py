import os
import subprocess
import itertools
import re

def main():
    conf_threshs = [0.1, 0.25, 0.4]
    track_threshs = [0.1, 0.25]
    high_threshs = [0.4, 0.5, 0.6]
    match_threshs = [0.8, 0.9]

    best_mota = -100.0
    best_params = None

    seqs = "MOT17-04-FRCNN"

    for ct, tt, ht, mt in itertools.product(conf_threshs, track_threshs, high_threshs, match_threshs):
        print(f"\n--- Testing conf={ct}, track={tt}, high={ht}, match={mt} ---", flush=True)
        cmd = [
            "uv", "run", "python", "scripts/eval/mot17.py",
            "--sequences", seqs,
            "--conf-threshold", str(ct),
            "--track-thresh", str(tt),
            "--high-thresh", str(ht),
            "--match-thresh", str(mt),
            "--output", "results/grid_search_eval",
            "--no-reid"
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        calc_cmd = [
            "uv", "run", "python", "scripts/eval/calculate_mota.py",
            "--results", "results/grid_search_eval"
        ]
        
        res = subprocess.run(calc_cmd, capture_output=True, text=True)
        output = res.stdout
        
        # Find OVERALL MOTA
        mota_match = re.search(r'OVERALL\s+.*?([0-9\.\-]+)%', output)
        if mota_match:
            mota = float(mota_match.group(1))
            print(f"MOTA: {mota}%")
            if mota > best_mota:
                best_mota = mota
                best_params = (ct, tt, ht, mt)
                print(f"🌟 New best MOTA: {best_mota}% with params {best_params}")
        else:
            print("Could not find MOTA in output.")

    print(f"\n🏆 BEST PARAMS: conf={best_params[0]}, track={best_params[1]}, high={best_params[2]}, match={best_params[3]} -> MOTA {best_mota}%")

if __name__ == "__main__":
    main()
