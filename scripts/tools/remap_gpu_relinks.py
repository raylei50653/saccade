import numpy as np
from pathlib import Path
from collections import defaultdict


def remap_seq(name: str):
    project_root = Path(__file__).resolve().parent.parent.parent
    result_path = project_root / "results" / "demo" / f"{name}.txt"
    map_path = project_root / "results" / "demo" / "_global_id_map.txt"
    raw_path = project_root / "datasets" / "demo" / f"{name}_raw_data.npy"

    if not (result_path.exists() and map_path.exists() and raw_path.exists()):
        print(f"Skipping {name}: files not found")
        return

    # Load global ID mapping
    local_to_global = {}
    with open(map_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[0] == name:
                local_id = int(parts[1].split("=")[1])
                global_id = int(parts[2].split("=")[1])
                local_to_global[local_id] = global_id

    # Load raw GPU attempts and find accepted ones
    arr = np.load(raw_path)
    accepts = arr[arr[:, 5] == 0]

    relink_map = {}
    for row in accepts:
        lost_local = int(row[7])
        cand_local = int(row[8])
        lost_global = local_to_global.get(lost_local)
        cand_global = local_to_global.get(cand_local)
        if lost_global is not None and cand_global is not None:
            relink_map[cand_global] = lost_global
            print(
                f"[{name}] Relink Map: Local {lost_local}->{cand_local} | Global ID {cand_global} -> {lost_global}"
            )

    if not relink_map:
        print(f"[{name}] No relinks to remap")
        return

    # Read, remap, and group by (frame, tid)
    grouped = defaultdict(list)
    remapped_count = 0
    with open(result_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                fid = int(parts[0])
                tid = int(parts[1])
                if tid in relink_map:
                    parts[1] = str(relink_map[tid])
                    tid = relink_map[tid]
                    remapped_count += 1
                grouped[(fid, tid)].append(parts)

    # Deduplicate: keep only the highest score for each (fid, tid)
    final_lines = []
    dedup_count = 0
    for (fid, tid), boxes in sorted(grouped.items()):
        if len(boxes) > 1:
            # Sort by score descending (index 6)
            boxes.sort(key=lambda b: float(b[6]) if len(b) > 6 else -1.0, reverse=True)
            dedup_count += len(boxes) - 1
        final_lines.append(",".join(boxes[0]) + "\n")

    with open(result_path, "w") as f:
        f.writelines(final_lines)
    print(
        f"[{name}] Successfully remapped {remapped_count} boxes and deduplicated {dedup_count} double boxes."
    )


if __name__ == "__main__":
    remap_seq("custom_seq_clean")
    remap_seq("custom_seq_occ")
