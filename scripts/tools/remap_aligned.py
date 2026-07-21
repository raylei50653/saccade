"""Remap/dedupe aligned track IDs (keep highest score per frame-id)."""

# status: diagnostic
from pathlib import Path
from collections import defaultdict


def remap_and_deduplicate(input_path: Path, output_path: Path, mapping: dict):
    grouped = defaultdict(list)
    remapped_count = 0
    with open(input_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                fid = int(parts[0])
                tid = int(parts[1])
                if tid in mapping:
                    parts[1] = str(mapping[tid])
                    tid = mapping[tid]
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

    with open(output_path, "w") as f:
        f.writelines(final_lines)
    print(
        f"[{input_path.name} -> {output_path.name}] Remapped {remapped_count} boxes, deduplicated {dedup_count} double boxes."
    )


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    demo_dir = project_root / "results" / "demo"

    clean_local = demo_dir / "custom_seq_clean_local.txt"
    clean_global_out = demo_dir / "custom_seq_clean.txt"

    occ_local = demo_dir / "custom_seq_occ_local.txt"
    occ_global_out = demo_dir / "custom_seq_occ.txt"
    occ_diff_out = demo_dir / "custom_seq_occ_for_diffusion.txt"

    # Clean local to global mapping (from _global_id_map.txt)
    clean_map = {
        2: 1,
        3: 2,
        5: 3,
        6: 4,
        10: 5,
        11: 6,
        13: 7,
        16: 8,
        21: 9,
        22: 10,
        27: 11,
        29: 12,
        32: 13,
    }

    # Occ local to global mapping (aligned with clean)
    occ_map = {
        1: 4,  # Person C (matches clean local 6 -> global 4)
        3: 2,  # Person A (matches clean local 3 -> global 2)
        4: 3,  # Person B (matches clean local 5 -> global 3)
        5: 4,  # Person C (frame 43-43, matches clean local 6 -> global 4)
        8: 2,  # Person A (after occlusion, matches clean local 3 -> global 2)
        9: 4,  # Person C (after occlusion, matches clean local 6 -> global 4)
        10: 5,  # Person D (matches clean local 10 -> global 5)
        11: 6,  # Person E (matches clean local 11 -> global 6)
        15: 5,  # Person D (after occlusion, matches clean local 10 -> global 5)
        17: 7,  # Person F (matches clean local 13 -> global 7)
        18: 7,  # Person F (another segment, matches clean local 13 -> global 7)
        19: 6,  # Person E (after occlusion, matches clean local 11 -> global 6)
        25: 6,  # Person E (short segment, matches clean local 11 -> global 6)
        27: 11,  # Person G (matches clean local 27 -> global 11)
        31: 12,  # Person H (matches clean local 29 -> global 12)
        34: 11,  # Person G (short segment, matches clean local 27 -> global 11)
        35: 11,  # Person G (short segment, matches clean local 27 -> global 11)
        39: 12,  # Person H (short segment, matches clean local 29 -> global 12)
    }

    # Occ mapping for diffusion video rendering: keep lost/candidate separate
    # Lost (source) tracks map to: 2, 4, 5, 6
    # Candidate tracks map to: 102, 104, 105, 106
    occ_diff_map = {
        1: 4,  # Person C source
        3: 2,  # Person A source
        4: 3,
        5: 4,  # Person C short segment before occlusion
        8: 102,  # Person A candidate
        9: 104,  # Person C candidate
        10: 5,  # Person D source
        11: 6,  # Person E source
        15: 105,  # Person D candidate
        17: 7,
        18: 7,
        19: 106,  # Person E candidate
        25: 106,  # Person E candidate short segment
        27: 11,
        31: 12,
        34: 11,
        35: 11,
        39: 12,
    }

    # Clean mapping for diffusion video rendering
    clean_diff_map = {
        2: 1,
        3: 2,
        5: 3,
        6: 4,
        10: 5,
        11: 6,  # Person E source
        13: 7,
        16: 8,
        21: 106,  # Person E candidate
        22: 10,
        27: 11,
        29: 12,  # Person H source
        32: 112,  # Person H candidate
    }

    clean_diff_out = demo_dir / "custom_seq_clean_for_diffusion.txt"

    # Process files
    remap_and_deduplicate(clean_local, clean_global_out, clean_map)
    remap_and_deduplicate(clean_local, clean_diff_out, clean_diff_map)
    remap_and_deduplicate(occ_local, occ_global_out, occ_map)
    remap_and_deduplicate(occ_local, occ_diff_out, occ_diff_map)


if __name__ == "__main__":
    main()
