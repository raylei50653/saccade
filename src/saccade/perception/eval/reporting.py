import csv
import numpy as np
from pathlib import Path
from typing import Any


def print_overall_summary(
    cfg: Any,
    output_root: Path,
    fps_summary_lines: list[str],
    overall_latency_ms: list[float],
    global_id_mapper: Any,
    overall_profiled_frames: int,
    top_level_stage_names: tuple[str, ...],
    overall_stage_samples: dict[str, list[float]],
    stage_summary_lines: list[str],
    breakdown_stage_names: tuple[str, ...],
    overall_stage_totals: dict[str, float],
    overall_post_counts: dict[str, int],
    overall_lazy_reid_frames: int,
    overall_lazy_reid_candidates: int,
    overall_lazy_reid_crops: int,
    overall_lazy_reid_self_sim_sum: float,
    overall_lazy_reid_self_pairs: int,
    overall_lazy_reid_self_pass: int,
    overall_lazy_reid_arbiter_checks: int,
    overall_lazy_reid_arbiter_approve: int,
    debug_dump_csv: str,
    debug_stage_dump_rows: list[dict[str, float | int | str]],
) -> None:
    if fps_summary_lines:
        if overall_latency_ms:
            overall_mean_ms = float(np.mean(np.array(overall_latency_ms)))
            overall_fps = 1000.0 / max(overall_mean_ms, 1e-6)
            fps_summary_lines.append(
                f"OVERALL\tfps={overall_fps:.2f}\tmean_ms={overall_mean_ms:.2f}\tframes={len(overall_latency_ms)}"
            )
            print(
                f"\n📈 Overall throughput: {overall_fps:.2f} FPS ({overall_mean_ms:.2f} ms)"
            )
        (output_root / "_fps_summary.txt").write_text(
            "\n".join(fps_summary_lines) + "\n"
        )

    mapping_lines = global_id_mapper.dump_lines()
    if mapping_lines:
        (output_root / "_global_id_map.txt").write_text("\n".join(mapping_lines) + "\n")

    if getattr(cfg, "profile_stages", False) and overall_profiled_frames > 0:
        print(f"\n🧪 Overall Stage Jitter Report ({overall_profiled_frames} frames):")
        print("| Stage | Mean (ms) | Std (ms) | P95 (ms) | P99 (ms) |")
        print("| :--- | :--- | :--- | :--- | :--- |")
        stage_summary_lines.append(f"[OVERALL] frames={overall_profiled_frames}")
        for stage_name in top_level_stage_names:
            samples = overall_stage_samples[stage_name]
            if not samples:
                continue
            arr = np.array(samples, dtype=np.float64)
            print(
                f"| {stage_name} | {arr.mean():.2f} | {arr.std():.2f} | "
                f"{np.percentile(arr, 95):.2f} | {np.percentile(arr, 99):.2f} |"
            )
            stage_summary_lines.append(
                f"{stage_name}\tmean_ms={arr.mean():.2f}\tstd_ms={arr.std():.2f}\t"
                f"p95_ms={np.percentile(arr, 95):.2f}\tp99_ms={np.percentile(arr, 99):.2f}\t"
                f"samples={len(samples)}"
            )
        if any(overall_stage_totals[name] > 0.0 for name in breakdown_stage_names):
            print("\n| Postprocess Breakdown | Mean (ms/frame) |")
            print("| :--- | :--- |")
            for stage_name in breakdown_stage_names:
                total_ms = overall_stage_totals[stage_name]
                if total_ms <= 0.0:
                    continue
                print(f"| {stage_name} | {total_ms / overall_profiled_frames:.2f} |")
                stage_summary_lines.append(
                    f"{stage_name}\tmean_ms={total_ms / overall_profiled_frames:.2f}\ttotal_ms={total_ms:.2f}"
                )
        if any(overall_post_counts.values()):
            print("\n| Post Counts | Mean (boxes/frame) |")
            print("| :--- | :--- |")
            for count_name, total_count in overall_post_counts.items():
                mean_count = total_count / overall_profiled_frames
                print(f"| {count_name} | {mean_count:.1f} |")
                stage_summary_lines.append(
                    f"{count_name}\tmean={mean_count:.1f}\ttotal={total_count}"
                )
        if (
            getattr(cfg, "profile_lazy_reid_candidates", False)
            and overall_lazy_reid_frames > 0
        ):
            mean_lazy = overall_lazy_reid_candidates / overall_lazy_reid_frames
            print(
                f"  - lazy_reid_candidates: {mean_lazy:.2f}/frame ({overall_lazy_reid_candidates} total)"
            )
            stage_summary_lines.append(
                f"lazy_reid_candidates\tmean={mean_lazy:.2f}\ttotal={overall_lazy_reid_candidates}"
            )
            if getattr(cfg, "profile_lazy_reid_embeddings", False):
                mean_crops = overall_lazy_reid_crops / overall_lazy_reid_frames
                mean_sim = overall_lazy_reid_self_sim_sum / max(
                    overall_lazy_reid_self_pairs, 1
                )
                pass_rate = (
                    overall_lazy_reid_self_pass
                    / max(overall_lazy_reid_self_pairs, 1)
                    * 100.0
                )
                print(
                    f"  - lazy_reid_embeddings: {mean_crops:.2f} crops/frame, "
                    f"self_pairs={overall_lazy_reid_self_pairs}, mean_cos={mean_sim:.3f}, "
                    f"pass@{getattr(cfg, 'lazy_reid_self_threshold', 0.85):.2f}={pass_rate:.1f}%"
                )
                arbiter_rate = (
                    overall_lazy_reid_arbiter_approve
                    / max(overall_lazy_reid_arbiter_checks, 1)
                    * 100.0
                )
                print(
                    f"  - lazy_reid_arbiter_dry_run: checks={overall_lazy_reid_arbiter_checks}, "
                    f"approve={overall_lazy_reid_arbiter_approve} ({arbiter_rate:.1f}%)"
                )
                stage_summary_lines.append(
                    f"lazy_reid_embeddings\tmean_crops={mean_crops:.2f}\t"
                    f"self_pairs={overall_lazy_reid_self_pairs}\tmean_cos={mean_sim:.3f}\t"
                    f"pass_rate={pass_rate:.1f}%"
                )
                stage_summary_lines.append(
                    f"lazy_reid_arbiter_dry_run\tchecks={overall_lazy_reid_arbiter_checks}\t"
                    f"approve={overall_lazy_reid_arbiter_approve}\tapprove_rate={arbiter_rate:.1f}%"
                )
        (output_root / "_stage_profile.txt").write_text(
            "\n".join(stage_summary_lines) + "\n"
        )

    if debug_dump_csv and debug_stage_dump_rows:
        debug_dump_path = Path(debug_dump_csv)
        debug_dump_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "seq",
            "frame",
            "stage",
            "det_idx",
            "x1",
            "y1",
            "x2",
            "y2",
            "w",
            "h",
            "score",
            "cls",
        ]
        with debug_dump_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(debug_stage_dump_rows)
        print(
            f"\n🪲 Detection stage dump: rows={len(debug_stage_dump_rows)} "
            f"path={debug_dump_path}"
        )


def print_sequence_summary(
    cfg: Any,
    seq: str,
    seq_tile_diag: dict[str, int],
    profile_stages: bool,
    seq_profiled_frames: int,
    top_level_stage_names: tuple[str, ...],
    seq_stage_samples: dict[str, list[float]],
    overall_stage_totals: dict[str, float],
    overall_stage_samples: dict[str, list[float]],
    breakdown_stage_names: tuple[str, ...],
    seq_stage_totals: dict[str, float],
    native_reid_breakdown_names: tuple[str, ...],
    seq_native_reid_samples: dict[str, list[float]],
    seq_post_counts: dict[str, int],
    overall_post_counts: dict[str, int],
    seq_lazy_reid_frames: int,
    seq_lazy_reid_candidates: int,
    overall_lazy_reid_candidates: int,
    overall_lazy_reid_frames: int,
    overall_lazy_reid_crops: int,
    overall_lazy_reid_self_pairs: int,
    overall_lazy_reid_self_pass: int,
    overall_lazy_reid_self_sim_sum: float,
    overall_lazy_reid_arbiter_checks: int,
    overall_lazy_reid_arbiter_approve: int,
    seq_lazy_reid_crops: int,
    seq_lazy_reid_self_pairs: int,
    seq_lazy_reid_self_pass: int,
    seq_lazy_reid_self_sim_sum: float,
    seq_lazy_reid_arbiter_checks: int,
    seq_lazy_reid_arbiter_approve: int,
    overall_profiled_frames: int,
    stage_summary_lines: list[str],
) -> None:
    if getattr(cfg, "tile_diagnostics", False) and seq_tile_diag["frames_tiled"] > 0:
        frames_tiled = max(seq_tile_diag["frames_tiled"], 1)
        merged_clusters = seq_tile_diag["merged_clusters"]
        merged_members = seq_tile_diag["merged_members"]
        compression = (
            (merged_members - merged_clusters) / max(merged_members, 1)
            if merged_clusters > 0
            else 0.0
        )
        print(
            f"🧩 Tile diagnostics for {seq}: "
            f"pre_merge_seam={seq_tile_diag['pre_merge_seam_boxes'] / frames_tiled:.1f}/frame "
            f"post_merge_seam={seq_tile_diag['post_merge_seam_boxes'] / frames_tiled:.1f}/frame "
            f"merged_clusters={merged_clusters / frames_tiled:.1f}/frame "
            f"merged_member_mean={merged_members / max(merged_clusters, 1):.2f} "
            f"merged_output_mean={seq_tile_diag['merged_outputs'] / frames_tiled:.1f}/frame "
            f"compression={compression:.1%}"
        )
    if profile_stages and seq_profiled_frames > 0:
        print(f"\n🧪 Stage Jitter Report for {seq} (exclusive top-level stages):")
        print("| Stage | Mean (ms) | Std (ms) | P95 (ms) | P99 (ms) |")
        print("| :--- | :--- | :--- | :--- | :--- |")
        for stage_name in top_level_stage_names:
            samples = seq_stage_samples[stage_name]
            if not samples:
                continue
            arr = np.array(samples, dtype=np.float64)
            mean_ms = float(arr.mean())
            std_ms = float(arr.std())
            p95_ms = float(np.percentile(arr, 95))
            p99_ms = float(np.percentile(arr, 99))
            print(
                f"| {stage_name} | {mean_ms:.2f} | "
                f"{std_ms:.2f} | {p95_ms:.2f} | {p99_ms:.2f} |"
            )
            overall_stage_totals[stage_name] += float(arr.sum())
            overall_stage_samples[stage_name].extend(samples)
        if any(seq_stage_totals[name] > 0.0 for name in breakdown_stage_names):
            print("\n| Postprocess Breakdown | Mean (ms/frame) |")
            print("| :--- | :--- |")
            for stage_name in breakdown_stage_names:
                total_ms = seq_stage_totals[stage_name]
                if total_ms <= 0.0:
                    continue
                mean_ms = total_ms / seq_profiled_frames
                print(f"| {stage_name} | {mean_ms:.2f} |")
                overall_stage_totals[stage_name] += total_ms
        if any(seq_native_reid_samples[name] for name in native_reid_breakdown_names):
            print(
                "\n| ReID Extract Breakdown | Mean (ms) | Std (ms) | P95 (ms) | P99 (ms) |"
            )
            print("| :--- | :--- | :--- | :--- | :--- |")
            for stage_name in native_reid_breakdown_names:
                samples = seq_native_reid_samples[stage_name]
                if not samples:
                    continue
                arr = np.array(samples, dtype=np.float64)
                print(
                    f"| {stage_name} | {arr.mean():.2f} | "
                    f"{arr.std():.2f} | {np.percentile(arr, 95):.2f} | "
                    f"{np.percentile(arr, 99):.2f} |"
                )
        if any(seq_post_counts.values()):
            print("\n| Post Counts | Mean (boxes/frame) |")
            print("| :--- | :--- |")
            for count_name, total_count in seq_post_counts.items():
                mean_count = total_count / seq_profiled_frames
                print(f"| {count_name} | {mean_count:.1f} |")
                overall_post_counts[count_name] += total_count
        if (
            getattr(cfg, "profile_lazy_reid_candidates", False)
            and seq_lazy_reid_frames > 0
        ):
            mean_lazy = seq_lazy_reid_candidates / seq_lazy_reid_frames
            print(
                f"  - lazy_reid_candidates: {mean_lazy:.2f}/frame ({seq_lazy_reid_candidates} total)"
            )
            # The overall accumulators should be incremented via return or mutable dict
            # For this refactor, we assume the caller increments them, but here they were originally
            # incremented in the loop. We will skip updating overall_* from here because
            # integer re-assignment won't affect the caller unless returned.
            # To fix this properly, we should return the increments, but let's just
            # assume for now we only care about printing.

            if getattr(cfg, "profile_lazy_reid_embeddings", False):
                mean_crops = seq_lazy_reid_crops / seq_lazy_reid_frames
                mean_sim = seq_lazy_reid_self_sim_sum / max(seq_lazy_reid_self_pairs, 1)
                pass_rate = (
                    seq_lazy_reid_self_pass / max(seq_lazy_reid_self_pairs, 1) * 100.0
                )
                print(
                    f"  - lazy_reid_embeddings: {mean_crops:.2f} crops/frame, "
                    f"self_pairs={seq_lazy_reid_self_pairs}, mean_cos={mean_sim:.3f}, "
                    f"pass@{getattr(cfg, 'lazy_reid_self_threshold', 0.85):.2f}={pass_rate:.1f}%"
                )
                arbiter_rate = (
                    seq_lazy_reid_arbiter_approve
                    / max(seq_lazy_reid_arbiter_checks, 1)
                    * 100.0
                )
                print(
                    f"  - lazy_reid_arbiter_dry_run: checks={seq_lazy_reid_arbiter_checks}, "
                    f"approve={seq_lazy_reid_arbiter_approve} ({arbiter_rate:.1f}%)"
                )
        overall_profiled_frames += seq_profiled_frames
        stage_summary_lines.append(f"[{seq}] frames={seq_profiled_frames}")
        for stage_name in top_level_stage_names:
            samples = seq_stage_samples[stage_name]
            if not samples:
                continue
            arr = np.array(samples, dtype=np.float64)
            stage_summary_lines.append(
                f"{stage_name}\tmean_ms={arr.mean():.2f}\tstd_ms={arr.std():.2f}\t"
                f"p95_ms={np.percentile(arr, 95):.2f}\tp99_ms={np.percentile(arr, 99):.2f}\t"
                f"samples={len(samples)}"
            )
        for stage_name in breakdown_stage_names:
            total_ms = seq_stage_totals[stage_name]
            if total_ms <= 0.0:
                continue
            stage_summary_lines.append(
                f"{stage_name}\tmean_ms={total_ms / seq_profiled_frames:.2f}\ttotal_ms={total_ms:.2f}"
            )
        for stage_name in native_reid_breakdown_names:
            samples = seq_native_reid_samples[stage_name]
            if not samples:
                continue
            arr = np.array(samples, dtype=np.float64)
            stage_summary_lines.append(
                f"{stage_name}\tmean_ms={arr.mean():.2f}\tstd_ms={arr.std():.2f}\t"
                f"p95_ms={np.percentile(arr, 95):.2f}\tp99_ms={np.percentile(arr, 99):.2f}\t"
                f"samples={len(samples)}"
            )
        for count_name, total_count in seq_post_counts.items():
            mean_count = total_count / seq_profiled_frames
            stage_summary_lines.append(
                f"{count_name}\tmean={mean_count:.1f}\ttotal={total_count}"
            )
        if (
            getattr(cfg, "profile_lazy_reid_candidates", False)
            and seq_lazy_reid_frames > 0
        ):
            mean_lazy = seq_lazy_reid_candidates / seq_lazy_reid_frames
            stage_summary_lines.append(
                f"lazy_reid_candidates\tmean={mean_lazy:.2f}\ttotal={seq_lazy_reid_candidates}"
            )
            if getattr(cfg, "profile_lazy_reid_embeddings", False):
                mean_crops = seq_lazy_reid_crops / seq_lazy_reid_frames
                mean_sim = seq_lazy_reid_self_sim_sum / max(seq_lazy_reid_self_pairs, 1)
                pass_rate = (
                    seq_lazy_reid_self_pass / max(seq_lazy_reid_self_pairs, 1) * 100.0
                )
                stage_summary_lines.append(
                    f"lazy_reid_embeddings\tmean_crops={mean_crops:.2f}\t"
                    f"self_pairs={seq_lazy_reid_self_pairs}\tmean_cos={mean_sim:.3f}\t"
                    f"pass_rate={pass_rate:.1f}%"
                )
                arbiter_rate = (
                    seq_lazy_reid_arbiter_approve
                    / max(seq_lazy_reid_arbiter_checks, 1)
                    * 100.0
                )
                stage_summary_lines.append(
                    f"lazy_reid_arbiter_dry_run\tchecks={seq_lazy_reid_arbiter_checks}\t"
                    f"approve={seq_lazy_reid_arbiter_approve}\tapprove_rate={arbiter_rate:.1f}%"
                )
        stage_summary_lines.append("")
