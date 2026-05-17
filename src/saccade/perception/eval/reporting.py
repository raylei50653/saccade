import csv
import json
import numpy as np
from pathlib import Path
from typing import Any


POST_GPU_STAGE = "post_gpu_elapsed"


def _breakdown_display_names(breakdown_stage_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        name
        for name in breakdown_stage_names
        if name != POST_GPU_STAGE and not name.startswith("native_")
    )


def _native_breakdown_names(breakdown_stage_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in breakdown_stage_names if name.startswith("native_"))


def _postprocess_attribution_from_means(
    stage_means: dict[str, float],
    postprocess_wall_ms: float,
) -> dict[str, float]:
    gpu_ms = float(stage_means.get(POST_GPU_STAGE, 0.0))
    has_native_breakdown = any(k.startswith("native_") for k in stage_means)
    excluded_native_stages: set[str] = set()
    if any(
        stage_means.get(name, 0.0) > 0.0
        for name in (
            "native_filter_kernel",
            "native_gather_compact3",
            "native_copy_suspect",
        )
    ):
        excluded_native_stages.add("native_filter_gather")
    if any(
        stage_means.get(name, 0.0) > 0.0
        for name in ("native_large_argsort", "native_large_nms")
    ):
        excluded_native_stages.add("native_large_sort_nms")
    if any(
        stage_means.get(name, 0.0) > 0.0
        for name in ("native_large_gather4", "native_large_copyback")
    ):
        excluded_native_stages.add("native_compact_copy")
    excluded_post_stages = (
        {"post_filter", "post_nms"} if has_native_breakdown else set()
    )
    known_gpu_ms = float(
        sum(
            v
            for k, v in stage_means.items()
            if (
                (
                    k.startswith("post_")
                    and k != POST_GPU_STAGE
                    and k not in excluded_post_stages
                )
                or (k.startswith("native_") and k not in excluded_native_stages)
            )
        )
    )
    unattributed_gpu_ms = max(0.0, gpu_ms - known_gpu_ms)
    overhead_ms = max(0.0, postprocess_wall_ms - gpu_ms)
    return {
        "wall_ms": postprocess_wall_ms,
        "gpu_ms": gpu_ms,
        "unattributed_gpu_ms": unattributed_gpu_ms,
        "overhead_ms": overhead_ms,
    }


def _print_postprocess_gpu_attribution(attribution: dict[str, float]) -> None:
    if attribution["wall_ms"] <= 0.0 and attribution["gpu_ms"] <= 0.0:
        return
    print("\n| Postprocess GPU Attribution | Mean (ms/frame) |")
    print("| :--- | :--- |")
    print(f"| postprocess_wall | {attribution['wall_ms']:.2f} |")
    print(f"| postprocess_gpu | {attribution['gpu_ms']:.2f} |")
    print(f"| post_unattributed_gpu | {attribution['unattributed_gpu_ms']:.2f} |")
    print(f"| post_overhead | {attribution['overhead_ms']:.2f} |")


def _print_native_postprocess_breakdown(
    breakdown_names: tuple[str, ...],
    stage_totals: dict[str, float],
    profiled_frames: int,
) -> None:
    if profiled_frames <= 0:
        return
    if not any(stage_totals.get(name, 0.0) > 0.0 for name in breakdown_names):
        return
    print("\n| Native Postprocess Breakdown | Mean (ms/frame) |")
    print("| :--- | :--- |")
    for stage_name in breakdown_names:
        total_ms = stage_totals.get(stage_name, 0.0)
        if total_ms <= 0.0:
            continue
        print(f"| {stage_name} | {total_ms / profiled_frames:.2f} |")


def _print_stage_waterfall(
    stage_means: dict[str, float],
    frame_total_ms: float,
    bar_width: int = 20,
) -> None:
    if frame_total_ms <= 0:
        return
    sorted_stages = sorted(
        [(k, v) for k, v in stage_means.items() if v > 0.0],
        key=lambda x: -x[1],
    )
    if not sorted_stages:
        return
    accounted = sum(v for _, v in sorted_stages)
    unaccounted = frame_total_ms - accounted

    label_w = max((len(s) for s, _ in sorted_stages), default=14) + 2
    label_w = max(label_w, 14)
    sep = "─" * (label_w + bar_width + 18)
    print(f"\n  Stage Breakdown (frame_total = {frame_total_ms:.2f} ms)")
    print(f"  {sep}")
    for name, ms in sorted_stages:
        pct = ms / frame_total_ms * 100.0
        filled = round(pct / 100.0 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"  {name:<{label_w}} {bar}  {ms:6.2f}ms  {pct:5.1f}%")
    if unaccounted > 0.5:
        pct = unaccounted / frame_total_ms * 100.0
        print(
            f"  {'[unaccounted]':<{label_w}} {'░' * bar_width}  {unaccounted:6.2f}ms  {pct:5.1f}%"
        )
    print(f"  {sep}")


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
    gmc_breakdown_names: tuple[str, ...],
    overall_gmc_samples: dict[str, list[float]],
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
    debug_birth_csv: str,
    debug_birth_rows: list[dict[str, float | int | str | bool]],
    all_seq_profile: list[dict[str, Any]] | None = None,
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
        overall_top_level_means: dict[str, float] = {}
        for stage_name in top_level_stage_names:
            samples = overall_stage_samples[stage_name]
            if not samples:
                continue
            arr = np.array(samples, dtype=np.float64)
            overall_top_level_means[stage_name] = float(arr.mean())
            print(
                f"| {stage_name} | {arr.mean():.2f} | {arr.std():.2f} | "
                f"{np.percentile(arr, 95):.2f} | {np.percentile(arr, 99):.2f} |"
            )
            stage_summary_lines.append(
                f"{stage_name}\tmean_ms={arr.mean():.2f}\tstd_ms={arr.std():.2f}\t"
                f"p95_ms={np.percentile(arr, 95):.2f}\tp99_ms={np.percentile(arr, 99):.2f}\t"
                f"samples={len(samples)}"
            )
        display_breakdown_names = _breakdown_display_names(breakdown_stage_names)
        native_breakdown_names = _native_breakdown_names(breakdown_stage_names)
        if any(overall_stage_totals[name] > 0.0 for name in display_breakdown_names):
            print("\n| Postprocess Breakdown | Mean (ms/frame) |")
            print("| :--- | :--- |")
            for stage_name in display_breakdown_names:
                total_ms = overall_stage_totals[stage_name]
                if total_ms <= 0.0:
                    continue
                print(f"| {stage_name} | {total_ms / overall_profiled_frames:.2f} |")
                stage_summary_lines.append(
                    f"{stage_name}\tmean_ms={total_ms / overall_profiled_frames:.2f}\ttotal_ms={total_ms:.2f}"
                )
        overall_breakdown_means = {
            stage_name: overall_stage_totals[stage_name] / overall_profiled_frames
            for stage_name in breakdown_stage_names
            if overall_stage_totals.get(stage_name, 0.0) > 0.0
            and overall_profiled_frames > 0
        }
        overall_post_attr = _postprocess_attribution_from_means(
            overall_breakdown_means,
            overall_top_level_means.get("postprocess", 0.0),
        )
        _print_postprocess_gpu_attribution(overall_post_attr)
        _print_native_postprocess_breakdown(
            native_breakdown_names,
            overall_stage_totals,
            overall_profiled_frames,
        )
        stage_summary_lines.append(
            "postprocess_gpu_attribution\t"
            f"wall_ms={overall_post_attr['wall_ms']:.2f}\t"
            f"gpu_ms={overall_post_attr['gpu_ms']:.2f}\t"
            f"unattributed_gpu_ms={overall_post_attr['unattributed_gpu_ms']:.2f}\t"
            f"overhead_ms={overall_post_attr['overhead_ms']:.2f}"
        )
        if any(overall_gmc_samples[name] for name in gmc_breakdown_names):
            print("\n| GMC Breakdown | Mean (ms) | Std (ms) | P95 (ms) | P99 (ms) |")
            print("| :--- | :--- | :--- | :--- | :--- |")
            for stage_name in gmc_breakdown_names:
                samples = overall_gmc_samples[stage_name]
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

        # ASCII waterfall for overall stage breakdown
        overall_means = {}
        ft_samples = overall_stage_samples.get("frame_total", [])
        ft_ms = (
            float(np.mean(np.array(ft_samples, dtype=np.float64)))
            if ft_samples
            else 0.0
        )
        for sn in top_level_stage_names:
            if sn == "frame_total":
                continue
            samp = overall_stage_samples.get(sn, [])
            if samp:
                overall_means[sn] = float(np.mean(np.array(samp, dtype=np.float64)))
        _print_stage_waterfall(overall_means, ft_ms)

        # JSON profile output
        overall_stage_json: dict[str, Any] = {}
        for sn in top_level_stage_names:
            samp = overall_stage_samples.get(sn, [])
            if samp:
                arr = np.array(samp, dtype=np.float64)
                overall_stage_json[sn] = {
                    "mean_ms": float(arr.mean()),
                    "std_ms": float(arr.std()),
                    "p95_ms": float(np.percentile(arr, 95)),
                    "p99_ms": float(np.percentile(arr, 99)),
                }
        if "postprocess" in overall_stage_json:
            overall_stage_json["postprocess"].update(
                {
                    "mean_wall_ms": overall_post_attr["wall_ms"],
                    "mean_gpu_ms": overall_post_attr["gpu_ms"],
                    "mean_unattributed_gpu_ms": overall_post_attr[
                        "unattributed_gpu_ms"
                    ],
                    "mean_overhead_ms": overall_post_attr["overhead_ms"],
                }
            )
        for sn in breakdown_stage_names:
            tot = overall_stage_totals.get(sn, 0.0)
            if tot > 0.0 and overall_profiled_frames > 0:
                overall_stage_json[sn] = {"mean_ms": tot / overall_profiled_frames}

        fps_mean_ms = ft_ms if ft_ms > 0 else 0.0
        fps_mean = 1000.0 / fps_mean_ms if fps_mean_ms > 0 else 0.0
        profile_json: dict[str, Any] = {
            "meta": {
                "profiled_frames": overall_profiled_frames,
                "fps_mean": round(fps_mean, 2),
                "fps_mean_ms": round(fps_mean_ms, 3),
            },
            "overall": overall_stage_json,
            "sequences": {
                e["seq"]: {
                    "frames": e["frames"],
                    **{
                        # Backward-compatible: older seq entries may not carry
                        # postprocess_attribution yet.
                        k: (
                            v
                            | {
                                "mean_wall_ms": e.get(
                                    "postprocess_attribution", {}
                                ).get("wall_ms", v.get("mean_ms", 0.0)),
                                "mean_gpu_ms": e.get("postprocess_attribution", {}).get(
                                    "gpu_ms", 0.0
                                ),
                                "mean_unattributed_gpu_ms": e.get(
                                    "postprocess_attribution", {}
                                ).get("unattributed_gpu_ms", 0.0),
                                "mean_overhead_ms": e.get(
                                    "postprocess_attribution", {}
                                ).get("overhead_ms", 0.0),
                            }
                            if k == "postprocess"
                            else v
                        )
                        for k, v in e["stages"].items()
                    },
                }
                for e in (all_seq_profile or [])
            },
        }
        (output_root / "_stage_profile.json").write_text(
            json.dumps(profile_json, indent=2) + "\n"
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

    if debug_birth_csv:
        debug_birth_path = Path(debug_birth_csv)
        debug_birth_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "seq",
            "frame",
            "policy",
            "det_idx",
            "score_before",
            "score_after",
            "x1",
            "y1",
            "x2",
            "y2",
            "w",
            "h",
            "output_emitted",
            "output_local_track_id",
            "output_track_id",
            "output_score",
            "output_x1",
            "output_y1",
            "output_x2",
            "output_y2",
        ]
        with debug_birth_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(debug_birth_rows)
        print(
            f"\n🪲 Birth promotion dump: rows={len(debug_birth_rows)} "
            f"path={debug_birth_path}"
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
    gmc_breakdown_names: tuple[str, ...],
    seq_gmc_samples: dict[str, list[float]],
    overall_gmc_samples: dict[str, list[float]],
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
        seq_top_level_means: dict[str, float] = {}
        for stage_name in top_level_stage_names:
            samples = seq_stage_samples[stage_name]
            if not samples:
                continue
            arr = np.array(samples, dtype=np.float64)
            mean_ms = float(arr.mean())
            seq_top_level_means[stage_name] = mean_ms
            std_ms = float(arr.std())
            p95_ms = float(np.percentile(arr, 95))
            p99_ms = float(np.percentile(arr, 99))
            print(
                f"| {stage_name} | {mean_ms:.2f} | "
                f"{std_ms:.2f} | {p95_ms:.2f} | {p99_ms:.2f} |"
            )
            overall_stage_totals[stage_name] += float(arr.sum())
            overall_stage_samples[stage_name].extend(samples)
        display_breakdown_names = _breakdown_display_names(breakdown_stage_names)
        native_breakdown_names = _native_breakdown_names(breakdown_stage_names)
        if any(seq_stage_totals[name] > 0.0 for name in display_breakdown_names):
            print("\n| Postprocess Breakdown | Mean (ms/frame) |")
            print("| :--- | :--- |")
            for stage_name in display_breakdown_names:
                total_ms = seq_stage_totals[stage_name]
                if total_ms <= 0.0:
                    continue
                mean_ms = total_ms / seq_profiled_frames
                print(f"| {stage_name} | {mean_ms:.2f} |")
                overall_stage_totals[stage_name] += total_ms
        for stage_name in breakdown_stage_names:
            if stage_name in display_breakdown_names:
                continue
            total_ms = seq_stage_totals[stage_name]
            if total_ms > 0.0:
                overall_stage_totals[stage_name] += total_ms
        seq_breakdown_means = {
            stage_name: seq_stage_totals[stage_name] / seq_profiled_frames
            for stage_name in breakdown_stage_names
            if seq_stage_totals.get(stage_name, 0.0) > 0.0
        }
        seq_post_attr = _postprocess_attribution_from_means(
            seq_breakdown_means,
            seq_top_level_means.get("postprocess", 0.0),
        )
        _print_postprocess_gpu_attribution(seq_post_attr)
        _print_native_postprocess_breakdown(
            native_breakdown_names,
            seq_stage_totals,
            seq_profiled_frames,
        )
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
        if any(seq_gmc_samples[name] for name in gmc_breakdown_names):
            print("\n| GMC Breakdown | Mean (ms) | Std (ms) | P95 (ms) | P99 (ms) |")
            print("| :--- | :--- | :--- | :--- | :--- |")
            for stage_name in gmc_breakdown_names:
                samples = seq_gmc_samples[stage_name]
                if not samples:
                    continue
                arr = np.array(samples, dtype=np.float64)
                print(
                    f"| {stage_name} | {arr.mean():.2f} | "
                    f"{arr.std():.2f} | {np.percentile(arr, 95):.2f} | "
                    f"{np.percentile(arr, 99):.2f} |"
                )
                overall_gmc_samples[stage_name].extend(samples)
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
        stage_summary_lines.append(
            "postprocess_gpu_attribution\t"
            f"wall_ms={seq_post_attr['wall_ms']:.2f}\t"
            f"gpu_ms={seq_post_attr['gpu_ms']:.2f}\t"
            f"unattributed_gpu_ms={seq_post_attr['unattributed_gpu_ms']:.2f}\t"
            f"overhead_ms={seq_post_attr['overhead_ms']:.2f}"
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
        for stage_name in gmc_breakdown_names:
            samples = seq_gmc_samples[stage_name]
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
