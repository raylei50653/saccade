"""Format benchmark/eval stage tables for reports."""

# status: archive-candidate
from pathlib import Path

file_path = Path("src/saccade/perception/eval/runner.py")
content = file_path.read_text()

# 1. Per-sequence stage jitter report
old1 = """        if profile_stages and seq_profiled_frames > 0:
            print(f"\\n🧪 Stage Jitter Report for {seq} (exclusive top-level stages):")
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
                    f"  - {stage_name}: mean={mean_ms:.2f} ms "
                    f"std={std_ms:.2f} ms p95={p95_ms:.2f} ms p99={p99_ms:.2f} ms"
                )"""

new1 = """        if profile_stages and seq_profiled_frames > 0:
            print(f"\\n🧪 Stage Jitter Report for {seq} (exclusive top-level stages):")
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
                )"""

content = content.replace(old1, new1)

# 2. Per-sequence breakdowns
old2 = """            if any(seq_stage_totals[name] > 0.0 for name in breakdown_stage_names):
                print("  - postprocess_breakdown:")
                for stage_name in breakdown_stage_names:
                    total_ms = seq_stage_totals[stage_name]
                    if total_ms <= 0.0:
                        continue
                    mean_ms = total_ms / seq_profiled_frames
                    print(f"    - {stage_name}: {mean_ms:.2f} ms/frame")
                    overall_stage_totals[stage_name] += total_ms
            if any(
                seq_native_reid_samples[name] for name in native_reid_breakdown_names
            ):
                print("  - reid_extract_breakdown:")
                for stage_name in native_reid_breakdown_names:
                    samples = seq_native_reid_samples[stage_name]
                    if not samples:
                        continue
                    arr = np.array(samples, dtype=np.float64)
                    print(
                        f"    - {stage_name}: mean={arr.mean():.2f} ms "
                        f"std={arr.std():.2f} ms p95={np.percentile(arr, 95):.2f} ms "
                        f"p99={np.percentile(arr, 99):.2f} ms"
                    )
            if any(seq_post_counts.values()):
                print("  - post_counts:")
                for count_name, total_count in seq_post_counts.items():
                    mean_count = total_count / seq_profiled_frames
                    print(f"    - {count_name}: {mean_count:.1f} boxes/frame")
                    overall_post_counts[count_name] += total_count"""

new2 = """            if any(seq_stage_totals[name] > 0.0 for name in breakdown_stage_names):
                print("\\n| Postprocess Breakdown | Mean (ms/frame) |")
                print("| :--- | :--- |")
                for stage_name in breakdown_stage_names:
                    total_ms = seq_stage_totals[stage_name]
                    if total_ms <= 0.0:
                        continue
                    mean_ms = total_ms / seq_profiled_frames
                    print(f"| {stage_name} | {mean_ms:.2f} |")
                    overall_stage_totals[stage_name] += total_ms
            if any(
                seq_native_reid_samples[name] for name in native_reid_breakdown_names
            ):
                print("\\n| ReID Extract Breakdown | Mean (ms) | Std (ms) | P95 (ms) | P99 (ms) |")
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
                print("\\n| Post Counts | Mean (boxes/frame) |")
                print("| :--- | :--- |")
                for count_name, total_count in seq_post_counts.items():
                    mean_count = total_count / seq_profiled_frames
                    print(f"| {count_name} | {mean_count:.1f} |")
                    overall_post_counts[count_name] += total_count"""

content = content.replace(old2, new2)

# 3. Overall Report
old3 = """    if profile_stages and overall_profiled_frames > 0:
        print(f"\\n🧪 Overall Stage Jitter Report ({overall_profiled_frames} frames):")
        stage_summary_lines.append(f"[OVERALL] frames={overall_profiled_frames}")
        for stage_name in top_level_stage_names:
            samples = overall_stage_samples[stage_name]
            if not samples:
                continue
            arr = np.array(samples, dtype=np.float64)
            print(
                f"  - {stage_name}: mean={arr.mean():.2f} ms "
                f"std={arr.std():.2f} ms p95={np.percentile(arr, 95):.2f} ms "
                f"p99={np.percentile(arr, 99):.2f} ms"
            )"""

new3 = """    if profile_stages and overall_profiled_frames > 0:
        print(f"\\n🧪 Overall Stage Jitter Report ({overall_profiled_frames} frames):")
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
            )"""

content = content.replace(old3, new3)

# 4. Overall Breakdowns
old4 = """        if any(overall_stage_totals[name] > 0.0 for name in breakdown_stage_names):
            print("  - postprocess_breakdown:")
            for stage_name in breakdown_stage_names:
                total_ms = overall_stage_totals[stage_name]
                if total_ms <= 0.0:
                    continue
                print(
                    f"    - {stage_name}: {total_ms / overall_profiled_frames:.2f} ms/frame"
                )
                stage_summary_lines.append(
                    f"{stage_name}\\tmean_ms={total_ms / overall_profiled_frames:.2f}\\ttotal_ms={total_ms:.2f}"
                )
        if any(overall_post_counts.values()):
            print("  - post_counts:")
            for count_name, total_count in overall_post_counts.items():
                mean_count = total_count / overall_profiled_frames
                print(f"    - {count_name}: {mean_count:.1f} boxes/frame")
                stage_summary_lines.append(
                    f"{count_name}\\tmean={mean_count:.1f}\\ttotal={total_count}"
                )"""

new4 = """        if any(overall_stage_totals[name] > 0.0 for name in breakdown_stage_names):
            print("\\n| Postprocess Breakdown | Mean (ms/frame) |")
            print("| :--- | :--- |")
            for stage_name in breakdown_stage_names:
                total_ms = overall_stage_totals[stage_name]
                if total_ms <= 0.0:
                    continue
                print(
                    f"| {stage_name} | {total_ms / overall_profiled_frames:.2f} |"
                )
                stage_summary_lines.append(
                    f"{stage_name}\\tmean_ms={total_ms / overall_profiled_frames:.2f}\\ttotal_ms={total_ms:.2f}"
                )
        if any(overall_post_counts.values()):
            print("\\n| Post Counts | Mean (boxes/frame) |")
            print("| :--- | :--- |")
            for count_name, total_count in overall_post_counts.items():
                mean_count = total_count / overall_profiled_frames
                print(f"| {count_name} | {mean_count:.1f} |")
                stage_summary_lines.append(
                    f"{count_name}\\tmean={mean_count:.1f}\\ttotal={total_count}"
                )"""

content = content.replace(old4, new4)

file_path.write_text(content)
print("done")
