import os
import glob
from pathlib import Path
from collections import OrderedDict
from saccade.perception.eval.metrics import _calculate_hota


def evaluate_folder(output_dir: str):
    data_root = "datasets/MOT17"
    split = "train"

    gt_folder = os.path.join(data_root, split)
    gtfiles = glob.glob(os.path.join(gt_folder, "*/gt/gt.txt"))
    tsfiles = sorted(glob.glob(os.path.join(output_dir, "*.txt")))

    if not gtfiles or not tsfiles:
        return None

    gt_by_name = OrderedDict((Path(f).parts[-3], f) for f in gtfiles)
    jobs = []
    for ts_path in tsfiles:
        ts_name = Path(ts_path).stem
        matched_gt = None
        for gt_name in gt_by_name.keys():
            if ts_name == gt_name or ts_name.startswith(gt_name + "_"):
                matched_gt = gt_name
                break
        if matched_gt:
            jobs.append((matched_gt, gt_by_name[matched_gt], ts_path))

    if not jobs:
        return None

    hota = _calculate_hota(data_root, split, output_dir, jobs)
    return hota


def main():
    results_dir = Path("results")
    print("| Folder | HOTA |")
    print("|---|---|")
    for folder_path in sorted(results_dir.iterdir()):
        if not folder_path.is_dir():
            continue
        hota = evaluate_folder(str(folder_path))
        if hota is not None:
            print(f"| {folder_path.name} | {hota['HOTA'] * 100:.1f}% |")


if __name__ == "__main__":
    main()
