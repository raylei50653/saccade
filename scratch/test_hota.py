import sys
import argparse
from pathlib import Path

# NumPy 2.0 backward compatibility monkeypatch for TrackEval
import numpy as np

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

# Add TrackEval to Python path
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "third_party" / "TrackEval")
)
import trackeval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--tracker-name", default=None)
    args = parser.parse_args()

    results_path = Path(args.results).resolve()
    tracker_name = args.tracker_name or results_path.name
    trackers_folder = results_path.parent

    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config["DISPLAY_LESS_PROGRESS"] = True
    default_eval_config["PRINT_CONFIG"] = False

    default_dataset_config = (
        trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    )

    # Custom config to match our results layout
    dataset_config = {
        **default_dataset_config,
        "GT_FOLDER": "datasets/MOT17/train",
        "SKIP_SPLIT_FOL": True,
        "TRACKERS_FOLDER": str(trackers_folder),
        "TRACKER_SUB_FOLDER": "",
        "TRACKERS_TO_EVAL": [tracker_name],
        "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train",
        "SEQ_INFO": {
            "MOT17-02-SDP": None,
            "MOT17-04-SDP": None,
            "MOT17-05-SDP": None,
            "MOT17-09-SDP": None,
            "MOT17-10-SDP": None,
            "MOT17-11-SDP": None,
            "MOT17-13-SDP": None,
        },
    }

    metrics_config = {"METRICS": ["HOTA"]}

    evaluator = trackeval.Evaluator(default_eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    metrics_list = [trackeval.metrics.HOTA(metrics_config)]

    print(f"Evaluating HOTA on {results_path}...")
    evaluator.evaluate(dataset_list, metrics_list)


if __name__ == "__main__":
    main()
