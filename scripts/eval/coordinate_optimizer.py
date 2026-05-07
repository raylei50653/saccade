import sys
import json
import time
import copy
import csv
import torch
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.eval.mot17_args import build_parser  # noqa: E402
from saccade.perception.eval.runner import run_eval  # noqa: E402
from saccade.perception.detector_trt import TRTYoloDetector  # noqa: E402
from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402


class CoordinateOptimizer:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.output_dir / "tuning_history.json"
        self.log_file = self.output_dir / "optimization_log.csv"

        # Load all possible parameters from mot17.py
        self.mot17_parser = build_parser()
        self.all_params = self._get_tunable_params()

        # Current state
        self.current_params = self._get_initial_params()
        self.best_value = -float("inf")
        self.best_params = copy.deepcopy(self.current_params)
        self.best_metrics = {}

        self.history = []
        if self.history_file.exists() and not args.fresh:
            self._load_history()

        # Initialize CSV log if it doesn't exist
        if not self.log_file.exists() or args.fresh:
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "round",
                        "parameter",
                        "action",
                        "value",
                        "idf1",
                        "mota",
                        "ids",
                        "score",
                        "is_best",
                    ]
                )

        # Shared models (In-process optimization)
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Pre-loading models to memory..."
        )
        self.detector = TRTYoloDetector(engine_path=args.engine)

        reid_work_enabled = args.reid_mode != "off" or args.profile_lazy_reid_embeddings
        self.extractor = None
        if reid_work_enabled:
            _reid_engine = args.reid_engine_path or ""
            self.extractor = TRTFeatureExtractor(
                engine_path=_reid_engine,
                model_type=args.reid_model,
                max_batch=64,
            )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Models loaded.")

    def _get_tunable_params(self):
        tunable = {}
        ignored_groups = {"I/O and dataset scope", "Profiling and diagnostics"}

        for group in self.mot17_parser._action_groups:
            if group.title in ignored_groups:
                continue
            for action in group._group_actions:
                if (
                    hasattr(action, "type")
                    and action.type in [float, int]
                    and action.dest != "help"
                ):
                    default = action.default if action.default is not None else 0.0
                    if action.type is int:
                        step = 1
                    else:
                        if "threshold" in action.dest or "thresh" in action.dest:
                            step = 0.01
                        elif "weight" in action.dest or "ema" in action.dest:
                            step = 0.05
                        elif "ratio" in action.dest:
                            step = 0.01
                        else:
                            step = 0.1

                    tunable[action.dest] = {
                        "default": default,
                        "type": action.type,
                        "step": step,
                        "help": action.help,
                    }
        return tunable

    def _get_initial_params(self):
        args_dict = vars(self.args)
        return {k: args_dict.get(k, v["default"]) for k, v in self.all_params.items()}

    def _load_history(self):
        try:
            with open(self.history_file, "r") as f:
                data = json.load(f)
                self.history = data.get("history", [])
                self.best_params = data.get("best_params", self.best_params)
                self.best_value = data.get("best_value", -float("inf"))
                self.current_params = copy.deepcopy(self.best_params)
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Loaded history. Best value: {self.best_value:.4f}"
                )
        except Exception as e:
            print(f"Failed to load history: {e}")

    def _save_history(self):
        data = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "best_metrics": self.best_metrics,
            "history": self.history,
        }
        with open(self.history_file, "w") as f:
            json.dump(data, f, indent=2)

    def _log_trial(
        self, round_idx, param_name, action, param_val, metrics, score, is_best
    ):
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    round_idx,
                    param_name,
                    action,
                    param_val,
                    metrics.get("IDF1", "0"),
                    metrics.get("MOTA", "0"),
                    metrics.get("IDs", "0"),
                    f"{score:.6f}",
                    "YES" if is_best else "NO",
                ]
            )

    def calculate_value(self, metrics):
        idf1 = float(metrics.get("IDF1", "0").strip("%"))
        mota = float(metrics.get("MOTA", "0").strip("%"))
        ids = float(metrics.get("IDs", 0))

        if self.args.formula:
            try:
                context = {"IDF1": idf1, "MOTA": mota, "IDs": ids}
                return eval(self.args.formula, {"__builtins__": None}, context)
            except Exception as e:
                print(f"Error in formula evaluation: {e}")

        return idf1 + 0.5 * mota - 0.001 * ids

    def run_eval_internal(self, params):
        # Merge our tuned params with fixed args
        eval_args = copy.deepcopy(vars(self.args))
        eval_args.update(params)

        # Cleanup internal tracker outputs from previous runs
        tmp_eval_dir = self.output_dir / "tmp_eval"
        if tmp_eval_dir.exists():
            import shutil

            shutil.rmtree(tmp_eval_dir)
        tmp_eval_dir.mkdir(parents=True, exist_ok=True)
        eval_args["output"] = str(tmp_eval_dir)

        # Pass pre-loaded models
        eval_args["detector"] = self.detector
        eval_args["extractor"] = self.extractor

        # Run eval function directly
        try:
            metrics = run_eval(**eval_args)

            # Explicit memory cleanup
            torch.cuda.empty_cache()

            return metrics
        except Exception as e:
            print(f"  Eval error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def optimize(self):
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting Coordinate Optimization (In-Process)..."
        )
        print(
            f"  Target: {len(self.all_params)} parameters, {self.args.rounds} rounds."
        )

        if self.best_value == -float("inf"):
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Running initial evaluation..."
            )
            metrics = self.run_eval_internal(self.current_params)
            if metrics:
                self.best_value = self.calculate_value(metrics)
                self.best_metrics = metrics
                print(
                    f"  Initial Value: {self.best_value:.4f} (IDF1={metrics['IDF1']}, MOTA={metrics['MOTA']}, IDs={metrics['IDs']})"
                )
                self._log_trial(
                    0, "INITIAL", "STAY", "N/A", metrics, self.best_value, True
                )

        for r in range(self.args.rounds):
            round_start_time = time.time()
            print(
                f"\n--- Round {r + 1}/{self.args.rounds} (Current Best: {self.best_value:.4f}) ---"
            )

            param_names = list(self.all_params.keys())

            for name in param_names:
                info = self.all_params[name]
                curr_val = self.current_params[name]
                step = info["step"]

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Optimizing {name} (current: {curr_val})"
                )

                candidates = []
                val_plus = curr_val + step
                if info["type"] is int:
                    val_plus = int(val_plus)
                candidates.append((val_plus, "PLUS"))

                val_minus = curr_val - step
                if info["type"] is int:
                    val_minus = int(val_minus)
                candidates.append((val_minus, "MINUS"))

                improved_this_param = False

                for cand_val, label in candidates:
                    if "threshold" in name or "thresh" in name:
                        if cand_val < 0 or cand_val > 1:
                            continue

                    test_params = copy.deepcopy(self.current_params)
                    test_params[name] = cand_val

                    metrics = self.run_eval_internal(test_params)
                    if not metrics:
                        continue

                    val = self.calculate_value(metrics)
                    is_new_best = val > self.best_value

                    self._log_trial(
                        r + 1, name, label, cand_val, metrics, val, is_new_best
                    )

                    if is_new_best:
                        print(
                            f"    ✅ Improvement ({label}): {self.best_value:.4f} -> {val:.4f} (IDF1: {metrics['IDF1']})"
                        )
                        self.best_value = val
                        self.current_params[name] = cand_val
                        self.best_params = copy.deepcopy(self.current_params)
                        self.best_metrics = metrics
                        improved_this_param = True

                        self.history.append(
                            {
                                "round": r + 1,
                                "parameter": name,
                                "old_value": curr_val,
                                "new_value": cand_val,
                                "score": val,
                                "metrics": metrics,
                                "timestamp": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            }
                        )
                        self._save_history()
                    else:
                        print(f"    ❌ No improvement ({label}): {val:.4f}")

                if not improved_this_param:
                    print("    保持不變")

            elapsed = time.time() - round_start_time
            print(f"\nRound {r + 1} finished in {elapsed / 60:.1f} minutes.")


if __name__ == "__main__":
    parser = build_parser()  # Use mot17.py's full parser

    # Add optimizer-specific arguments that aren't in the base parser
    existing_args = [a.dest for a in parser._actions]

    if "rounds" not in existing_args:
        parser.add_argument("--rounds", type=int, default=20)
    if "formula" not in existing_args:
        parser.add_argument("--formula", default=None)
    if "fresh" not in existing_args:
        parser.add_argument("--fresh", action="store_true")
    if "params" not in existing_args:
        parser.add_argument("--params", default=None)

    # No need to add --reid-engine-path as it's already in build_parser()

    args = parser.parse_args()

    optimizer = CoordinateOptimizer(args)
    if args.params:
        target_params = args.params.split(",")
        optimizer.all_params = {
            k: v for k, v in optimizer.all_params.items() if k in target_params
        }
        print(f"Filtered to {len(optimizer.all_params)} parameters.")

    optimizer.optimize()
