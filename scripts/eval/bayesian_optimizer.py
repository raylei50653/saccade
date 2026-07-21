"""Bayesian hyperparameter optimizer over MOT eval objectives."""

# status: stable
import os
import sys
import copy
import re
import argparse
from pathlib import Path
from datetime import datetime

import torch
import optuna
from optuna.samplers import TPESampler

# Add project root and src to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from scripts.eval.mot17_args import build_parser  # noqa: E402
from saccade.perception.eval.runner import run_eval  # noqa: E402
from saccade.perception.detector_trt import TRTYoloDetector  # noqa: E402
from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402


class BayesianOptimizer:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.storage = os.getenv(
            "OPTUNA_STORAGE",
            "postgresql://saccade:saccade@localhost:5432/optuna",
        )
        self.study_name = (
            args.study_name
            or f"saccade_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Load all possible parameters from mot17.py
        self.mot17_parser = build_parser()
        self.tunable_config = self._discover_tunable_params()

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

    def _discover_tunable_params(self):
        tunable = {}
        ignored_groups = {"I/O and dataset scope", "Profiling and diagnostics"}

        # Filter by --params if provided
        target_params = None
        if self.args.params:
            # Normalize: convert dashes to underscores to match argparse 'dest'
            target_params = {p.replace("-", "_") for p in self.args.params.split(",")}

        for group in self.mot17_parser._action_groups:
            if group.title in ignored_groups:
                continue
            for action in group._group_actions:
                if action.dest == "help" or action.dest is None:
                    continue

                if target_params and action.dest not in target_params:
                    continue

                # Check for Float/Int parameters
                if hasattr(action, "type") and action.type in [float, int]:
                    default = action.default if action.default is not None else 0.0
                    help_str = action.help or ""
                    low, high = self._parse_range(help_str, default, action.type)

                    tunable[action.dest] = {
                        "default": default,
                        "type": action.type,
                        "low": low,
                        "high": high,
                        "help": help_str,
                    }
                # Check for Boolean parameters (ON/OFF)
                elif isinstance(action, argparse.BooleanOptionalAction):
                    tunable[action.dest] = {
                        "default": action.default,
                        "type": bool,
                        "help": action.help,
                    }
        return tunable

    def _parse_range(self, help_str, default, ptype):
        # Look for "Range: 0-1" or "Range: 0.1-2.0"
        range_match = re.search(r"Range: ([\d.]+)-([\d.]+)", help_str)
        if range_match:
            try:
                low = float(range_match.group(1).rstrip("."))
                high = float(range_match.group(2).rstrip("."))
                return low, high
            except ValueError:
                pass

        # Look for "Range: >=0"
        ge_match = re.search(r"Range: >=([\d.]+)", help_str)
        if ge_match:
            try:
                low = float(ge_match.group(1).rstrip("."))
                return low, max(low + 1.0, default * 2.0)
            except ValueError:
                pass

        # Look for "Range: >0"
        gt_match = re.search(r"Range: >([\d.]+)", help_str)
        if gt_match:
            try:
                low = float(gt_match.group(1).rstrip(".")) + 0.0001
                return low, max(low + 1.0, default * 2.0)
            except ValueError:
                pass

        # Heuristics based on name or default
        if 0 <= default <= 1.0:
            return 0.0, 1.0

        if ptype is int:
            return max(0, int(default * 0.5)), int(default * 2.0) + 5

        return default * 0.5, default * 2.0 + 1.0

    def calculate_value(self, metrics):
        # Extract metrics as floats
        try:
            idf1 = float(str(metrics.get("IDF1", "0")).replace("%", ""))
            mota = float(str(metrics.get("MOTA", "0")).replace("%", ""))
            hota = float(str(metrics.get("HOTA", "0")).replace("%", ""))
            ids = float(metrics.get("IDs", 0))
        except (ValueError, TypeError):
            return -100.0

        if self.args.formula:
            try:
                context = {"IDF1": idf1, "MOTA": mota, "HOTA": hota, "IDs": ids}
                return eval(self.args.formula, {"__builtins__": None}, context)
            except Exception as e:
                print(f"Error in formula evaluation: {e}")

        # Default objective: high IDF1 and MOTA, low IDs
        return idf1 + 0.5 * mota - 0.001 * ids

    def objective(self, trial):
        params = copy.deepcopy(vars(self.args))

        # Suggest parameters
        for name, config in self.tunable_config.items():
            if config["type"] is int:
                params[name] = trial.suggest_int(
                    name, int(config["low"]), int(config["high"])
                )
            elif config["type"] is float:
                params[name] = trial.suggest_float(name, config["low"], config["high"])
            elif config["type"] is bool:
                params[name] = trial.suggest_categorical(name, [True, False])

        # Run evaluation
        params["detector"] = self.detector
        params["extractor"] = self.extractor

        # Force output to a temp subdir per trial
        trial_output = self.output_dir / f"trial_{trial.number}"
        params["output"] = str(trial_output)

        try:
            metrics = run_eval(**params)
            score = self.calculate_value(metrics)

            # Store metrics in trial user attributes
            for k, v in metrics.items():
                trial.set_user_attr(k, v)

            # Cleanup trial output to save space
            if trial_output.exists():
                import shutil

                shutil.rmtree(trial_output)

            return score
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            import traceback

            traceback.print_exc()
            return -100.0
        finally:
            torch.cuda.empty_cache()

    def optimize(self):
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting Bayesian Optimization..."
        )
        print(f"  Study: {self.study_name}")
        print(f"  Storage: {self.storage}")
        print(f"  Parameters: {list(self.tunable_config.keys())}")

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",
            load_if_exists=True,
            sampler=TPESampler(n_startup_trials=min(10, self.args.n_trials // 2)),
        )

        # Add default parameters as first trial if study is new
        if len(study.trials) == 0:
            default_params = {k: v["default"] for k, v in self.tunable_config.items()}
            study.enqueue_trial(default_params)
            print("  Enqueued default parameters for first trial.")

        study.optimize(self.objective, n_trials=self.args.n_trials)

        print("\nOptimization Finished!")
        print(f"Best value: {study.best_value}")
        print("Best parameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        print(f"\nTo view dashboard, run: optuna-dashboard {self.storage}")


if __name__ == "__main__":
    parser = build_parser()

    # Optimizer-specific arguments
    opt_group = parser.add_argument_group("Bayesian Optimization Settings")
    opt_group.add_argument(
        "--n-trials", type=int, default=50, help="Number of Optuna trials."
    )
    opt_group.add_argument("--study-name", default=None, help="Optuna study name.")
    opt_group.add_argument(
        "--formula", default=None, help="Formula for score (e.g. 'IDF1 + 0.5 * MOTA')."
    )
    opt_group.add_argument(
        "--params",
        default=None,
        help="Comma-separated list of parameter names to tune.",
    )

    args = parser.parse_args()

    # Adjust some defaults for optimization
    if args.sequences == "":
        print("Warning: Running optimization on ALL sequences might be very slow.")

    optimizer = BayesianOptimizer(args)
    optimizer.optimize()
