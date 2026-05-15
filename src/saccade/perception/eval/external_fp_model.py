import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

STRUCTURAL_FEATURE_COLUMNS = [
    "score",
    "width",
    "height",
    "area",
    "aspect_ratio",
    "center_x_norm",
    "center_y_norm",
    "edge_margin_norm",
    "touches_edge",
]
SOFTMAX3_CLASS_NAMES = ("tp", "fp", "np")


@dataclass(frozen=True)
class RuleBaselineConfig:
    min_score: float = 0.05
    low_score: float = 0.10
    medium_score: float = 0.18
    min_height: float = 72.0
    medium_height: float = 96.0
    min_aspect: float = 1.6


@dataclass(frozen=True)
class RuleBaselineMetrics:
    tp_total: int
    fp_total: int
    tp_kept: int
    fp_kept: int
    tp_removed: int
    fp_removed: int
    precision_before: float
    precision_after: float
    recall_after: float
    fp_reduction: float


@dataclass(frozen=True)
class FilterEvalMetrics:
    tp_total: int
    fp_total: int
    tp_kept: int
    fp_kept: int
    tp_removed: int
    fp_removed: int
    precision_before: float
    precision_after: float
    recall_after: float
    fp_reduction: float


@dataclass(frozen=True)
class SweepCandidate:
    mode: str
    max_score: float
    threshold: float
    penalty: float
    tp_total: int
    fp_total: int
    tp_kept: int
    fp_kept: int
    tp_removed: int
    fp_removed: int
    precision_before: float
    precision_after: float
    recall_after: float
    fp_reduction: float


@dataclass(frozen=True)
class CascadeFilterConfig:
    """Two-stage cascade: rule baseline + logistic classifier."""

    # Stage 1: rule baseline config
    rule: RuleBaselineConfig = None  # type: ignore[assignment]
    # Stage 2: logistic filter params
    log_threshold: float = 0.50
    log_max_score: float = 0.18
    log_penalty: float | None = None
    log_min_score: float = 0.05

    def __post_init__(self) -> None:
        if self.rule is None:
            object.__setattr__(self, "rule", RuleBaselineConfig())


@dataclass(frozen=True)
class CascadeMetrics:
    tp_total: int
    fp_total: int
    tp_kept: int
    fp_kept: int
    tp_removed: int
    fp_removed: int
    precision_before: float
    precision_after: float
    recall_after: float
    fp_reduction: float
    # Stage breakdown
    s1_kept: int
    s1_removed: int
    s2_kept: int
    s2_removed: int


@dataclass(frozen=True)
class LogisticModel:
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def predict_proba(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        features = rows_to_feature_matrix(rows, feature_names=self.feature_names)
        standardized = _standardize(features, self.mean, self.std)
        logits = standardized @ np.asarray(self.weights, dtype=np.float64) + self.bias
        return 1.0 / (1.0 + np.exp(-logits))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "model_type": "logistic",
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
            "bias": self.bias,
            "mean": list(self.mean),
            "std": list(self.std),
        }


@dataclass(frozen=True)
class BandedLogisticModel:
    feature_names: tuple[str, ...]
    band_edges: tuple[float, ...]
    band_models: tuple[LogisticModel, ...]

    def predict_proba(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        features = rows_to_feature_matrix(rows, feature_names=self.feature_names)
        return predict_external_fp_matrix(self, features)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "model_type": "banded_logistic",
            "feature_names": list(self.feature_names),
            "band_edges": list(self.band_edges),
            "band_models": [model.to_json_dict() for model in self.band_models],
        }


@dataclass(frozen=True)
class SoftmaxLinearModel:
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def predict_proba(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        features = rows_to_feature_matrix(rows, feature_names=self.feature_names)
        return predict_softmax_matrix(self, features)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "model_type": "softmax3",
            "class_names": list(self.class_names),
            "feature_names": list(self.feature_names),
            "weights": [list(row) for row in self.weights],
            "bias": list(self.bias),
            "mean": list(self.mean),
            "std": list(self.std),
        }


def load_external_fp_model(
    path: Path,
) -> "LogisticModel | BandedLogisticModel | SoftmaxLinearModel":
    data = json.loads(path.read_text(encoding="utf-8"))
    model_type = str(data.get("model_type", "logistic")).lower()
    if model_type == "banded_logistic":
        return BandedLogisticModel(
            feature_names=tuple(str(name) for name in data["feature_names"]),
            band_edges=tuple(float(v) for v in data["band_edges"]),
            band_models=tuple(
                LogisticModel(
                    feature_names=tuple(
                        str(name) for name in band_data["feature_names"]
                    ),
                    weights=tuple(float(v) for v in band_data["weights"]),
                    bias=float(band_data["bias"]),
                    mean=tuple(float(v) for v in band_data["mean"]),
                    std=tuple(float(v) for v in band_data["std"]),
                )
                for band_data in data["band_models"]
            ),
        )
    if model_type == "softmax3":
        return SoftmaxLinearModel(
            feature_names=tuple(str(name) for name in data["feature_names"]),
            class_names=tuple(
                str(name).lower()
                for name in data.get("class_names", SOFTMAX3_CLASS_NAMES)
            ),
            weights=tuple(tuple(float(v) for v in row) for row in data["weights"]),
            bias=tuple(float(v) for v in data["bias"]),
            mean=tuple(float(v) for v in data["mean"]),
            std=tuple(float(v) for v in data["std"]),
        )
    return LogisticModel(
        feature_names=tuple(str(name) for name in data["feature_names"]),
        weights=tuple(float(v) for v in data["weights"]),
        bias=float(data["bias"]),
        mean=tuple(float(v) for v in data["mean"]),
        std=tuple(float(v) for v in data["std"]),
    )


def load_logistic_model(
    path: Path,
) -> "LogisticModel | BandedLogisticModel | SoftmaxLinearModel":
    return load_external_fp_model(path)


def load_external_rows_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed = dict(row)
            for key in [
                "image_width",
                "image_height",
                "touches_edge",
            ]:
                parsed[key] = int(parsed[key])
            for key in [
                "x1",
                "y1",
                "x2",
                "y2",
                "score",
                "matched_iou",
                "width",
                "height",
                "area",
                "aspect_ratio",
                "center_x_norm",
                "center_y_norm",
                "edge_margin_norm",
            ]:
                parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def count_labels(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["label"]).lower()
        counts[label] = counts.get(label, 0) + 1
    return counts


def compute_quantiles(
    rows: Sequence[dict[str, Any]],
    *,
    feature: str,
    labels: Iterable[str],
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> dict[str, dict[float, float]]:
    result: dict[str, dict[float, float]] = {}
    for label in labels:
        values = [
            float(row[feature])
            for row in rows
            if str(row["label"]).lower() == str(label).lower()
        ]
        if not values:
            result[str(label).lower()] = {}
            continue
        arr = np.asarray(values, dtype=np.float64)
        result[str(label).lower()] = {
            float(q): float(np.quantile(arr, q)) for q in quantiles
        }
    return result


def bucketize_feature(
    rows: Sequence[dict[str, Any]],
    *,
    feature: str,
    bins: Sequence[float],
    labels: Iterable[str] = ("tp", "fp"),
) -> list[dict[str, Any]]:
    if len(bins) < 2:
        raise ValueError("bins must have at least two edges")
    label_names = [str(label).lower() for label in labels]
    summaries: list[dict[str, Any]] = []
    values_by_label = {
        label: np.asarray(
            [float(row[feature]) for row in rows if str(row["label"]).lower() == label],
            dtype=np.float64,
        )
        for label in label_names
    }
    for idx in range(len(bins) - 1):
        lower = float(bins[idx])
        upper = float(bins[idx + 1])
        entry: dict[str, Any] = {
            "range": f"[{lower:.3f}, {upper:.3f})",
            "lower": lower,
            "upper": upper,
        }
        for label in label_names:
            values = values_by_label[label]
            count = int(np.sum((values >= lower) & (values < upper)))
            total = max(int(values.size), 1)
            entry[f"{label}_count"] = count
            entry[f"{label}_share"] = count / total
        summaries.append(entry)
    return summaries


def apply_rule_baseline(
    rows: Sequence[dict[str, Any]],
    *,
    config: RuleBaselineConfig | None = None,
) -> tuple[list[dict[str, Any]], RuleBaselineMetrics]:
    cfg = config or RuleBaselineConfig()
    kept_rows: list[dict[str, Any]] = []
    tp_total = 0
    fp_total = 0
    tp_kept = 0
    fp_kept = 0
    for row in rows:
        label = str(row["label"]).lower()
        if label == "tp":
            tp_total += 1
        elif label == "fp":
            fp_total += 1
        keep = _rule_keep(row, cfg)
        if keep:
            kept_rows.append(dict(row))
            if label == "tp":
                tp_kept += 1
            elif label == "fp":
                fp_kept += 1
    precision_before = tp_total / max(tp_total + fp_total, 1)
    precision_after = tp_kept / max(tp_kept + fp_kept, 1)
    recall_after = tp_kept / max(tp_total, 1)
    fp_reduction = (fp_total - fp_kept) / max(fp_total, 1)
    metrics = RuleBaselineMetrics(
        tp_total=tp_total,
        fp_total=fp_total,
        tp_kept=tp_kept,
        fp_kept=fp_kept,
        tp_removed=tp_total - tp_kept,
        fp_removed=fp_total - fp_kept,
        precision_before=precision_before,
        precision_after=precision_after,
        recall_after=recall_after,
        fp_reduction=fp_reduction,
    )
    return kept_rows, metrics


def apply_low_score_logistic_filter(
    rows: Sequence[dict[str, Any]],
    *,
    model: "LogisticModel | BandedLogisticModel",
    threshold: float,
    max_score: float,
    penalty: float | None = None,
    min_score: float = 0.05,
) -> tuple[list[dict[str, Any]], FilterEvalMetrics]:
    filtered_rows = [
        dict(row) for row in rows if str(row["label"]).lower() in {"tp", "fp"}
    ]
    if not filtered_rows:
        empty = FilterEvalMetrics(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
        return [], empty

    probs = model.predict_proba(filtered_rows)
    kept_rows: list[dict[str, Any]] = []
    tp_total = 0
    fp_total = 0
    tp_kept = 0
    fp_kept = 0
    for row, prob in zip(filtered_rows, probs, strict=False):
        label = str(row["label"]).lower()
        if label == "tp":
            tp_total += 1
        else:
            fp_total += 1
        score = float(row["score"])
        keep = True
        if score <= max_score and prob < threshold:
            if penalty is None:
                keep = False
            else:
                adjusted_score = score * penalty
                row["adjusted_score"] = adjusted_score
                keep = adjusted_score >= min_score
        if keep:
            kept_rows.append(row)
            if label == "tp":
                tp_kept += 1
            else:
                fp_kept += 1
    return kept_rows, _build_filter_eval_metrics(
        tp_total=tp_total,
        fp_total=fp_total,
        tp_kept=tp_kept,
        fp_kept=fp_kept,
    )


def apply_cascade_filter(
    rows: Sequence[dict[str, Any]],
    *,
    config: CascadeFilterConfig | None = None,
    stage2_model: LogisticModel | BandedLogisticModel | None = None,
) -> tuple[list[dict[str, Any]], CascadeMetrics]:
    """Two-stage cascade: rule baseline (Stage 1) + logistic filter (Stage 2).

    Stage 1 applies the rule baseline to quickly drop low-score / small FP.
    Stage 2 runs the logistic classifier on the Stage 1 output to further
    prune the remaining FP while preserving more TP.
    """
    cfg = config or CascadeFilterConfig()

    # Stage 1: rule baseline (zero-cost)
    stage1_rows, s1_metrics = apply_rule_baseline(rows, config=cfg.rule)
    s1_kept = len(stage1_rows)
    s1_removed = len(rows) - s1_kept

    # Stage 2: logistic filter on Stage 1 output
    if stage2_model is None:
        # No model provided — return Stage 1 result as final
        s2_kept = s1_kept
        s2_removed = 0
        final_rows = list(stage1_rows)
        tp_kept = s1_metrics.tp_kept
        fp_kept = s1_metrics.fp_kept
    else:
        stage2_rows, s2_metrics = apply_low_score_logistic_filter(
            stage1_rows,
            model=stage2_model,
            threshold=cfg.log_threshold,
            max_score=cfg.log_max_score,
            penalty=cfg.log_penalty,
            min_score=cfg.log_min_score,
        )
        final_rows = stage2_rows
        tp_kept = s2_metrics.tp_kept
        fp_kept = s2_metrics.fp_kept
        s2_kept = len(stage2_rows)
        s2_removed = s1_kept - s2_kept

    precision_before = s1_metrics.tp_total / max(
        s1_metrics.tp_total + s1_metrics.fp_total, 1
    )
    precision_after = tp_kept / max(tp_kept + fp_kept, 1)
    recall_after = tp_kept / max(s1_metrics.tp_total, 1)
    fp_reduction = (s1_metrics.fp_total - fp_kept) / max(s1_metrics.fp_total, 1)

    metrics = CascadeMetrics(
        tp_total=s1_metrics.tp_total,
        fp_total=s1_metrics.fp_total,
        tp_kept=tp_kept,
        fp_kept=fp_kept,
        tp_removed=s1_metrics.tp_total - tp_kept,
        fp_removed=s1_metrics.fp_total - fp_kept,
        precision_before=precision_before,
        precision_after=precision_after,
        recall_after=recall_after,
        fp_reduction=fp_reduction,
        s1_kept=s1_kept,
        s1_removed=s1_removed,
        s2_kept=s2_kept,
        s2_removed=s2_removed,
    )
    return final_rows, metrics


def sweep_cascade_config(
    rows: Sequence[dict[str, Any]],
    *,
    model: LogisticModel | None = None,
    rule_configs: Sequence[RuleBaselineConfig] | None = None,
    log_thresholds: Sequence[float] | None = None,
    log_max_scores: Sequence[float] | None = None,
    log_penalties: Sequence[float | None] | None = None,
) -> list[SweepCandidate]:
    """Sweep cascade configurations to find Pareto-optimal combos."""
    if rule_configs is None:
        rule_configs = [RuleBaselineConfig()]
    if log_thresholds is None:
        log_thresholds = [0.50]
    if log_max_scores is None:
        log_max_scores = [0.18]
    if log_penalties is None:
        log_penalties = [None]

    candidates: list[SweepCandidate] = []
    for rc in rule_configs:
        for lt in log_thresholds:
            for lms in log_max_scores:
                for lp in log_penalties:
                    cfg = CascadeFilterConfig(
                        rule=rc,
                        log_threshold=lt,
                        log_max_score=lms,
                        log_penalty=lp,
                    )
                    m = model if model is not None else None
                    _, metrics = apply_cascade_filter(rows, config=cfg, stage2_model=m)
                    c = SweepCandidate(
                        mode="cascade",
                        max_score=float(lms),
                        threshold=float(lt),
                        penalty=float(lp) if lp is not None else 0.0,
                        tp_total=metrics.tp_total,
                        fp_total=metrics.fp_total,
                        tp_kept=metrics.tp_kept,
                        fp_kept=metrics.fp_kept,
                        tp_removed=metrics.tp_removed,
                        fp_removed=metrics.fp_removed,
                        precision_before=metrics.precision_before,
                        precision_after=metrics.precision_after,
                        recall_after=metrics.recall_after,
                        fp_reduction=metrics.fp_reduction,
                    )
                    candidates.append(c)
    return candidates


def sweep_low_score_logistic_filter(
    rows: Sequence[dict[str, Any]],
    *,
    model: LogisticModel,
    max_scores: Sequence[float],
    thresholds: Sequence[float],
    penalties: Sequence[float],
    min_score: float = 0.05,
) -> list[SweepCandidate]:
    candidates: list[SweepCandidate] = []
    for max_score in max_scores:
        for threshold in thresholds:
            _, hard_metrics = apply_low_score_logistic_filter(
                rows,
                model=model,
                threshold=threshold,
                max_score=max_score,
                penalty=None,
                min_score=min_score,
            )
            candidates.append(
                SweepCandidate(
                    mode="hard_keep",
                    max_score=float(max_score),
                    threshold=float(threshold),
                    penalty=1.0,
                    **hard_metrics.__dict__,
                )
            )
            for penalty in penalties:
                _, penalty_metrics = apply_low_score_logistic_filter(
                    rows,
                    model=model,
                    threshold=threshold,
                    max_score=max_score,
                    penalty=penalty,
                    min_score=min_score,
                )
                candidates.append(
                    SweepCandidate(
                        mode="score_penalty",
                        max_score=float(max_score),
                        threshold=float(threshold),
                        penalty=float(penalty),
                        **penalty_metrics.__dict__,
                    )
                )
    return candidates


def rows_to_feature_matrix(
    rows: Sequence[dict[str, Any]],
    *,
    feature_names: Sequence[str] = STRUCTURAL_FEATURE_COLUMNS,
) -> np.ndarray:
    matrix = np.asarray(
        [[float(row[feature_name]) for feature_name in feature_names] for row in rows],
        dtype=np.float64,
    )
    if matrix.ndim != 2:
        raise ValueError("feature matrix must be 2D")
    return matrix


def binary_labels_from_rows(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    labels = []
    for row in rows:
        label = str(row["label"]).lower()
        if label not in {"tp", "fp"}:
            continue
        labels.append(1.0 if label == "tp" else 0.0)
    return np.asarray(labels, dtype=np.float64)


def multiclass_labels_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    class_names: Sequence[str] = SOFTMAX3_CLASS_NAMES,
) -> np.ndarray:
    label_to_idx = {str(name).lower(): idx for idx, name in enumerate(class_names)}
    alias_to_idx = dict(label_to_idx)
    if "np" in label_to_idx:
        alias_to_idx.setdefault("ignore", label_to_idx["np"])
    labels: list[int] = []
    for row in rows:
        label = str(row["label"]).lower()
        if label not in alias_to_idx:
            continue
        labels.append(alias_to_idx[label])
    return np.asarray(labels, dtype=np.int64)


def fit_logistic_classifier(
    rows: Sequence[dict[str, Any]],
    *,
    feature_names: Sequence[str] = STRUCTURAL_FEATURE_COLUMNS,
    epochs: int = 400,
    learning_rate: float = 0.1,
    l2: float = 1e-4,
) -> LogisticModel:
    train_rows = [row for row in rows if str(row["label"]).lower() in {"tp", "fp"}]
    if not train_rows:
        raise ValueError("No TP/FP rows available for training")
    x = rows_to_feature_matrix(train_rows, feature_names=feature_names)
    y = binary_labels_from_rows(train_rows)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_std = (x - mean) / std
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    sample_count = max(x.shape[0], 1)
    for _ in range(max(epochs, 1)):
        logits = x_std @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        error = probs - y
        grad_w = (x_std.T @ error) / sample_count + l2 * weights
        grad_b = float(np.sum(error) / sample_count)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return LogisticModel(
        feature_names=tuple(str(name) for name in feature_names),
        weights=tuple(float(v) for v in weights.tolist()),
        bias=float(bias),
        mean=tuple(float(v) for v in mean.tolist()),
        std=tuple(float(v) for v in std.tolist()),
    )


def fit_banded_logistic_classifier(
    rows: Sequence[dict[str, Any]],
    *,
    band_edges: Sequence[float],
    feature_names: Sequence[str] = STRUCTURAL_FEATURE_COLUMNS,
    epochs: int = 400,
    learning_rate: float = 0.1,
    l2: float = 1e-4,
) -> BandedLogisticModel:
    train_rows = [row for row in rows if str(row["label"]).lower() in {"tp", "fp"}]
    if len(band_edges) < 3:
        raise ValueError("band_edges must define at least two score bands")
    if not train_rows:
        raise ValueError("No TP/FP rows available for training")
    global_model = fit_logistic_classifier(
        train_rows,
        feature_names=feature_names,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    band_models: list[LogisticModel] = []
    for lower, upper in zip(band_edges[:-1], band_edges[1:], strict=False):
        band_rows = [
            row
            for row in train_rows
            if float(row["score"]) >= float(lower)
            and float(row["score"]) < float(upper)
        ]
        if len(band_rows) < 8 or not _has_both_classes(band_rows):
            band_models.append(global_model)
            continue
        band_models.append(
            fit_logistic_classifier(
                band_rows,
                feature_names=feature_names,
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
            )
        )
    return BandedLogisticModel(
        feature_names=tuple(str(name) for name in feature_names),
        band_edges=tuple(float(v) for v in band_edges),
        band_models=tuple(band_models),
    )


def fit_softmax_classifier(
    rows: Sequence[dict[str, Any]],
    *,
    feature_names: Sequence[str] = STRUCTURAL_FEATURE_COLUMNS,
    class_names: Sequence[str] = SOFTMAX3_CLASS_NAMES,
    class_weight_multipliers: Sequence[float] | None = None,
    epochs: int = 400,
    learning_rate: float = 0.1,
    l2: float = 1e-4,
) -> SoftmaxLinearModel:
    normalized_class_names = tuple(str(name).lower() for name in class_names)
    alias_names = set(normalized_class_names)
    if "np" in alias_names:
        alias_names.add("ignore")
    train_rows = [row for row in rows if str(row["label"]).lower() in alias_names]
    if not train_rows:
        raise ValueError("No TP/FP/NP rows available for training")
    x = rows_to_feature_matrix(train_rows, feature_names=feature_names)
    y = multiclass_labels_from_rows(train_rows, class_names=normalized_class_names)
    required_classes = set(range(len(normalized_class_names)))
    if set(y.tolist()) != required_classes:
        missing = [
            normalized_class_names[idx]
            for idx in sorted(required_classes - set(y.tolist()))
        ]
        raise ValueError(f"Missing training rows for classes: {', '.join(missing)}")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_std = (x - mean) / std
    class_count = len(normalized_class_names)
    weights = np.zeros((x.shape[1], class_count), dtype=np.float64)
    bias = np.zeros(class_count, dtype=np.float64)
    class_counts = np.bincount(y, minlength=class_count).astype(np.float64)
    class_weights = class_counts.sum() / np.maximum(class_counts * class_count, 1.0)
    if class_weight_multipliers is not None:
        if len(class_weight_multipliers) != class_count:
            raise ValueError(f"class_weight_multipliers must have length {class_count}")
        class_weights = class_weights * np.asarray(
            class_weight_multipliers, dtype=np.float64
        )
    sample_weights = class_weights[y]
    sample_weight_total = max(float(np.sum(sample_weights)), 1.0)
    targets = np.eye(class_count, dtype=np.float64)[y]
    for _ in range(max(epochs, 1)):
        logits = x_std @ weights + bias
        probs = _softmax(logits)
        error = (probs - targets) * sample_weights[:, None]
        grad_w = (x_std.T @ error) / sample_weight_total + l2 * weights
        grad_b = np.sum(error, axis=0) / sample_weight_total
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return SoftmaxLinearModel(
        feature_names=tuple(str(name) for name in feature_names),
        class_names=normalized_class_names,
        weights=tuple(tuple(float(v) for v in row.tolist()) for row in weights),
        bias=tuple(float(v) for v in bias.tolist()),
        mean=tuple(float(v) for v in mean.tolist()),
        std=tuple(float(v) for v in std.tolist()),
    )


def evaluate_logistic_classifier(
    model: "LogisticModel | BandedLogisticModel",
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    eval_rows = [row for row in rows if str(row["label"]).lower() in {"tp", "fp"}]
    if not eval_rows:
        raise ValueError("No TP/FP rows available for evaluation")
    y_true = binary_labels_from_rows(eval_rows)
    probs = model.predict_proba(eval_rows)
    y_pred = (probs >= threshold).astype(np.float64)
    tp = int(np.sum((y_true == 1.0) & (y_pred == 1.0)))
    fp = int(np.sum((y_true == 0.0) & (y_pred == 1.0)))
    tn = int(np.sum((y_true == 0.0) & (y_pred == 0.0)))
    fn = int(np.sum((y_true == 1.0) & (y_pred == 0.0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "threshold": threshold,
    }


def evaluate_softmax_classifier(
    model: SoftmaxLinearModel,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    alias_names = set(model.class_names)
    if "np" in alias_names:
        alias_names.add("ignore")
    eval_rows = [row for row in rows if str(row["label"]).lower() in alias_names]
    if not eval_rows:
        raise ValueError("No TP/FP/NP rows available for evaluation")
    y_true = multiclass_labels_from_rows(eval_rows, class_names=model.class_names)
    probs = model.predict_proba(eval_rows)
    y_pred = np.argmax(probs, axis=1)
    class_count = len(model.class_names)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true.tolist(), y_pred.tolist(), strict=False):
        confusion[true_idx, pred_idx] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for idx, name in enumerate(model.class_names):
        tp = int(confusion[idx, idx])
        predicted = int(np.sum(confusion[:, idx]))
        support = int(np.sum(confusion[idx, :]))
        precision = tp / max(predicted, 1)
        recall = tp / max(support, 1)
        f1 = (
            0.0
            if (precision + recall) <= 1e-12
            else 2.0 * precision * recall / (precision + recall)
        )
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    accuracy = float(np.trace(confusion) / max(int(np.sum(confusion)), 1))
    macro_f1 = float(np.mean([float(metrics["f1"]) for metrics in per_class.values()]))
    class_counts = {
        name: int(np.sum(y_true == idx)) for idx, name in enumerate(model.class_names)
    }
    return {
        "class_names": list(model.class_names),
        "class_counts": class_counts,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def split_rows_train_eval(
    rows: Sequence[dict[str, Any]],
    *,
    eval_ratio: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image_id"]), []).append(dict(row))
    image_ids = sorted(grouped)
    if not image_ids:
        return [], []
    eval_count = max(1, int(math.ceil(len(image_ids) * eval_ratio)))
    eval_ids = set(image_ids[-eval_count:])
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for image_id, image_rows in grouped.items():
        if image_id in eval_ids:
            eval_rows.extend(image_rows)
        else:
            train_rows.extend(image_rows)
    if _has_both_classes(train_rows) and _has_both_classes(eval_rows):
        return train_rows, eval_rows
    filtered_rows = [
        dict(row) for row in rows if str(row["label"]).lower() in {"tp", "fp"}
    ]
    if len(filtered_rows) < 2:
        return train_rows, eval_rows
    tp_rows = [row for row in filtered_rows if str(row["label"]).lower() == "tp"]
    fp_rows = [row for row in filtered_rows if str(row["label"]).lower() == "fp"]
    if not tp_rows or not fp_rows:
        return train_rows, eval_rows
    tp_eval_count = min(
        max(1, int(math.ceil(len(tp_rows) * eval_ratio))), len(tp_rows) - 1
    )
    fp_eval_count = min(
        max(1, int(math.ceil(len(fp_rows) * eval_ratio))), len(fp_rows) - 1
    )
    if tp_eval_count <= 0 or fp_eval_count <= 0:
        return train_rows, eval_rows
    fallback_eval = tp_rows[-tp_eval_count:] + fp_rows[-fp_eval_count:]
    fallback_train = tp_rows[:-tp_eval_count] + fp_rows[:-fp_eval_count]
    if _has_both_classes(fallback_train) and _has_both_classes(fallback_eval):
        return fallback_train, fallback_eval
    return train_rows, eval_rows


def save_logistic_model(
    path: Path,
    model: "LogisticModel | BandedLogisticModel | SoftmaxLinearModel",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_json_dict(), indent=2) + "\n", encoding="utf-8")


def predict_logistic_matrix(
    model: LogisticModel,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    standardized = _standardize(feature_matrix, model.mean, model.std)
    logits = standardized @ np.asarray(model.weights, dtype=np.float64) + model.bias
    return 1.0 / (1.0 + np.exp(-logits))


def predict_softmax_matrix(
    model: SoftmaxLinearModel,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    standardized = _standardize(feature_matrix, model.mean, model.std)
    weights = np.asarray(model.weights, dtype=np.float64)
    bias = np.asarray(model.bias, dtype=np.float64)
    logits = standardized @ weights + bias
    return _softmax(logits)


def predict_external_fp_matrix(
    model: "LogisticModel | BandedLogisticModel | SoftmaxLinearModel",
    feature_matrix: np.ndarray,
) -> np.ndarray:
    if isinstance(model, LogisticModel):
        return predict_logistic_matrix(model, feature_matrix)
    if isinstance(model, SoftmaxLinearModel):
        return predict_softmax_matrix(model, feature_matrix)
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be 2D")
    if not model.band_models:
        raise ValueError("banded model has no band models")
    score_idx = model.feature_names.index("score")
    scores = feature_matrix[:, score_idx]
    band_probs = np.stack(
        [
            predict_logistic_matrix(band_model, feature_matrix)
            for band_model in model.band_models
        ],
        axis=1,
    )
    band_weights = _compute_band_weights(scores, model.band_edges)
    return np.sum(band_probs * band_weights, axis=1)  # type: ignore[no-any-return]


def _rule_keep(row: dict[str, Any], cfg: RuleBaselineConfig) -> bool:
    score = float(row["score"])
    height = float(row["height"])
    aspect = float(row["aspect_ratio"])
    if score < cfg.min_score:
        return False
    if score < cfg.low_score and height < cfg.min_height:
        return False
    if (
        score < cfg.medium_score
        and height < cfg.medium_height
        and aspect < cfg.min_aspect
    ):
        return False
    return True


def _build_filter_eval_metrics(
    *,
    tp_total: int,
    fp_total: int,
    tp_kept: int,
    fp_kept: int,
) -> FilterEvalMetrics:
    precision_before = tp_total / max(tp_total + fp_total, 1)
    precision_after = tp_kept / max(tp_kept + fp_kept, 1)
    recall_after = tp_kept / max(tp_total, 1)
    fp_reduction = (fp_total - fp_kept) / max(fp_total, 1)
    return FilterEvalMetrics(
        tp_total=tp_total,
        fp_total=fp_total,
        tp_kept=tp_kept,
        fp_kept=fp_kept,
        tp_removed=tp_total - tp_kept,
        fp_removed=fp_total - fp_kept,
        precision_before=precision_before,
        precision_after=precision_after,
        recall_after=recall_after,
        fp_reduction=fp_reduction,
    )


def _standardize(
    x: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float64)
    std_arr = np.asarray(std, dtype=np.float64)
    return (x - mean_arr) / std_arr


def _softmax(logits: np.ndarray) -> np.ndarray:
    stabilized = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(stabilized)
    import typing

    return typing.cast(
        np.ndarray, exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    )


def _compute_band_weights(
    scores: np.ndarray, band_edges: Sequence[float]
) -> np.ndarray:
    centers = np.asarray(
        [
            (float(lower) + float(upper)) * 0.5
            for lower, upper in zip(band_edges[:-1], band_edges[1:], strict=False)
        ],
        dtype=np.float64,
    )
    if centers.size == 0:
        raise ValueError("band_edges must define at least one band")
    distances = np.abs(scores[:, None] - centers[None, :])
    if centers.size == 1:
        return np.ones((scores.shape[0], 1), dtype=np.float64)
    spans = np.empty_like(centers)
    spans[0] = max(centers[1] - centers[0], 1e-6)
    spans[-1] = max(centers[-1] - centers[-2], 1e-6)
    if centers.size > 2:
        spans[1:-1] = np.maximum((centers[2:] - centers[:-2]) * 0.5, 1e-6)
    raw = np.maximum(0.0, 1.0 - (distances / spans[None, :]))
    zero_rows = np.where(raw.sum(axis=1) <= 1e-12)[0]
    if zero_rows.size > 0:
        nearest = np.argmin(distances[zero_rows], axis=1)
        raw[zero_rows] = 0.0
        raw[zero_rows, nearest] = 1.0
    res: np.ndarray = raw / raw.sum(axis=1, keepdims=True)
    return res


def train_cascade_stage2_model(
    rows: Sequence[dict[str, Any]],
    *,
    rule_config: RuleBaselineConfig | None = None,
    epochs: int = 400,
    learning_rate: float = 0.1,
    l2: float = 1e-4,
) -> LogisticModel:
    """Train a logistic model specifically for the cascade Stage 2.

    Stage 2 sees the output of the rule baseline — only "hard" FP
    (score ≥ rule thresholds). Training on the same distribution
    avoids distribution mismatch that caused the original logistic
    to underperform the rule baseline.

    The training data is the rule-baseline output (Stage 1 kept rows),
    so the model learns to distinguish hard FP from TP in that subspace.
    """
    # First, apply rule baseline to get Stage 1 output
    stage1_rows, _ = apply_rule_baseline(rows, config=rule_config)
    return fit_logistic_classifier(
        stage1_rows,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )


def load_cascade_config(path: Path) -> dict[str, Any]:
    """Load cascade config JSON with rule params + logistic model path."""
    import typing

    data = typing.cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    return data


def apply_cascade_from_json(
    rows: Sequence[dict[str, Any]],
    *,
    cascade_config_path: Path | None = None,
    model_path: Path | None = None,
) -> tuple[list[dict[str, Any]], CascadeMetrics]:
    """Apply cascade filter with config loaded from JSON.

    cascade_config_path: JSON with rule params + logistic params
    model_path: JSON path for the logistic model (overrides model in cascade_config_path)
    """
    if cascade_config_path is None and model_path is None:
        raise ValueError(
            "Must provide at least one of cascade_config_path or model_path"
        )

    # Build config
    rc = RuleBaselineConfig()
    lt = 0.50
    lms = 0.18
    lp = None

    if cascade_config_path is not None:
        data = load_cascade_config(cascade_config_path)
        if "rule" in data:
            rc_data = data["rule"]
            rc = RuleBaselineConfig(
                min_score=float(rc_data.get("min_score", 0.05)),
                low_score=float(rc_data.get("low_score", 0.10)),
                medium_score=float(rc_data.get("medium_score", 0.18)),
                min_height=float(rc_data.get("min_height", 72.0)),
                medium_height=float(rc_data.get("medium_height", 96.0)),
                min_aspect=float(rc_data.get("min_aspect", 1.6)),
            )
        lt = float(data.get("log_threshold", 0.50))
        lms = float(data.get("log_max_score", 0.18))
        if "log_penalty" in data:
            lp = float(data["log_penalty"]) if data["log_penalty"] is not None else None
        if model_path is None and "model_path" in data:
            model_path = Path(data["model_path"])

    config = CascadeFilterConfig(
        rule=rc,
        log_threshold=lt,
        log_max_score=lms,
        log_penalty=lp,
    )

    # Load model
    stage2_model = None
    if model_path is not None and model_path.exists():
        loaded_model = load_external_fp_model(model_path)
        import typing

        stage2_model = typing.cast("LogisticModel | BandedLogisticModel", loaded_model)

    return apply_cascade_filter(rows, config=config, stage2_model=stage2_model)


def _has_both_classes(rows: Sequence[dict[str, Any]]) -> bool:
    labels = {str(row["label"]).lower() for row in rows}
    return "tp" in labels and "fp" in labels
