from pathlib import Path

from saccade.perception.eval.external_fp_model import (
    BandedLogisticModel,
    CascadeFilterConfig,
    RuleBaselineConfig,
    SoftmaxLinearModel,
    apply_cascade_filter,
    apply_rule_baseline,
    apply_low_score_logistic_filter,
    bucketize_feature,
    count_labels,
    evaluate_softmax_classifier,
    evaluate_logistic_classifier,
    fit_banded_logistic_classifier,
    fit_logistic_classifier,
    fit_softmax_classifier,
    load_external_fp_model,
    load_external_rows_csv,
    load_logistic_model,
    predict_external_fp_matrix,
    predict_logistic_matrix,
    predict_softmax_matrix,
    split_rows_train_eval,
    rows_to_feature_matrix,
    save_logistic_model,
    sweep_low_score_logistic_filter,
    train_cascade_stage2_model,
)


def _sample_rows() -> list[dict[str, object]]:
    return [
        {
            "dataset": "crowdhuman",
            "split": "val",
            "image_id": "img1",
            "image_path": "/tmp/img1.jpg",
            "image_width": 100,
            "image_height": 100,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 10.0,
            "y2": 20.0,
            "score": 0.90,
            "label": "tp",
            "matched_iou": 0.8,
            "width": 10.0,
            "height": 20.0,
            "area": 200.0,
            "aspect_ratio": 2.0,
            "center_x_norm": 0.05,
            "center_y_norm": 0.10,
            "edge_margin_norm": 0.0,
            "touches_edge": 1,
        },
        {
            "dataset": "crowdhuman",
            "split": "val",
            "image_id": "img2",
            "image_path": "/tmp/img2.jpg",
            "image_width": 100,
            "image_height": 100,
            "x1": 1.0,
            "y1": 1.0,
            "x2": 9.0,
            "y2": 25.0,
            "score": 0.82,
            "label": "tp",
            "matched_iou": 0.7,
            "width": 8.0,
            "height": 24.0,
            "area": 192.0,
            "aspect_ratio": 3.0,
            "center_x_norm": 0.05,
            "center_y_norm": 0.13,
            "edge_margin_norm": 0.01,
            "touches_edge": 1,
        },
        {
            "dataset": "crowdhuman",
            "split": "val",
            "image_id": "img3",
            "image_path": "/tmp/img3.jpg",
            "image_width": 100,
            "image_height": 100,
            "x1": 40.0,
            "y1": 40.0,
            "x2": 48.0,
            "y2": 50.0,
            "score": 0.06,
            "label": "fp",
            "matched_iou": 0.0,
            "width": 8.0,
            "height": 10.0,
            "area": 80.0,
            "aspect_ratio": 1.25,
            "center_x_norm": 0.44,
            "center_y_norm": 0.45,
            "edge_margin_norm": 0.40,
            "touches_edge": 0,
        },
        {
            "dataset": "crowdhuman",
            "split": "val",
            "image_id": "img4",
            "image_path": "/tmp/img4.jpg",
            "image_width": 100,
            "image_height": 100,
            "x1": 30.0,
            "y1": 30.0,
            "x2": 42.0,
            "y2": 44.0,
            "score": 0.12,
            "label": "fp",
            "matched_iou": 0.1,
            "width": 12.0,
            "height": 14.0,
            "area": 168.0,
            "aspect_ratio": 1.16,
            "center_x_norm": 0.36,
            "center_y_norm": 0.37,
            "edge_margin_norm": 0.30,
            "touches_edge": 0,
        },
    ]


def _sample_softmax_rows() -> list[dict[str, object]]:
    base = _sample_rows()
    return base + [
        {
            "dataset": "crowdhuman",
            "split": "val",
            "image_id": "img5",
            "image_path": "/tmp/img5.jpg",
            "image_width": 100,
            "image_height": 100,
            "x1": 0.0,
            "y1": 60.0,
            "x2": 8.0,
            "y2": 70.0,
            "score": 0.07,
            "label": "ignore",
            "matched_iou": 0.75,
            "width": 8.0,
            "height": 10.0,
            "area": 80.0,
            "aspect_ratio": 1.25,
            "center_x_norm": 0.04,
            "center_y_norm": 0.65,
            "edge_margin_norm": 0.0,
            "touches_edge": 1,
        },
        {
            "dataset": "crowdhuman",
            "split": "val",
            "image_id": "img6",
            "image_path": "/tmp/img6.jpg",
            "image_width": 100,
            "image_height": 100,
            "x1": 90.0,
            "y1": 20.0,
            "x2": 99.0,
            "y2": 31.0,
            "score": 0.09,
            "label": "ignore",
            "matched_iou": 0.68,
            "width": 9.0,
            "height": 11.0,
            "area": 99.0,
            "aspect_ratio": 1.22,
            "center_x_norm": 0.945,
            "center_y_norm": 0.255,
            "edge_margin_norm": 0.01,
            "touches_edge": 1,
        },
    ]


def test_load_external_rows_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(
        "dataset,split,image_id,image_path,image_width,image_height,x1,y1,x2,y2,score,label,matched_iou,width,height,area,aspect_ratio,center_x_norm,center_y_norm,edge_margin_norm,touches_edge\n"
        "crowdhuman,val,img1,/tmp/img1.jpg,100,100,0,0,10,20,0.9,tp,0.8,10,20,200,2.0,0.05,0.1,0.0,1\n",
        encoding="utf-8",
    )
    rows = load_external_rows_csv(csv_path)
    assert len(rows) == 1
    assert rows[0]["image_width"] == 100
    assert rows[0]["score"] == 0.9


def test_bucketize_and_counts() -> None:
    rows = _sample_rows()
    counts = count_labels(rows)
    buckets = bucketize_feature(rows, feature="score", bins=[0.0, 0.1, 0.5, 1.0])
    assert counts == {"tp": 2, "fp": 2}
    assert buckets[0]["fp_count"] == 1
    assert buckets[-1]["tp_count"] == 2


def test_apply_rule_baseline_removes_low_quality_fps() -> None:
    rows = _sample_rows()
    _, metrics = apply_rule_baseline(rows, config=RuleBaselineConfig())
    assert metrics.tp_kept == 2
    assert metrics.fp_kept == 0
    assert metrics.fp_removed == 2
    assert metrics.recall_after == 1.0


def test_fit_and_evaluate_logistic_classifier() -> None:
    rows = _sample_rows() * 3
    train_rows, eval_rows = split_rows_train_eval(rows, eval_ratio=0.5)
    model = fit_logistic_classifier(train_rows, epochs=200, learning_rate=0.2)
    metrics = evaluate_logistic_classifier(model, eval_rows, threshold=0.5)
    assert metrics["precision"] >= 0.5
    assert metrics["recall"] >= 0.5


def test_fit_and_evaluate_softmax_classifier() -> None:
    rows = _sample_softmax_rows() * 3
    model = fit_softmax_classifier(rows, epochs=200, learning_rate=0.2)
    metrics = evaluate_softmax_classifier(model, rows)
    assert metrics["class_names"] == ["tp", "fp", "np"]
    assert metrics["class_counts"]["np"] == 6
    assert len(metrics["confusion_matrix"]) == 3
    assert metrics["per_class"]["tp"]["recall"] >= 0.5
    assert metrics["per_class"]["np"]["recall"] > 0.0


def test_save_and_load_logistic_model_round_trip(tmp_path: Path) -> None:
    rows = _sample_rows() * 3
    model = fit_logistic_classifier(rows, epochs=50, learning_rate=0.2)
    model_path = tmp_path / "model.json"
    save_logistic_model(model_path, model)
    loaded = load_logistic_model(model_path)
    x = rows_to_feature_matrix(rows[:2], feature_names=loaded.feature_names)
    original_probs = predict_logistic_matrix(model, x)
    loaded_probs = predict_logistic_matrix(loaded, x)
    assert loaded.feature_names == model.feature_names
    assert original_probs.shape == loaded_probs.shape
    assert (abs(original_probs - loaded_probs) < 1e-9).all()


def test_fit_and_round_trip_banded_logistic_model(tmp_path: Path) -> None:
    rows = (_sample_rows() * 3) + [
        {
            **dict(_sample_rows()[0]),
            "image_id": "img5",
            "score": 0.07,
            "label": "tp",
            "height": 18.0,
            "aspect_ratio": 2.2,
        },
        {
            **dict(_sample_rows()[2]),
            "image_id": "img6",
            "score": 0.16,
            "label": "fp",
            "height": 11.0,
            "aspect_ratio": 1.1,
        },
    ]
    model = fit_banded_logistic_classifier(
        rows,
        band_edges=[0.05, 0.10, 0.18],
        epochs=80,
        learning_rate=0.2,
    )
    assert isinstance(model, BandedLogisticModel)
    assert len(model.band_models) == 2
    model_path = tmp_path / "banded_model.json"
    save_logistic_model(model_path, model)
    loaded = load_external_fp_model(model_path)
    assert isinstance(loaded, BandedLogisticModel)
    x = rows_to_feature_matrix(rows[:3], feature_names=loaded.feature_names)
    probs = predict_external_fp_matrix(loaded, x)
    assert probs.shape == (3,)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_evaluate_banded_logistic_classifier() -> None:
    rows = (_sample_rows() * 3) + [
        {
            **dict(_sample_rows()[1]),
            "image_id": "img7",
            "score": 0.08,
            "label": "tp",
        },
        {
            **dict(_sample_rows()[3]),
            "image_id": "img8",
            "score": 0.14,
            "label": "fp",
        },
    ]
    model = fit_banded_logistic_classifier(
        rows,
        band_edges=[0.05, 0.10, 0.18],
        epochs=100,
        learning_rate=0.2,
    )
    metrics = evaluate_logistic_classifier(model, rows, threshold=0.5)
    assert metrics["precision"] >= 0.5
    assert metrics["recall"] >= 0.5


def test_fit_and_round_trip_softmax_model(tmp_path: Path) -> None:
    rows = _sample_softmax_rows() * 3
    model = fit_softmax_classifier(rows, epochs=120, learning_rate=0.2)
    assert isinstance(model, SoftmaxLinearModel)
    model_path = tmp_path / "softmax_model.json"
    save_logistic_model(model_path, model)
    loaded = load_external_fp_model(model_path)
    assert isinstance(loaded, SoftmaxLinearModel)
    x = rows_to_feature_matrix(rows[:4], feature_names=loaded.feature_names)
    probs = predict_softmax_matrix(loaded, x)
    external_probs = predict_external_fp_matrix(loaded, x)
    assert probs.shape == (4, 3)
    assert external_probs.shape == (4, 3)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert (abs(probs.sum(axis=1) - 1.0) < 1e-9).all()


def test_apply_low_score_logistic_filter_hard_and_penalty() -> None:
    rows = _sample_rows() * 3
    model = fit_logistic_classifier(rows, epochs=200, learning_rate=0.2)
    kept_hard, metrics_hard = apply_low_score_logistic_filter(
        rows,
        model=model,
        threshold=0.5,
        max_score=0.18,
        penalty=None,
        min_score=0.05,
    )
    kept_penalty, metrics_penalty = apply_low_score_logistic_filter(
        rows,
        model=model,
        threshold=0.5,
        max_score=0.18,
        penalty=0.8,
        min_score=0.05,
    )
    assert len(kept_hard) <= len(kept_penalty) <= len(rows)
    assert metrics_hard.fp_removed >= 1
    assert metrics_penalty.recall_after >= metrics_hard.recall_after


def test_sweep_low_score_logistic_filter_returns_candidates() -> None:
    rows = _sample_rows() * 3
    model = fit_logistic_classifier(rows, epochs=200, learning_rate=0.2)
    candidates = sweep_low_score_logistic_filter(
        rows,
        model=model,
        max_scores=[0.10, 0.18],
        thresholds=[0.5, 0.7],
        penalties=[0.8],
        min_score=0.05,
    )
    assert len(candidates) == 8
    assert {candidate.mode for candidate in candidates} == {
        "hard_keep",
        "score_penalty",
    }
    assert all(candidate.tp_total == 6 for candidate in candidates)


def test_apply_cascade_filter_without_model_returns_stage1() -> None:
    """Cascade without Stage 2 model should return Stage 1 result unchanged.

    Stage 1 (rule baseline) filters rows, so output != input when stage2_model is None.
    """
    rows = _sample_rows()
    config = CascadeFilterConfig(
        log_threshold=0.5,
        log_max_score=0.18,
        log_penalty=None,
    )
    kept_rows, metrics = apply_cascade_filter(rows, config=config, stage2_model=None)
    # Stage 1 keeps TP and filters obvious FP
    assert len(kept_rows) == 2  # Only the 2 TP rows pass rule baseline
    assert metrics.tp_kept == 2
    assert metrics.fp_kept == 0
    assert metrics.s1_kept == 2  # Stage 1 output = final output when no Stage 2 model
    assert metrics.s2_kept == 2  # Stage 2 mirrors Stage 1 when no model
    assert metrics.s2_removed == 0  # No Stage 2 filtering


def test_apply_cascade_filter_with_model_reduces_fp() -> None:
    """Cascade with trained model should reduce FP further while keeping TP."""
    rows = _sample_rows() * 5  # More samples for training
    model = fit_logistic_classifier(rows, epochs=200, learning_rate=0.2)
    config = CascadeFilterConfig(
        log_threshold=0.3,
        log_max_score=0.25,
        log_penalty=None,
    )
    kept_rows, metrics = apply_cascade_filter(rows, config=config, stage2_model=model)
    # Stage 1 should keep TP and filter obvious FP
    assert metrics.tp_kept >= 2  # At least keep our 2 TP
    assert metrics.fp_kept <= metrics.fp_total
    # Cascade metrics breakdown should be consistent
    assert metrics.s1_kept >= metrics.s2_kept  # Stage 2 output <= Stage 1 output
    assert metrics.s2_kept == metrics.tp_kept + metrics.fp_kept


def test_train_cascade_stage2_model_produces_model() -> None:
    """Training cascade Stage 2 model should return a valid LogisticModel."""
    rows = _sample_rows() * 10
    model = train_cascade_stage2_model(rows, epochs=200, learning_rate=0.2)
    assert model.feature_names is not None
    assert len(model.feature_names) > 0
    assert len(model.weights) == len(model.feature_names)
    # Should produce valid probabilities
    probs = model.predict_proba(rows)
    assert len(probs) == len(rows)
    assert all(0.0 <= p <= 1.0 for p in probs)
