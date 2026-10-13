"""Exploratory Random Forest pilot on revised production Module 2 geometry."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cloudchamber-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

from backend.src.dataset_preparation.random_forest_pilot import (
    CLASS_ORDER,
    MODEL_FEATURES,
    RF_PARAMETERS,
    build_dependency_groups,
    calculate_metrics,
    choose_group_splits,
)

from .confirmed_mapping_audit import association_2x2


TITLE = "PRODUCTION-MODULE-2 RF EXPLORATORY PILOT"
PREVIOUS_BALANCED_ACCURACY = 0.8571
PREVIOUS_MACRO_F1 = 0.8555
PREVIOUS_MCC = 0.7313
REQUIRED_METADATA = [
    "ground_truth_track_id",
    "ground_truth_class",
    "source_dataset",
    "source_image",
    "event_id",
    "roi_type",
    "ground_truth_quality",
    "module2_track_id",
    "segmentation_source",
    "processing_status",
]
OOF_COLUMNS = [
    "ground_truth_track_id",
    "true_class",
    "predicted_class",
    "probability_alpha",
    "probability_electron",
    "fold",
    "dependency_group",
    "source_dataset",
    "roi_type",
    "event_id",
    "source_image",
    "module2_track_id",
]
MISCLASSIFICATION_COLUMNS = [
    "ground_truth_track_id",
    "true_class",
    "predicted_class",
    "probability_alpha",
    "probability_electron",
    "source_dataset",
    "roi_type",
    "event_id",
    "module2_track_id",
]
CI_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "alpha_precision",
    "alpha_recall",
    "alpha_f1",
    "electron_precision",
    "electron_recall",
    "electron_f1",
]


@dataclass
class ProductionOOFResult:
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    fold_audit: pd.DataFrame
    feature_importance: pd.DataFrame
    group_count: int
    fold_count: int
    leakage_detected: bool


def load_production_feature_dataset(features_path: str | Path) -> pd.DataFrame:
    """Load the single authorized feature CSV without consulting labels.csv."""
    frame = pd.read_csv(features_path, dtype={column: str for column in REQUIRED_METADATA})
    missing = [column for column in [*REQUIRED_METADATA, *MODEL_FEATURES] if column not in frame]
    if missing:
        raise ValueError(f"Production feature CSV is missing: {', '.join(missing)}")
    if frame["ground_truth_track_id"].duplicated().any():
        raise ValueError("ground_truth_track_id must be unique in the production feature CSV")
    if frame["module2_track_id"].duplicated().any():
        raise ValueError("module2_track_id must be unique in the production feature CSV")
    if not frame["ground_truth_class"].isin(CLASS_ORDER).all():
        raise ValueError("Only Alpha/Electron rows are valid for this pilot")
    if not (frame["processing_status"] == "success").all():
        raise ValueError("Every production pilot row must have processing_status=success")
    if not (frame["segmentation_source"] == "production_module2").all():
        raise ValueError("Every pilot row must use segmentation_source=production_module2")
    for field in ("event_id", "source_image", "source_dataset", "roi_type", "module2_track_id"):
        blank = frame.loc[frame[field].fillna("").str.strip() == "", "ground_truth_track_id"].tolist()
        if blank:
            raise ValueError(f"Blank {field} for: {', '.join(blank)}")
    frame[MODEL_FEATURES] = frame[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(frame[MODEL_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Production pilot predictors contain NaN or infinite values")
    frame = frame.rename(
        columns={"ground_truth_track_id": "track_id", "ground_truth_class": "class"}
    )
    return frame.sort_values("track_id").reset_index(drop=True)


def _group_bootstrap_intervals(
    predictions: pd.DataFrame,
    *,
    repetitions: int,
    random_state: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_state)
    unique_groups = np.asarray(sorted(predictions["dependency_group"].unique()))
    group_rows = {
        group: np.flatnonzero(predictions["dependency_group"].to_numpy() == group)
        for group in unique_groups
    }
    samples: dict[str, list[float]] = {name: [] for name in CI_METRICS}
    valid_repetitions = 0
    for _ in range(repetitions):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indexes = np.concatenate([group_rows[group] for group in sampled_groups])
        subset = predictions.iloc[indexes]
        if subset["true_class"].nunique() < 2:
            continue
        metrics = calculate_metrics(subset["true_class"], subset["predicted_class"])
        valid_repetitions += 1
        for name in CI_METRICS:
            samples[name].append(float(metrics[name]))
    intervals: dict[str, Any] = {
        "bootstrap_requested_repetitions": repetitions,
        "bootstrap_valid_repetitions": valid_repetitions,
        "bootstrap_unit": "dependency_group",
    }
    for name, values in samples.items():
        intervals[f"{name}_ci_low"] = float(np.percentile(values, 2.5)) if values else math.nan
        intervals[f"{name}_ci_high"] = float(np.percentile(values, 97.5)) if values else math.nan
    return intervals


def run_production_grouped_oof(
    frame: pd.DataFrame,
    *,
    bootstrap_repetitions: int = 5000,
    permutation_repeats: int = 50,
) -> ProductionOOFResult:
    """Produce one leakage-free OOF prediction for every input row."""
    groups = build_dependency_groups(frame)
    maximum_folds = min(
        int(groups.nunique()),
        int(frame["class"].value_counts().min()),
    )
    splits = choose_group_splits(
        frame,
        groups,
        maximum_folds=maximum_folds,
        random_state=42,
    )
    x = frame[MODEL_FEATURES].to_numpy(dtype=float)
    y = frame["class"].to_numpy()
    group_values = groups.to_numpy()
    predicted = np.empty(len(frame), dtype=object)
    probabilities = np.full((len(frame), len(CLASS_ORDER)), np.nan)
    fold_assignment = np.zeros(len(frame), dtype=int)
    test_coverage = np.zeros(len(frame), dtype=int)
    fold_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    leakage_detected = False

    for fold, (train, test) in enumerate(splits, start=1):
        leaked_groups = set(group_values[train]) & set(group_values[test])
        leakage_detected = leakage_detected or bool(leaked_groups)
        model = RandomForestClassifier(**RF_PARAMETERS)
        model.fit(x[train], y[train])
        predicted[test] = model.predict(x[test])
        fold_probabilities = model.predict_proba(x[test])
        for class_index, class_name in enumerate(model.classes_):
            probabilities[test, CLASS_ORDER.index(class_name)] = fold_probabilities[:, class_index]
        fold_assignment[test] = fold
        test_coverage[test] += 1
        fold_rows.append({
            "fold": fold,
            "train_n": len(train),
            "test_n": len(test),
            "train_alpha": int(np.sum(y[train] == "Alpha")),
            "train_electron": int(np.sum(y[train] == "Electron")),
            "test_alpha": int(np.sum(y[test] == "Alpha")),
            "test_electron": int(np.sum(y[test] == "Electron")),
            "train_groups": len(set(group_values[train])),
            "test_groups": len(set(group_values[test])),
            "leaked_group_count": len(leaked_groups),
        })
        heldout_importance = permutation_importance(
            model,
            x[test],
            y[test],
            scoring="balanced_accuracy",
            n_repeats=permutation_repeats,
            random_state=42 + fold,
        )
        for feature_index, feature in enumerate(MODEL_FEATURES):
            importance_rows.append({
                "fold": fold,
                "feature": feature,
                "importance": float(heldout_importance.importances_mean[feature_index]),
                "within_fold_std": float(heldout_importance.importances_std[feature_index]),
                "test_n": len(test),
            })

    if not np.all(test_coverage == 1) or not np.isfinite(probabilities).all():
        raise RuntimeError("OOF prediction coverage is incomplete")
    if np.any(pd.Series(frame["track_id"]).duplicated()):
        raise RuntimeError("OOF input identity is not unique")
    predictions = pd.DataFrame({
        "ground_truth_track_id": frame["track_id"],
        "true_class": y,
        "predicted_class": predicted,
        "probability_alpha": probabilities[:, 0],
        "probability_electron": probabilities[:, 1],
        "fold": fold_assignment,
        "dependency_group": groups,
        "source_dataset": frame["source_dataset"],
        "roi_type": frame["roi_type"],
        "event_id": frame["event_id"],
        "source_image": frame["source_image"],
        "module2_track_id": frame["module2_track_id"],
    }, columns=OOF_COLUMNS)
    if predictions["ground_truth_track_id"].duplicated().any():
        raise RuntimeError("A sample received more than one OOF prediction")
    probability_sums = predictions[["probability_alpha", "probability_electron"]].sum(axis=1)
    if not np.allclose(probability_sums, 1.0):
        raise RuntimeError("OOF probabilities do not sum to one")

    metrics = calculate_metrics(predictions["true_class"], predictions["predicted_class"])
    metrics.update(
        _group_bootstrap_intervals(
            predictions,
            repetitions=bootstrap_repetitions,
            random_state=42,
        )
    )
    importance_frame = pd.DataFrame(importance_rows)
    summaries = []
    for feature, group in importance_frame.groupby("feature", sort=False):
        summaries.append({
            "feature": feature,
            "importance_mean": float(np.average(group["importance"], weights=group["test_n"])),
            "importance_std_across_folds": float(group["importance"].std(ddof=1)),
            "mean_within_fold_std": float(group["within_fold_std"].mean()),
            "folds": len(group),
        })
    feature_importance = (
        pd.DataFrame(summaries)
        .sort_values(["importance_mean", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    feature_importance.insert(0, "rank", np.arange(1, len(feature_importance) + 1))
    return ProductionOOFResult(
        predictions=predictions,
        metrics=metrics,
        fold_audit=pd.DataFrame(fold_rows),
        feature_importance=feature_importance,
        group_count=int(groups.nunique()),
        fold_count=len(splits),
        leakage_detected=leakage_detected,
    )


def _stratified_metrics(predictions: pd.DataFrame, field: str) -> pd.DataFrame:
    rows = []
    for value, group in predictions.groupby(field, sort=True):
        rows.append({field: value, **calculate_metrics(group["true_class"], group["predicted_class"])})
    return pd.DataFrame(rows)


def production_source_heldout(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x = frame[MODEL_FEATURES].to_numpy(dtype=float)
    y = frame["class"].to_numpy()
    sources = frame["source_dataset"].to_numpy()
    for train_source, test_source in (("AWAN", "Lecturer"), ("Lecturer", "AWAN")):
        train = np.flatnonzero(sources == train_source)
        test = np.flatnonzero(sources == test_source)
        if set(y[train]) != set(CLASS_ORDER) or set(y[test]) != set(CLASS_ORDER):
            raise ValueError(
                f"Both classes are required for source-held-out {train_source}->{test_source}"
            )
        model = RandomForestClassifier(**RF_PARAMETERS)
        model.fit(x[train], y[train])
        rows.append({
            "train_source": train_source,
            "test_source": test_source,
            "train_n": len(train),
            "train_alpha": int(np.sum(y[train] == "Alpha")),
            "train_electron": int(np.sum(y[train] == "Electron")),
            **calculate_metrics(y[test], model.predict(x[test])),
        })
    return pd.DataFrame(rows)


def _source_association(frame: pd.DataFrame) -> dict[str, Any]:
    table = [
        [
            int(np.sum((frame["source_dataset"] == "AWAN") & (frame["class"] == "Alpha"))),
            int(np.sum((frame["source_dataset"] == "AWAN") & (frame["class"] == "Electron"))),
        ],
        [
            int(np.sum((frame["source_dataset"] == "Lecturer") & (frame["class"] == "Alpha"))),
            int(np.sum((frame["source_dataset"] == "Lecturer") & (frame["class"] == "Electron"))),
        ],
    ]
    return {"table": table, **association_2x2(table)}


def _atomic_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=index)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _plot_confusion(matrix: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 4.8))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=14)
    axis.set_xticks([0, 1], CLASS_ORDER)
    axis.set_yticks([0, 1], CLASS_ORDER)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(f"{TITLE}\nOOF confusion matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_feature_importance(importance: pd.DataFrame, path: Path) -> None:
    ordered = importance.sort_values("importance_mean")
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.barh(ordered["feature"], ordered["importance_mean"], color="#4472c4")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Held-out permutation importance\n(balanced-accuracy decrease)")
    axis.set_title(f"{TITLE}\nFeature importance")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _matrix_text(metrics: Mapping[str, Any]) -> str:
    return (
        f"[[{metrics['alpha_as_alpha']}, {metrics['alpha_as_electron']}], "
        f"[{metrics['electron_as_alpha']}, {metrics['electron_as_electron']}]]"
    )


def _performance_comparison(metrics: Mapping[str, Any]) -> tuple[str, float, float]:
    balanced_difference = float(metrics["balanced_accuracy"]) - PREVIOUS_BALANCED_ACCURACY
    macro_difference = float(metrics["macro_f1"]) - PREVIOUS_MACRO_F1
    if balanced_difference <= -0.10 and macro_difference <= -0.10:
        label = "SUBSTANTIALLY LOWER"
    elif balanced_difference >= 0.10 and macro_difference >= 0.10:
        label = "SUBSTANTIALLY HIGHER"
    else:
        label = "SIMILAR WITHIN A 10-PERCENTAGE-POINT DESCRIPTIVE BAND"
    return label, balanced_difference, macro_difference


def build_summary(
    frame: pd.DataFrame,
    primary: ProductionOOFResult,
    source_metrics: pd.DataFrame,
    roi_metrics: pd.DataFrame,
    heldout: pd.DataFrame,
    misclassified: pd.DataFrame,
    association: Mapping[str, Any],
) -> str:
    metrics = primary.metrics
    comparison, balanced_difference, macro_difference = _performance_comparison(metrics)
    ci_width = float(metrics["balanced_accuracy_ci_high"]) - float(metrics["balanced_accuracy_ci_low"])
    importance = primary.feature_importance
    mean_width_rank = int(importance.loc[importance["feature"] == "mean_width_px", "rank"].iloc[0])
    top_feature = str(importance.iloc[0]["feature"])
    interpretations = ["B. TOO UNSTABLE TO INTERPRET - MORE DATA NEEDED"]
    if comparison == "SUBSTANTIALLY LOWER":
        interpretations.append("C. PERFORMANCE DEGRADES SUBSTANTIALLY WITH PRODUCTION MODULE 2")
    elif float(metrics["balanced_accuracy"]) >= 0.70 and float(metrics["macro_f1"]) >= 0.70:
        interpretations.insert(0, "A. ENCOURAGING FOR CONTINUED MODULE 3 DEVELOPMENT")

    lines = [
        TITLE,
        "Exploratory pilot only. This is not final system accuracy or production performance.",
        "",
        "DATASET AND MODEL",
        f"n={len(frame)}; Alpha={sum(frame['class'] == 'Alpha')}; Electron={sum(frame['class'] == 'Electron')}",
        "Input: revised_confirmed_module2_geometric_features.csv only",
        "Segmentation source: production_module2",
        "Predictors (exactly seven): " + ", ".join(MODEL_FEATURES),
        "Metadata was used only for grouping and diagnostics, never as RF predictors.",
        f"RF parameters: {RF_PARAMETERS}",
        "No hyperparameter search or held-out tuning was performed.",
        "",
        "GROUPING / LEAKAGE",
        "Dependency rule: connected components sharing event_id and/or source_image.",
        f"Dependency groups: {primary.group_count}",
        f"Largest feasible StratifiedGroupKFold count: {primary.fold_count}",
        *[
            f"Fold {int(row.fold)}: train Alpha/Electron={int(row.train_alpha)}/{int(row.train_electron)}, "
            f"test Alpha/Electron={int(row.test_alpha)}/{int(row.test_electron)}, "
            f"train/test groups={int(row.train_groups)}/{int(row.test_groups)}"
            for row in primary.fold_audit.itertuples()
        ],
        f"Leakage detected: {'YES' if primary.leakage_detected else 'NO'}",
        f"OOF coverage: {len(primary.predictions)}/{len(frame)} exactly once",
        "",
        "OOF PERFORMANCE",
        f"Accuracy: {metrics['accuracy']:.6f} (95% group-bootstrap CI {metrics['accuracy_ci_low']:.6f}-{metrics['accuracy_ci_high']:.6f})",
        f"Balanced accuracy: {metrics['balanced_accuracy']:.6f} (95% CI {metrics['balanced_accuracy_ci_low']:.6f}-{metrics['balanced_accuracy_ci_high']:.6f})",
        f"Macro F1: {metrics['macro_f1']:.6f} (95% CI {metrics['macro_f1_ci_low']:.6f}-{metrics['macro_f1_ci_high']:.6f})",
        f"MCC: {metrics['mcc']:.6f} (95% CI {metrics['mcc_ci_low']:.6f}-{metrics['mcc_ci_high']:.6f})",
        f"Alpha precision/recall/F1: {metrics['alpha_precision']:.6f}/{metrics['alpha_recall']:.6f}/{metrics['alpha_f1']:.6f}",
        f"Electron precision/recall/F1: {metrics['electron_precision']:.6f}/{metrics['electron_recall']:.6f}/{metrics['electron_f1']:.6f}",
        f"Confusion matrix rows Alpha/Electron: {_matrix_text(metrics)}",
        f"Group-bootstrap repetitions: {metrics['bootstrap_valid_repetitions']}/{metrics['bootstrap_requested_repetitions']} valid",
        f"UNCERTAINTY WARNING: {'WIDE' if ci_width >= 0.20 else 'NOT WIDE BY THE REPORTED 0.20 WIDTH FLAG'} "
        f"(balanced-accuracy CI width={ci_width:.6f}). The bootstrap resamples dependency groups over fixed OOF predictions; it does not refit models.",
        "",
        "SOURCE PERFORMANCE FROM THE SAME OOF PREDICTIONS",
    ]
    for row in source_metrics.itertuples():
        lines.append(
            f"{row.source_dataset}: n={row.n}, Alpha/Electron={row.alpha_count}/{row.electron_count}, "
            f"balanced accuracy={row.balanced_accuracy:.6f}, macro F1={row.macro_f1:.6f}, "
            f"matrix={_matrix_text(row._asdict())}"
        )
    lines.extend([
        "AWAN subgroup metrics are highly unstable because AWAN has only two Electron examples.",
        "",
        "ROI PERFORMANCE FROM THE SAME OOF PREDICTIONS",
    ])
    for row in roi_metrics.itertuples():
        lines.append(
            f"{row.roi_type}: n={row.n}, Alpha/Electron={row.alpha_count}/{row.electron_count}, "
            f"balanced accuracy={row.balanced_accuracy:.6f}, macro F1={row.macro_f1:.6f}"
        )
    lines.extend([
        "Rotated-ROI performance is not reliable because the subset contains only one rotated Electron.",
        "",
        "SOURCE-HELD-OUT SENSITIVITY (DESCRIPTIVE; NO TUNING)",
    ])
    for row in heldout.itertuples():
        lines.append(
            f"Train {row.train_source} -> Test {row.test_source}: test n={row.n}, "
            f"Alpha/Electron={row.alpha_count}/{row.electron_count}, "
            f"balanced accuracy={row.balanced_accuracy:.6f}, macro F1={row.macro_f1:.6f}, "
            f"matrix={_matrix_text(row._asdict())}"
        )
    lines.extend([
        "Both source-held-out directions are highly unstable because AWAN contains only two Electron samples.",
        "",
        "HELD-OUT PERMUTATION FEATURE IMPORTANCE",
        *[
            f"{int(row.rank)}. {row.feature}: mean={row.importance_mean:.6f}, fold SD={row.importance_std_across_folds:.6f}"
            for row in importance.itertuples()
        ],
        f"Previous temporary-segmentation pilot: mean_width_px ranked first. In this production-segmentation pilot, {top_feature} ranks first (mean_width_px rank={mean_width_rank}).",
        "These importances are coarse and unstable because individual held-out folds contain only 2-9 samples; a zero value does not establish that a feature is irrelevant.",
        "Permutation importance is predictive sensitivity, not physical causality; the comparison is confounded by segmentation and subset changes.",
        "",
        "ERROR ANALYSIS",
        f"Misclassified samples: {len(misclassified)}",
        *[
            f"{row.ground_truth_track_id}: true={row.true_class}, predicted={row.predicted_class}, "
            f"P(Alpha)={row.probability_alpha:.6f}, P(Electron)={row.probability_electron:.6f}, "
            f"source={row.source_dataset}, ROI={row.roi_type}"
            for row in misclassified.itertuples()
        ],
        "No sample was relabelled or removed.",
        "",
        "COMPARISON WITH PREVIOUS TEMPORARY-SEGMENTATION PILOT",
        f"Previous: n=56, balanced accuracy={PREVIOUS_BALANCED_ACCURACY:.4f}, macro F1={PREVIOUS_MACRO_F1:.4f}, MCC={PREVIOUS_MCC:.4f}",
        f"Production Module 2: n={len(frame)}, balanced accuracy={metrics['balanced_accuracy']:.6f}, macro F1={metrics['macro_f1']:.6f}, MCC={metrics['mcc']:.6f}",
        f"Balanced-accuracy difference={balanced_difference:+.6f}; macro-F1 difference={macro_difference:+.6f}",
        f"Descriptive comparison: {comparison}",
        "The difference is not a clean classifier comparison: segmentation changed, the usable subset fell from 56 to 29, and source/ROI selection changed.",
        "",
        "BIAS AND LIMITATIONS",
        f"Class/source contingency: {association['table']}; |phi|={association['phi_cramers_v']:.6f}; Fisher p={association['fisher_p_value']:.6g}",
        "n=29 with 15 Alpha/14 Electron; only two AWAN Electron; only one rotated Electron;",
        "all samples are conditional on Module 2 producing a usable candidate; production segmentation replaced temporary segmentation.",
        "No final system accuracy claim is supported.",
        "",
        "FINAL INTERPRETATION",
        *interpretations,
        "Continued development may be justified only as an exploratory, leakage-aware pilot; more cross-source and rotated-Electron data are needed.",
        "",
        "PRODUCTION MODULE-2 RF PILOT TRAINED: YES",
        "FINAL CLASSIFIER TRAINED: NO",
        "PRODUCTION MODULE 3 CLASSIFIER MODIFIED: NO",
        "",
    ])
    return "\n".join(lines)


def run_production_module2_rf_pilot(
    features_path: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_repetitions: int = 5000,
    permutation_repeats: int = 50,
) -> dict[str, Any]:
    frame = load_production_feature_dataset(features_path)
    primary = run_production_grouped_oof(
        frame,
        bootstrap_repetitions=bootstrap_repetitions,
        permutation_repeats=permutation_repeats,
    )
    source_metrics = _stratified_metrics(primary.predictions, "source_dataset")
    roi_metrics = _stratified_metrics(primary.predictions, "roi_type")
    heldout = production_source_heldout(frame)
    misclassified = primary.predictions.loc[
        primary.predictions["true_class"] != primary.predictions["predicted_class"],
        MISCLASSIFICATION_COLUMNS,
    ].copy()
    association = _source_association(frame)
    matrix = confusion_matrix(
        primary.predictions["true_class"],
        primary.predictions["predicted_class"],
        labels=CLASS_ORDER,
    )
    metrics = pd.DataFrame([{
        "scope": "production_module2_grouped_oof",
        "title": TITLE,
        "groups": primary.group_count,
        "folds": primary.fold_count,
        "leakage_detected": primary.leakage_detected,
        **primary.metrics,
    }])
    confusion = pd.DataFrame({
        "true_class": CLASS_ORDER,
        "predicted_alpha": matrix[:, 0],
        "predicted_electron": matrix[:, 1],
    })
    summary = build_summary(
        frame,
        primary,
        source_metrics,
        roi_metrics,
        heldout,
        misclassified,
        association,
    )

    output = Path(output_dir)
    _atomic_text(output / "production_rf_pilot_summary.txt", summary)
    _atomic_csv(metrics, output / "production_rf_metrics.csv")
    _atomic_csv(confusion, output / "production_rf_confusion_matrix.csv")
    _atomic_csv(primary.predictions, output / "production_rf_oof_predictions.csv")
    _atomic_csv(source_metrics, output / "production_rf_source_metrics.csv")
    _atomic_csv(roi_metrics, output / "production_rf_roi_metrics.csv")
    _atomic_csv(heldout, output / "production_rf_source_heldout.csv")
    _atomic_csv(primary.feature_importance, output / "production_rf_feature_importance.csv")
    _atomic_csv(misclassified, output / "production_rf_misclassifications.csv")
    _plot_confusion(matrix, output / "plots" / "confusion_matrix.png")
    _plot_feature_importance(
        primary.feature_importance, output / "plots" / "feature_importance.png"
    )
    return {
        "frame": frame,
        "primary": primary,
        "source_metrics": source_metrics,
        "roi_metrics": roi_metrics,
        "source_heldout": heldout,
        "misclassified": misclassified,
        "source_association": association,
        "summary": summary,
        "output_dir": output,
    }
