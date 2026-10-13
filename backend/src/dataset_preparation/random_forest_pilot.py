"""Isolated, group-aware Random Forest pilot evaluation on existing geometry."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cloudchamber-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold

from backend.src.geometric_classification.model_contract import (
    AUTHORIZED_CLASS_NAMES,
    AUTHORIZED_FEATURE_NAMES,
)
from .source_confound_analysis import categorical_association, class_source_contingency


# Compatibility aliases: both point to the authoritative shared contract.
MODEL_FEATURES = AUTHORIZED_FEATURE_NAMES
CLASS_ORDER = AUTHORIZED_CLASS_NAMES
RF_PARAMETERS = {"n_estimators": 100, "random_state": 42, "class_weight": "balanced"}
OOF_COLUMNS = [
    "track_id",
    "true_label",
    "predicted_label",
    "probability_alpha",
    "probability_electron",
    "fold",
    "event_id",
    "source_dataset",
    "roi_type",
    "ground_truth_quality",
]


@dataclass
class CVResult:
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    splits: list[tuple[np.ndarray, np.ndarray]]
    permutation_importance: pd.DataFrame
    impurity_importance: pd.DataFrame


def load_pilot_dataset(labels_path: str | Path, features_path: str | Path) -> pd.DataFrame:
    """Align authoritative labels with existing successful feature rows by track ID."""

    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    features = pd.read_csv(features_path)
    if labels["track_id"].duplicated().any() or features["track_id"].duplicated().any():
        raise ValueError("track_id must be unique in labels and feature CSVs")
    labels = labels[labels["class"].isin(CLASS_ORDER)].copy()
    if set(labels["track_id"]) != set(features["track_id"]):
        missing = sorted(set(labels["track_id"]) - set(features["track_id"]))
        extra = sorted(set(features["track_id"]) - set(labels["track_id"]))
        raise ValueError(f"Feature/label track IDs do not align; missing={missing}, extra={extra}")
    for field in ("event_id", "source_dataset", "source_image"):
        missing_ids = labels.loc[labels[field].str.strip() == "", "track_id"].tolist()
        if missing_ids:
            raise ValueError(f"Blank {field} for: {', '.join(missing_ids)}")
    missing_features = [feature for feature in MODEL_FEATURES if feature not in features.columns]
    if missing_features:
        raise ValueError(f"Feature CSV is missing: {', '.join(missing_features)}")
    if "processing_status" in features and not (features["processing_status"] == "success").all():
        raise ValueError("All pilot feature rows must have processing_status=success")
    metadata = labels[
        [
            "track_id",
            "class",
            "event_id",
            "source_dataset",
            "source_image",
            "roi_type",
            "ground_truth_quality",
        ]
    ]
    frame = metadata.merge(features[["track_id", "segmentation_source", *MODEL_FEATURES]], on="track_id", validate="one_to_one")
    frame[MODEL_FEATURES] = frame[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    values = frame[MODEL_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Pilot features contain NaN or infinite values")
    return frame.sort_values("track_id").reset_index(drop=True)


def build_dependency_groups(frame: pd.DataFrame) -> pd.Series:
    """Union records sharing event_id or source_image into one leakage group."""

    parents = list(range(len(frame)))

    def find(item: int) -> int:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for field in ("event_id", "source_image"):
        first_index: dict[str, int] = {}
        for index, raw_value in enumerate(frame[field]):
            value = str(raw_value).strip()
            if not value:
                continue
            if value in first_index:
                union(first_index[value], index)
            else:
                first_index[value] = index
    roots = [find(index) for index in range(len(frame))]
    root_order = {root: group for group, root in enumerate(sorted(set(roots)), start=1)}
    return pd.Series([f"dependency_group_{root_order[root]:03d}" for root in roots], index=frame.index, name="group")


def choose_group_splits(
    frame: pd.DataFrame,
    groups: pd.Series,
    *,
    maximum_folds: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Choose the highest feasible StratifiedGroupKFold count without leakage."""

    y = frame["class"].to_numpy()
    x = frame[MODEL_FEATURES].to_numpy()
    group_values = groups.to_numpy()
    for folds in range(min(maximum_folds, groups.nunique()), 1, -1):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
        splits = list(splitter.split(x, y, group_values))
        feasible = True
        for train, test in splits:
            if set(y[train]) != set(CLASS_ORDER) or set(y[test]) != set(CLASS_ORDER):
                feasible = False
                break
            if set(group_values[train]) & set(group_values[test]):
                feasible = False
                break
        if feasible:
            return splits
    raise ValueError("No group-aware split with both classes in every train/test fold is feasible")


def calculate_metrics(true: Iterable[str], predicted: Iterable[str]) -> dict[str, Any]:
    y_true, y_pred = np.asarray(list(true)), np.asarray(list(predicted))
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "alpha_count": int(np.sum(y_true == "Alpha")),
        "electron_count": int(np.sum(y_true == "Electron")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "alpha_precision": float(precision[0]),
        "alpha_recall": float(recall[0]),
        "alpha_f1": float(f1[0]),
        "electron_precision": float(precision[1]),
        "electron_recall": float(recall[1]),
        "electron_f1": float(f1[1]),
        "alpha_as_alpha": int(matrix[0, 0]),
        "alpha_as_electron": int(matrix[0, 1]),
        "electron_as_alpha": int(matrix[1, 0]),
        "electron_as_electron": int(matrix[1, 1]),
    }


def _bootstrap_intervals(predictions: pd.DataFrame, groups: pd.Series, repetitions: int = 1000) -> dict[str, float]:
    rng = np.random.default_rng(42)
    unique_groups = np.asarray(sorted(groups.unique()))
    group_rows = {group: np.flatnonzero(groups.to_numpy() == group) for group in unique_groups}
    values: dict[str, list[float]] = {name: [] for name in ("accuracy", "balanced_accuracy", "macro_f1", "mcc")}
    for _ in range(repetitions):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indexes = np.concatenate([group_rows[group] for group in sampled])
        subset = predictions.iloc[indexes]
        if subset["true_label"].nunique() < 2:
            continue
        metrics = calculate_metrics(subset["true_label"], subset["predicted_label"])
        for name in values:
            values[name].append(metrics[name])
    intervals: dict[str, float] = {}
    for name, samples in values.items():
        intervals[f"{name}_ci_low"] = float(np.percentile(samples, 2.5)) if samples else np.nan
        intervals[f"{name}_ci_high"] = float(np.percentile(samples, 97.5)) if samples else np.nan
    return intervals


def run_grouped_oof(
    frame: pd.DataFrame,
    *,
    maximum_folds: int = 5,
    random_state: int = 42,
    bootstrap: bool = True,
) -> CVResult:
    groups = build_dependency_groups(frame)
    splits = choose_group_splits(frame, groups, maximum_folds=maximum_folds, random_state=random_state)
    x = frame[MODEL_FEATURES].to_numpy(dtype=float)
    y = frame["class"].to_numpy()
    predicted = np.empty(len(frame), dtype=object)
    probabilities = np.full((len(frame), 2), np.nan)
    folds = np.zeros(len(frame), dtype=int)
    permutation_rows, impurity_rows = [], []
    for fold, (train, test) in enumerate(splits, start=1):
        model = RandomForestClassifier(**{**RF_PARAMETERS, "random_state": random_state})
        model.fit(x[train], y[train])
        predicted[test] = model.predict(x[test])
        fold_probabilities = model.predict_proba(x[test])
        for class_index, class_name in enumerate(model.classes_):
            probabilities[test, CLASS_ORDER.index(class_name)] = fold_probabilities[:, class_index]
        folds[test] = fold
        permutation = permutation_importance(
            model,
            x[test],
            y[test],
            scoring="balanced_accuracy",
            n_repeats=20,
            random_state=random_state + fold,
        )
        for index, feature in enumerate(MODEL_FEATURES):
            permutation_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "importance": float(permutation.importances_mean[index]),
                    "within_fold_std": float(permutation.importances_std[index]),
                    "test_n": len(test),
                }
            )
            impurity_rows.append(
                {"fold": fold, "feature": feature, "importance": float(model.feature_importances_[index])}
            )
    if np.any(folds == 0) or not np.isfinite(probabilities).all():
        raise RuntimeError("OOF prediction coverage is incomplete")
    output = pd.DataFrame(
        {
            "track_id": frame["track_id"],
            "true_label": y,
            "predicted_label": predicted,
            "probability_alpha": probabilities[:, 0],
            "probability_electron": probabilities[:, 1],
            "fold": folds,
            "event_id": frame["event_id"],
            "source_dataset": frame["source_dataset"],
            "roi_type": frame["roi_type"],
            "ground_truth_quality": frame["ground_truth_quality"],
        },
        columns=OOF_COLUMNS,
    )
    metrics = calculate_metrics(output["true_label"], output["predicted_label"])
    if bootstrap:
        metrics.update(_bootstrap_intervals(output, groups))
    permutation_frame = pd.DataFrame(permutation_rows)
    permutation_summary = pd.DataFrame(
        [
            {
                "feature": feature,
                "importance_mean": float(np.average(group["importance"], weights=group["test_n"])),
                "importance_std_across_folds": float(group["importance"].std(ddof=1)),
            }
            for feature, group in permutation_frame.groupby("feature", sort=False)
        ]
    ).sort_values("importance_mean", ascending=False)
    impurity_summary = (
        pd.DataFrame(impurity_rows)
        .groupby("feature")["importance"]
        .agg(importance_mean="mean", importance_std_across_folds="std")
        .reset_index()
        .sort_values("importance_mean", ascending=False)
    )
    return CVResult(output, metrics, splits, permutation_summary, impurity_summary)


def stratified_metrics(predictions: pd.DataFrame, field: str) -> pd.DataFrame:
    rows = []
    for value, group in predictions.groupby(field, sort=True):
        rows.append({field: value, **calculate_metrics(group["true_label"], group["predicted_label"])})
    return pd.DataFrame(rows)


def source_heldout_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x = frame[MODEL_FEATURES].to_numpy(dtype=float)
    y = frame["class"].to_numpy()
    sources = frame["source_dataset"].to_numpy()
    for train_source, test_source in (("AWAN", "Lecturer"), ("Lecturer", "AWAN")):
        train, test = np.flatnonzero(sources == train_source), np.flatnonzero(sources == test_source)
        if set(y[train]) != set(CLASS_ORDER) or set(y[test]) != set(CLASS_ORDER):
            raise ValueError(f"Both classes are required in source-held-out comparison {train_source}->{test_source}")
        model = RandomForestClassifier(**RF_PARAMETERS)
        model.fit(x[train], y[train])
        rows.append(
            {
                "train_source": train_source,
                "test_source": test_source,
                **calculate_metrics(y[test], model.predict(x[test])),
            }
        )
    return pd.DataFrame(rows)


def _plot_confusion(matrix: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.5, 4.8))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=14)
    axis.set_xticks([0, 1], CLASS_ORDER)
    axis.set_yticks([0, 1], CLASS_ORDER)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Exploratory pilot OOF confusion matrix")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_importance(importance: pd.DataFrame, path: Path) -> None:
    ordered = importance.sort_values("importance_mean")
    fig, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(ordered["feature"], ordered["importance_mean"], color="#4472c4")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("OOF fold permutation importance (balanced accuracy decrease)")
    axis.set_title("Exploratory pilot feature importance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_random_forest_pilot(
    labels_path: str | Path,
    features_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    frame = load_pilot_dataset(labels_path, features_path)
    groups = build_dependency_groups(frame)
    primary = run_grouped_oof(frame)
    source_metrics = stratified_metrics(primary.predictions, "source_dataset")
    roi_metrics = stratified_metrics(primary.predictions, "roi_type")
    high = frame[frame["ground_truth_quality"] == "high"].reset_index(drop=True)
    high_result = run_grouped_oof(high, bootstrap=False)
    quality_metrics = pd.DataFrame([{"scope": "high_only", **high_result.metrics}])
    heldout = source_heldout_sensitivity(frame)
    misclassified = primary.predictions[
        primary.predictions["true_label"] != primary.predictions["predicted_label"]
    ].copy()
    feature_importance = pd.concat(
        [
            primary.permutation_importance.assign(importance_type="permutation_oof"),
            primary.impurity_importance.assign(importance_type="impurity_mean_decrease"),
        ],
        ignore_index=True,
    )
    metrics = pd.DataFrame([{"scope": "primary_oof", **primary.metrics}])
    matrix = confusion_matrix(
        primary.predictions["true_label"], primary.predictions["predicted_label"], labels=CLASS_ORDER
    )
    confusion = pd.DataFrame(matrix, index=["true_alpha", "true_electron"], columns=["predicted_alpha", "predicted_electron"])
    source_association = categorical_association(class_source_contingency(frame))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    primary.predictions.to_csv(output / "pilot_oof_predictions.csv", index=False)
    metrics.to_csv(output / "pilot_metrics.csv", index=False)
    confusion.to_csv(output / "pilot_confusion_matrix.csv")
    source_metrics.to_csv(output / "pilot_source_metrics.csv", index=False)
    roi_metrics.to_csv(output / "pilot_roi_metrics.csv", index=False)
    quality_metrics.to_csv(output / "pilot_quality_sensitivity.csv", index=False)
    heldout.to_csv(output / "pilot_source_heldout.csv", index=False)
    feature_importance.to_csv(output / "pilot_feature_importance.csv", index=False)
    misclassified.to_csv(output / "pilot_misclassifications.csv", index=False)
    _plot_confusion(matrix, output / "plots" / "confusion_matrix.png")
    _plot_importance(primary.permutation_importance, output / "plots" / "feature_importance.png")

    metric = primary.metrics
    top_features = primary.permutation_importance.sort_values("importance_mean", ascending=False)
    summary = [
        "FIRST RANDOM FOREST PILOT — EXPLORATORY EVALUATION ONLY",
        "This is not final system performance and is not evidence of cross-domain generalization.",
        "",
        f"Dataset: Alpha={sum(frame['class']=='Alpha')}, Electron={sum(frame['class']=='Electron')}, Total={len(frame)}",
        f"Grouping key: connected dependency components sharing event_id OR source_image",
        f"Groups: {groups.nunique()}",
        f"CV: StratifiedGroupKFold, folds={len(primary.splits)}, shuffle=True, random_state=42",
        "Leakage detected: NO",
        f"RF parameters: {RF_PARAMETERS}",
        "Model inputs: " + ", ".join(MODEL_FEATURES),
        f"Source/class phi: {source_association['phi_cramers_v']}",
        "",
        "EXPLORATORY PILOT PERFORMANCE",
        f"Accuracy={metric['accuracy']}",
        f"Balanced accuracy={metric['balanced_accuracy']}",
        f"Macro F1={metric['macro_f1']}",
        f"MCC={metric['mcc']}",
        f"Confusion matrix rows Alpha/Electron: {matrix.tolist()}",
        "",
        "TOP OOF PERMUTATION IMPORTANCE",
    ]
    summary.extend(f"{row.feature}: {row.importance_mean}" for row in top_features.itertuples())
    summary.extend(
        [
            "",
            "LIMITATIONS",
            "n=56; moderate source/class and ROI/class association; pixel-scale effects; only two sources; temporary_module3_development segmentation; subgroup and held-out estimates are unstable.",
            "Extremely high performance would be treated as potentially optimistic rather than definitive.",
            "",
            "RANDOM FOREST PILOT TRAINED: YES",
            "PRODUCTION MODULE 3 MODIFIED: NO",
            "FINAL CLASSIFIER TRAINED: NO",
        ]
    )
    report = "\n".join(summary) + "\n"
    (output / "pilot_summary.txt").write_text(report, encoding="utf-8")
    return {
        "frame": frame,
        "groups": groups,
        "primary": primary,
        "source_metrics": source_metrics,
        "roi_metrics": roi_metrics,
        "quality_metrics": quality_metrics,
        "source_heldout": heldout,
        "feature_importance": feature_importance,
        "misclassified": misclassified,
        "source_association": source_association,
        "summary": report,
    }
