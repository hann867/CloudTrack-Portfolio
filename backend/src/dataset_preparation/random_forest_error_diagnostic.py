"""Read-only diagnostics for errors from the isolated Random Forest pilot."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cloudchamber-matplotlib"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.src.geometric_classification.feature_extractor import extract_geometric_features
from backend.src.geometric_classification.track_segmentation import segment_track

from .random_forest_pilot import CLASS_ORDER, MODEL_FEATURES


MISCLASSIFIED_IDS = ["A001", "A003", "A005", "A008", "A011", "A012", "A022", "E021"]
WIDTH_FEATURES = ["mean_width_px", "median_width_px", "max_width_px"]
RELATIONSHIP_FEATURES = ["mean_width_px", "track_length_px", "mean_curvature", "tortuosity"]
VISUAL_DIAGNOSES = {
    "A001": (
        "E. MULTIFACTORIAL",
        "Mask follows the visible track, but length is Electron-typical, width is in the low Alpha range, the OOF decision is borderline (0.52 Electron), and this is an AWAN sample.",
    ),
    "A003": (
        "E. MULTIFACTORIAL",
        "Mask follows the track; width is in the lower Alpha tail (19.6th percentile), with the error occurring in the higher-error AWAN and rotated subgroups.",
    ),
    "A005": (
        "E. MULTIFACTORIAL",
        "Mask follows the track, but width lies in the class-overlap region and at the 8.9th Alpha percentile; AWAN and rotated subgroup effects may also contribute.",
    ),
    "A008": (
        "E. MULTIFACTORIAL",
        "Mask follows the track; width is in the lower Alpha tail (16.1st percentile), with AWAN acquisition and rotated ROI association also present.",
    ),
    "A011": (
        "E. MULTIFACTORIAL",
        "Mask follows the track, but width is in the overlap region and at the 12.5th Alpha percentile; AWAN and rotated subgroup effects may also contribute.",
    ),
    "A012": (
        "E. MULTIFACTORIAL",
        "The long track is Alpha-like, but mean width is Electron-typical and at the 5.4th Alpha percentile; this conflicts with length and occurs in AWAN/rotated data.",
    ),
    "A022": (
        "B. SEGMENTATION-SUSPECT",
        "The reproduced temporary mask selects a narrow background structure rather than the prominent curved Alpha track, making the stored geometry Electron-like.",
    ),
    "E021": (
        "E. MULTIFACTORIAL",
        "Mask follows a bright curved segment, but mean width is at the 98.2nd Electron percentile and Alpha-typical; source-scale and partial-track selection may both contribute.",
    ),
}


def load_diagnostic_data(
    labels_path: str | Path,
    features_path: str | Path,
    predictions_path: str | Path,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    features = pd.read_csv(features_path)
    predictions = pd.read_csv(predictions_path)
    for name, table in (("labels", labels), ("features", features), ("predictions", predictions)):
        if table["track_id"].duplicated().any():
            raise ValueError(f"Duplicate track_id in {name}")
    if set(labels["track_id"]) != set(features["track_id"]) or set(labels["track_id"]) != set(predictions["track_id"]):
        raise ValueError("Labels, features, and OOF predictions must contain identical track IDs")
    label_fields = [
        "track_id", "class", "crop_filename", "source_dataset", "source_image",
        "roi_type", "ground_truth_quality", "event_id",
    ]
    feature_fields = ["track_id", "crop_width_px", "crop_height_px", "segmentation_source", *MODEL_FEATURES]
    prediction_fields = [
        "track_id", "true_label", "predicted_label", "probability_alpha", "probability_electron", "fold",
    ]
    frame = labels[label_fields].merge(features[feature_fields], on="track_id", validate="one_to_one")
    frame = frame.merge(predictions[prediction_fields], on="track_id", validate="one_to_one")
    if not (frame["class"] == frame["true_label"]).all():
        raise ValueError("OOF true labels do not agree with labels.csv")
    numeric = ["crop_width_px", "crop_height_px", "probability_alpha", "probability_electron", *MODEL_FEATURES]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    frame["crop_area_px"] = frame["crop_width_px"] * frame["crop_height_px"]
    frame["correct"] = frame["true_label"] == frame["predicted_label"]
    actual_errors = frame.loc[~frame["correct"], "track_id"].tolist()
    if set(actual_errors) != set(MISCLASSIFIED_IDS):
        raise ValueError(f"Unexpected OOF error set: {actual_errors}")
    return frame.sort_values("track_id").reset_index(drop=True)


def _percentile(values: pd.Series, value: float) -> float:
    return float(100.0 * ((values < value).sum() + 0.5 * (values == value).sum()) / len(values))


def _robust_distance(values: pd.Series, value: float) -> float:
    scale = float(values.quantile(0.75) - values.quantile(0.25))
    if scale <= 1e-12:
        scale = float(values.std(ddof=1))
    return abs(value - float(values.median())) / max(scale, 1e-12)


def feature_typicality(frame: pd.DataFrame, class_name: str, feature: str, value: float) -> str:
    own = frame.loc[frame["true_label"] == class_name, feature]
    opposite_name = "Electron" if class_name == "Alpha" else "Alpha"
    opposite = frame.loc[frame["true_label"] == opposite_name, feature]
    own_iqr = (float(own.quantile(0.25)), float(own.quantile(0.75)))
    opposite_iqr = (float(opposite.quantile(0.25)), float(opposite.quantile(0.75)))
    if own_iqr[0] <= value <= own_iqr[1] and opposite_iqr[0] <= value <= opposite_iqr[1]:
        return "overlapping region"
    own_distance = _robust_distance(own, value)
    opposite_distance = _robust_distance(opposite, value)
    if abs(own_distance - opposite_distance) <= 0.15:
        return "overlapping region"
    return class_name if own_distance < opposite_distance else opposite_name


def build_misclassified_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in frame.loc[~frame["correct"]].itertuples(index=False):
        row = record._asdict()
        for feature in MODEL_FEATURES:
            true_values = frame.loc[frame["true_label"] == record.true_label, feature]
            row[f"{feature}_true_class_percentile"] = _percentile(true_values, float(getattr(record, feature)))
            row[f"{feature}_typical_of"] = feature_typicality(
                frame, record.true_label, feature, float(getattr(record, feature))
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)


def width_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, class_name), group in frame.groupby(["source_dataset", "true_label"], sort=True):
        for feature in WIDTH_FEATURES:
            values = group[feature]
            rows.append(
                {
                    "source_dataset": source,
                    "class": class_name,
                    "feature": feature,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "sd": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def error_summary(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(field, sort=True):
        errors = group.loc[~group["correct"], "track_id"].tolist()
        rows.append(
            {
                field: value,
                "n": len(group),
                "errors": len(errors),
                "error_rate": len(errors) / len(group),
                "track_ids": ";".join(errors),
            }
        )
    return pd.DataFrame(rows)


def choose_comparators(frame: pd.DataFrame) -> pd.DataFrame:
    correct = frame[frame["correct"]].copy()
    scales = frame[MODEL_FEATURES].std(ddof=1).replace(0, 1)
    rows = []
    for error in frame[~frame["correct"]].itertuples(index=False):
        for role, class_name in (("same_true_class", error.true_label), ("opposite_class", error.predicted_label)):
            candidates = correct[correct["true_label"] == class_name].copy()
            same_source = candidates[candidates["source_dataset"] == error.source_dataset]
            if not same_source.empty:
                candidates = same_source
                source_matched = True
            else:
                source_matched = False
            error_values = pd.Series({feature: float(getattr(error, feature)) for feature in MODEL_FEATURES})
            distances = np.sqrt((((candidates[MODEL_FEATURES] - error_values) / scales) ** 2).sum(axis=1))
            comparator = candidates.loc[distances.idxmin()]
            rows.append(
                {
                    "misclassified_track_id": error.track_id,
                    "comparator_role": role,
                    "comparator_track_id": comparator["track_id"],
                    "comparator_class": comparator["true_label"],
                    "source_dataset": comparator["source_dataset"],
                    "source_matched": source_matched,
                    "feature_distance": float(distances.loc[comparator.name]),
                    **{feature: comparator[feature] for feature in MODEL_FEATURES},
                }
            )
    return pd.DataFrame(rows)


def _read_and_segment(record: pd.Series, dataset_root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = dataset_root / str(record["true_label"]).lower() / str(record["crop_filename"])
    original = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if original is None:
        raise FileNotFoundError(path)
    segmentation_input = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR) if original.ndim == 3 and original.shape[2] == 4 else original
    segmentation = segment_track(segmentation_input)
    if segmentation["processing_status"] != "success":
        raise ValueError(f"Segmentation failed for {record['track_id']}: {segmentation['error_message']}")
    mask = np.asarray(segmentation["mask"])
    geometry = extract_geometric_features(mask)
    if geometry["processing_status"] != "success":
        raise ValueError(f"Geometry failed for {record['track_id']}: {geometry['error_message']}")
    return original, mask, geometry


def _rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _segmentation_indicators(mask: np.ndarray, geometry: dict[str, Any]) -> dict[str, Any]:
    binary = mask > 0
    border_pixels = int(binary[0].sum() + binary[-1].sum() + binary[:, 0].sum() + binary[:, -1].sum())
    path = np.asarray(geometry["principal_path"], dtype=int)
    height, width = binary.shape
    path_near_border = bool(
        np.any((path[:, 0] <= 1) | (path[:, 0] >= height - 2) | (path[:, 1] <= 1) | (path[:, 1] >= width - 2))
    )
    return {
        "mask_area_fraction": float(binary.mean()),
        "mask_border_pixels": border_pixels,
        "path_near_border": path_near_border,
    }


def _save_sample_diagnostic(
    record: pd.Series,
    original: np.ndarray,
    mask: np.ndarray,
    geometry: dict[str, Any],
    path: Path,
) -> None:
    binary = mask > 0
    skeleton = np.asarray(geometry["skeleton"], dtype=bool)
    principal = np.asarray(geometry["principal_path"], dtype=int)
    overlay = _rgb(original).copy()
    overlay[skeleton] = [0, 255, 0]
    overlay[principal[:, 0], principal[:, 1]] = [255, 40, 40]
    distances = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) * 2.0
    width_map = np.where(binary, distances, np.nan)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(_rgb(original)); axes[0].set_title("Original crop")
    axes[1].imshow(binary, cmap="gray"); axes[1].set_title("Temporary mask")
    axes[2].imshow(overlay); axes[2].set_title("Skeleton / principal path")
    width_image = axes[3].imshow(width_map, cmap="viridis")
    axes[3].plot(principal[:, 1], principal[:, 0], color="white", linewidth=0.7)
    axes[3].set_title("Local width (px)")
    fig.colorbar(width_image, ax=axes[3], fraction=0.046)
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"{record['track_id']} | true {record['true_label']} -> predicted {record['predicted_label']} | "
        f"{record['source_dataset']} | {record['roi_type']}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_montage(frame: pd.DataFrame, segmented: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    errors = frame[~frame["correct"]].sort_values("track_id")
    fig, axes = plt.subplots(4, 4, figsize=(15, 16))
    for row_index, record in enumerate(errors.itertuples(index=False)):
        row, pair = divmod(row_index, 2)
        original_axis, mask_axis = axes[row, pair * 2], axes[row, pair * 2 + 1]
        original, mask = segmented[record.track_id]
        original_axis.imshow(_rgb(original))
        original_axis.set_title(
            f"{record.track_id}: {record.true_label}->{record.predicted_label}\n"
            f"{record.source_dataset}, {record.roi_type}\n"
            f"width={record.mean_width_px:.2f}, length={record.track_length_px:.1f}"
        )
        mask_axis.imshow(mask > 0, cmap="gray")
        mask_axis.set_title(f"curvature={record.mean_curvature:.4f}\ntortuosity={record.tortuosity:.3f}")
        original_axis.axis("off"); mask_axis.axis("off")
    fig.suptitle("Random Forest pilot misclassifications: original crops and reproduced masks", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_comparator_montage(
    frame: pd.DataFrame,
    comparators: pd.DataFrame,
    segmented: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
) -> None:
    errors = frame[~frame["correct"]].sort_values("track_id")
    fig, axes = plt.subplots(len(errors), 6, figsize=(18, 24))
    for row_index, error in enumerate(errors.itertuples(index=False)):
        pairs = [(error.track_id, "ERROR")]
        selected = comparators[comparators.misclassified_track_id == error.track_id]
        for role in ("same_true_class", "opposite_class"):
            track_id = selected.loc[selected.comparator_role == role, "comparator_track_id"].iloc[0]
            pairs.append((track_id, "SAME CLASS" if role == "same_true_class" else "OPPOSITE CLASS"))
        for pair_index, (track_id, role) in enumerate(pairs):
            original, mask = segmented[track_id]
            record = frame.loc[frame.track_id == track_id].iloc[0]
            original_axis, mask_axis = axes[row_index, pair_index * 2], axes[row_index, pair_index * 2 + 1]
            original_axis.imshow(_rgb(original)); mask_axis.imshow(mask > 0, cmap="gray")
            original_axis.set_title(
                f"{role}: {track_id} ({record.true_label})\n"
                f"width={record.mean_width_px:.2f}, length={record.track_length_px:.1f}"
            )
            mask_axis.set_title(f"mask\ncurv={record.mean_curvature:.4f}, tort={record.tortuosity:.3f}")
            original_axis.axis("off"); mask_axis.axis("off")
    fig.suptitle("Misclassified samples with source-matched correctly classified comparators", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_width_groups(frame: pd.DataFrame, path: Path) -> None:
    groups = [("AWAN", "Alpha"), ("AWAN", "Electron"), ("Lecturer", "Alpha"), ("Lecturer", "Electron")]
    values = [frame.loc[(frame.source_dataset == source) & (frame.true_label == label), "mean_width_px"] for source, label in groups]
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.boxplot(values, tick_labels=[f"{source}\n{label}" for source, label in groups], showmeans=True)
    axis.set_ylabel("mean_width_px")
    axis.set_title("Width by true class and acquisition source")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _plot_relationship(frame: pd.DataFrame, feature: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 5.5))
    for class_name, color in (("Alpha", "#3366cc"), ("Electron", "#dd6633")):
        subset = frame[(frame.true_label == class_name) & frame.correct]
        axis.scatter(subset[feature], subset["probability_alpha"], label=f"{class_name}, correct", color=color, alpha=0.7)
    errors = frame[~frame["correct"]]
    axis.scatter(errors[feature], errors["probability_alpha"], marker="X", s=100, color="black", label="Misclassified")
    for record in errors.itertuples(index=False):
        axis.annotate(record.track_id, (getattr(record, feature), record.probability_alpha), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set_xlabel(feature); axis.set_ylabel("Existing OOF probability_alpha")
    axis.set_title(f"{feature} versus existing OOF probability")
    axis.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _diagnose_errors(frame: pd.DataFrame, diagnostic: pd.DataFrame) -> pd.DataFrame:
    result = diagnostic.copy()
    categories, reasons = [], []
    for row in result.itertuples(index=False):
        opposite_count = sum(getattr(row, f"{feature}_typical_of") == row.predicted_label for feature in MODEL_FEATURES)
        segmentation_flag = bool(row.path_near_border or row.mask_border_pixels > 0)
        source_error_rate = float(frame.loc[frame.source_dataset == row.source_dataset, "correct"].eq(False).mean())
        roi_error_rate = float(frame.loc[frame.roi_type == row.roi_type, "correct"].eq(False).mean())
        width_opposite = row.mean_width_px_typical_of == row.predicted_label
        factors = []
        if opposite_count >= 3 or width_opposite:
            factors.append("geometry overlaps the predicted class")
        if segmentation_flag:
            factors.append("reproduced mask/path reaches the crop boundary")
        if source_error_rate >= 0.25:
            factors.append(f"source subgroup has elevated error rate ({source_error_rate:.1%})")
        if roi_error_rate >= 0.20:
            factors.append(f"ROI subgroup has elevated error rate ({roi_error_rate:.1%})")
        if len(factors) >= 2:
            category = "E. MULTIFACTORIAL"
        elif segmentation_flag:
            category = "B. SEGMENTATION-SUSPECT"
        elif opposite_count >= 3 or width_opposite:
            category = "A. CLASS-OVERLAP"
        elif source_error_rate >= 0.25:
            category = "C. SCALE-SUSPECT"
        elif roi_error_rate >= 0.20:
            category = "D. ROI-SUSPECT"
        else:
            category = "F. NO CLEAR CAUSE"
        if row.track_id in VISUAL_DIAGNOSES:
            category, reason = VISUAL_DIAGNOSES[row.track_id]
        else:
            reason = "; ".join(factors) if factors else "No single measured diagnostic factor stands out"
        categories.append(category)
        reasons.append(reason)
    result["diagnostic_category"] = categories
    result["diagnostic_reason"] = reasons
    return result


def _correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in ("crop_width_px", "crop_height_px", "crop_area_px"):
        for outcome in ("mean_width_px", "track_length_px", "probability_alpha"):
            rows.append(
                {
                    "dimension": dimension,
                    "outcome": outcome,
                    "pearson_r": float(frame[dimension].corr(frame[outcome], method="pearson")),
                    "spearman_rho": float(frame[dimension].corr(frame[outcome], method="spearman")),
                }
            )
    return pd.DataFrame(rows)


def _build_report(
    frame: pd.DataFrame,
    diagnostic: pd.DataFrame,
    widths: pd.DataFrame,
    by_source: pd.DataFrame,
    by_roi: pd.DataFrame,
    correlations: pd.DataFrame,
) -> str:
    class_width = frame.groupby("true_label")["mean_width_px"].agg(["mean", "median"])
    lines = [
        "ISOLATED RANDOM FOREST PILOT ERROR DIAGNOSTIC",
        "No classifier was fit or modified. Existing OOF probabilities were used unchanged.",
        "Segmentation visualizations reproduce temporary_module3_development for diagnosis only.",
        "",
        "MISCLASSIFICATIONS",
        f"Total: {(~frame['correct']).sum()}",
        f"Alpha -> Electron: {((frame.true_label == 'Alpha') & ~frame.correct).sum()}",
        f"Electron -> Alpha: {((frame.true_label == 'Electron') & ~frame.correct).sum()}",
        "",
        "ERRORS BY SOURCE",
    ]
    for row in by_source.itertuples(index=False):
        lines.append(f"{row.source_dataset}: n={row.n}, errors={row.errors}, rate={row.error_rate:.4f}, IDs={row.track_ids}")
    lines.extend(["", "ERRORS BY ROI"])
    for row in by_roi.itertuples(index=False):
        lines.append(f"{row.roi_type}: n={row.n}, errors={row.errors}, rate={row.error_rate:.4f}, IDs={row.track_ids}")
    lines.extend(
        [
            "",
            "WIDTH FINDINGS",
            f"Alpha mean/median mean_width_px: {class_width.loc['Alpha', 'mean']:.4f} / {class_width.loc['Alpha', 'median']:.4f}",
            f"Electron mean/median mean_width_px: {class_width.loc['Electron', 'mean']:.4f} / {class_width.loc['Electron', 'median']:.4f}",
        ]
    )
    for source in ("AWAN", "Lecturer"):
        for class_name in CLASS_ORDER:
            row = widths[(widths.source_dataset == source) & (widths["class"] == class_name) & (widths.feature == "mean_width_px")].iloc[0]
            lines.append(f"{source} {class_name}: n={row.n}, mean={row['mean']:.4f}, median={row['median']:.4f}, SD={row.sd:.4f}, min={row['min']:.4f}, max={row['max']:.4f}")
    lines.extend(["", "CROP-SCALE CORRELATIONS (all 56; exploratory)"])
    for row in correlations.itertuples(index=False):
        lines.append(f"{row.dimension} vs {row.outcome}: Pearson r={row.pearson_r:.4f}, Spearman rho={row.spearman_rho:.4f}")
    lines.extend(["", "CROP-SCALE GROUP COMPARISONS (mean width x height, median area)"])
    for field in ("true_label", "source_dataset", "correct"):
        for value, group in frame.groupby(field, sort=True):
            lines.append(
                f"{field}={value}: n={len(group)}, mean crop={group.crop_width_px.mean():.1f}x{group.crop_height_px.mean():.1f}, "
                f"median area={group.crop_area_px.median():.1f} px2"
            )
    lines.extend(["", "INDIVIDUAL ERROR DIAGNOSIS"])
    for row in diagnostic.itertuples(index=False):
        lines.append(f"{row.track_id}: {row.diagnostic_category} - {row.diagnostic_reason}")
    lines.extend(
        [
            "",
            "INTERPRETIVE CAUTION",
            "Categories are diagnostic flags, not corrected labels. Boundary contact is suggestive, not proof, of truncation or ROI effects.",
            "Pixel widths are uncalibrated and may mix particle morphology with acquisition/crop scale. Small subgroup counts preclude definitive inference.",
            "Width appears BOTH class-related and source-related: the class separation is large, while within each class AWAN and Lecturer medians also shift.",
            "Main likely failure mode: a combination of class-overlap in pixel width, source/scale effects, and ROI association, plus a clear temporary-segmentation failure for A022.",
            "Recommended next actions (not implemented): improve/integrate segmentation; investigate physical scale normalization; collect more cross-source data; review feature design; continue development with these safeguards.",
            "",
            "RANDOM FOREST RETRAINED: NO",
            "PRODUCTION MODULE 3 MODIFIED: NO",
        ]
    )
    return "\n".join(lines) + "\n"


def run_error_diagnostic(
    labels_path: str | Path,
    features_path: str | Path,
    predictions_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    frame = load_diagnostic_data(labels_path, features_path, predictions_path)
    errors = build_misclassified_table(frame)
    widths = width_summary(frame)
    by_source = error_summary(frame, "source_dataset")
    by_roi = error_summary(frame, "roi_type")
    comparators = choose_comparators(frame)
    correlations = _correlations(frame)

    output = Path(output_dir)
    plots = output / "plots"
    samples = output / "diagnostic_samples"
    plots.mkdir(parents=True, exist_ok=True)
    samples.mkdir(parents=True, exist_ok=True)
    segmented: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    indicators: dict[str, dict[str, Any]] = {}
    recomputed: dict[str, dict[str, float]] = {}
    for _, record in frame[~frame["correct"]].iterrows():
        original, mask, geometry = _read_and_segment(record, Path(dataset_root))
        segmented[record["track_id"]] = (original, mask)
        indicators[record["track_id"]] = _segmentation_indicators(mask, geometry)
        recomputed[record["track_id"]] = {feature: float(geometry[feature]) for feature in MODEL_FEATURES}
        _save_sample_diagnostic(record, original, mask, geometry, samples / f"{record['track_id']}_diagnostic.png")
    comparator_ids = comparators["comparator_track_id"].drop_duplicates().tolist()
    for track_id in comparator_ids:
        if track_id in segmented:
            continue
        record = frame.loc[frame.track_id == track_id].iloc[0]
        original, mask, _ = _read_and_segment(record, Path(dataset_root))
        segmented[track_id] = (original, mask)
    for name in ("mask_area_fraction", "mask_border_pixels", "path_near_border"):
        errors[name] = errors["track_id"].map(lambda track_id: indicators[track_id][name])
    errors["max_recomputed_feature_abs_difference"] = errors.apply(
        lambda row: max(abs(float(row[feature]) - recomputed[row["track_id"]][feature]) for feature in MODEL_FEATURES), axis=1
    )
    errors = _diagnose_errors(frame, errors)

    crop_scale = frame[
        [
            "track_id", "true_label", "predicted_label", "source_dataset", "roi_type", "correct",
            "crop_width_px", "crop_height_px", "crop_area_px", "mean_width_px", "track_length_px", "probability_alpha",
        ]
    ].copy()
    errors.to_csv(output / "misclassified_diagnostic.csv", index=False)
    widths.to_csv(output / "width_by_class_source.csv", index=False)
    by_source.to_csv(output / "error_by_source.csv", index=False)
    by_roi.to_csv(output / "error_by_roi.csv", index=False)
    crop_scale.to_csv(output / "crop_scale_diagnostic.csv", index=False)
    comparators.to_csv(output / "comparator_samples.csv", index=False)
    _save_montage(frame, segmented, plots / "misclassified_montage.png")
    _save_comparator_montage(frame, comparators, segmented, plots / "comparator_montage.png")
    _plot_width_groups(frame, plots / "width_by_class_source.png")
    for feature, filename in (
        ("mean_width_px", "width_vs_probability.png"),
        ("track_length_px", "length_vs_probability.png"),
        ("mean_curvature", "curvature_vs_probability.png"),
        ("tortuosity", "tortuosity_vs_probability.png"),
    ):
        _plot_relationship(frame, feature, plots / filename)
    report = _build_report(frame, errors, widths, by_source, by_roi, correlations)
    (output / "diagnostic_report.txt").write_text(report, encoding="utf-8")
    return {
        "frame": frame,
        "misclassified": errors,
        "widths": widths,
        "by_source": by_source,
        "by_roi": by_roi,
        "crop_scale": crop_scale,
        "comparators": comparators,
        "correlations": correlations,
        "report": report,
    }
