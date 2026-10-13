"""Read-only diagnostic for the three production Module-2 RF OOF errors.

This module consumes saved production masks, features, mappings, labels, and OOF
predictions.  It never fits a classifier or changes any upstream artifact.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cloudchamber-matplotlib"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.src.dataset_preparation.random_forest_pilot import MODEL_FEATURES

from .feature_extractor import extract_geometric_features


ERROR_IDS = ("A018", "A024", "E022")
WIDTH_FEATURES = ("mean_width_px", "median_width_px", "max_width_px")
OTHER_FEATURES = ("track_length_px", "mean_curvature", "max_curvature", "tortuosity")
EXPECTED_OOF = {
    "A018": ("Alpha", "Electron", 0.45, 0.55),
    "A024": ("Alpha", "Electron", 0.37, 0.63),
    "E022": ("Electron", "Alpha", 0.94, 0.06),
}

# These observations are based on the original/manual ROI and production-mask
# panels, never on the classifier output.
VISUAL_ASSESSMENTS: dict[str, dict[str, str]] = {
    "A018": {
        "mask_trustworthy": "NO",
        "visual_code": "B,E",
        "visual_finding": (
            "The selected component follows the chamber frame and a small top-edge artifact. "
            "It does not isolate the bright curved Alpha track inside the manual ROI."
        ),
        "primary_category": "A. SEGMENTATION_FAILURE",
        "action": "Exclude this production component from geometric interpretation and correct Module 2 upstream in a separate task; preserve the label and saved OOF result.",
    },
    "A024": {
        "mask_trustworthy": "YES",
        "visual_code": "A",
        "visual_finding": (
            "The compact production ROI and mask follow the intended single curved track, "
            "with no material unrelated component or visible truncation."
        ),
        "primary_category": "D. GENUINE_GEOMETRIC_OVERLAP",
        "action": "Retain the mask and label; treat this as class overlap and prioritize additional balanced data before changing Module 3.",
    },
    "E022": {
        "mask_trustworthy": "NO",
        "visual_code": "B,E",
        "visual_finding": (
            "The selected component follows the chamber frame/background rather than the thin "
            "horizontal Electron track and terminal blob inside the manual ROI."
        ),
        "primary_category": "A. SEGMENTATION_FAILURE",
        "action": "Exclude this production component from geometric interpretation and correct Module 2 upstream in a separate task; preserve the label and saved OOF result.",
    },
}


def _read_csv(path: str | Path, *, strings: bool = False) -> pd.DataFrame:
    kwargs = {"dtype": str, "keep_default_na": False} if strings else {}
    return pd.read_csv(path, **kwargs)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def load_diagnostic_frame(
    features_path: str | Path,
    predictions_path: str | Path,
    labels_path: str | Path,
    mapping_path: str | Path,
) -> pd.DataFrame:
    """Load and validate saved inputs without regenerating predictions/features."""
    features = _read_csv(features_path)
    predictions = _read_csv(predictions_path)
    labels = _read_csv(labels_path, strings=True).rename(columns={"track_id": "ground_truth_track_id"})
    mapping = _read_csv(mapping_path, strings=True)
    key = "ground_truth_track_id"
    for name, table in (("features", features), ("predictions", predictions), ("labels", labels), ("mapping", mapping)):
        if key not in table or table[key].duplicated().any():
            raise ValueError(f"{name} must contain unique {key} values")
    required_features = [key, "ground_truth_class", "source_dataset", "source_image", "event_id", "roi_type",
                         "module2_track_id", "module2_mask_file", "segmentation_source", "processing_status", *MODEL_FEATURES]
    missing = [column for column in required_features if column not in features]
    if missing:
        raise ValueError(f"Missing production feature columns: {missing}")
    if set(features["segmentation_source"]) != {"production_module2"} or set(features["processing_status"]) != {"success"}:
        raise ValueError("Diagnostic requires successful production_module2 feature rows only")
    prediction_columns = [key, "true_class", "predicted_class", "probability_alpha", "probability_electron", "fold"]
    label_columns = [key, "class", "crop_filename", "source_dataset", "source_image", "roi_type", "event_id",
                     "x", "y", "width", "height"]
    mapping_columns = [key, "match_method", "module2_track_id", "iou", "manual_roi_coverage", "candidate_coverage"]
    frame = features[required_features].merge(
        predictions[prediction_columns], on=key, validate="one_to_one"
    ).merge(
        labels[label_columns], on=key, suffixes=("", "_label"), validate="one_to_one"
    ).merge(
        mapping[mapping_columns], on=key, suffixes=("", "_mapping"), validate="one_to_one"
    )
    if not (frame["ground_truth_class"] == frame["true_class"]).all() or not (frame["class"] == frame["true_class"]).all():
        raise ValueError("Saved feature, label, and OOF classes disagree")
    for column in ("source_dataset", "source_image", "roi_type", "event_id"):
        if not (frame[column].astype(str) == frame[f"{column}_label"].astype(str)).all():
            raise ValueError(f"Feature and labels.csv provenance disagree for {column}")
    if not (frame["module2_track_id"].astype(str) == frame["module2_track_id_mapping"].astype(str)).all():
        raise ValueError("Feature and confirmed mapping Module-2 track IDs disagree")
    numeric = [*MODEL_FEATURES, "probability_alpha", "probability_electron", "x", "y", "width", "height",
               "iou", "manual_roi_coverage", "candidate_coverage"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Diagnostic inputs contain non-finite numeric values")
    if not np.allclose(frame[["probability_alpha", "probability_electron"]].sum(axis=1), 1.0):
        raise ValueError("Saved OOF probabilities do not sum to one")
    errors = frame.loc[frame["true_class"] != frame["predicted_class"], key].tolist()
    if set(errors) != set(ERROR_IDS):
        raise ValueError(f"Unexpected saved OOF error set: {errors}")
    for sample_id, (true_class, predicted_class, p_alpha, p_electron) in EXPECTED_OOF.items():
        row = frame.loc[frame[key] == sample_id].iloc[0]
        actual = (row.true_class, row.predicted_class, row.probability_alpha, row.probability_electron)
        expected = (true_class, predicted_class, p_alpha, p_electron)
        if actual[:2] != expected[:2] or not np.allclose(actual[2:], expected[2:]):
            raise ValueError(f"Saved OOF values changed for {sample_id}: {actual}")
    frame["correct"] = frame["true_class"] == frame["predicted_class"]
    return frame.sort_values(key).reset_index(drop=True)


def percentile(values: pd.Series, value: float) -> float:
    """Midrank empirical percentile, including ties at half weight."""
    numeric = pd.to_numeric(values, errors="raise")
    return float(100.0 * ((numeric < value).sum() + 0.5 * (numeric == value).sum()) / len(numeric))


def classify_percentiles(alpha_percentile: float, electron_percentile: float) -> str:
    """Deterministic descriptive class based only on the two empirical percentiles."""
    alpha_central = 10.0 <= alpha_percentile <= 90.0
    electron_central = 10.0 <= electron_percentile <= 90.0
    if alpha_central and electron_central:
        return "OVERLAP"
    if alpha_central:
        return "ALPHA_TYPICAL"
    if electron_central:
        return "ELECTRON_TYPICAL"
    same_extreme_tail = (
        (alpha_percentile <= 5.0 and electron_percentile <= 5.0)
        or (alpha_percentile >= 95.0 and electron_percentile >= 95.0)
    )
    if same_extreme_tail:
        return "EXTREME"
    alpha_distance = abs(alpha_percentile - 50.0)
    electron_distance = abs(electron_percentile - 50.0)
    if abs(alpha_distance - electron_distance) <= 10.0:
        return "OVERLAP"
    return "ALPHA_TYPICAL" if alpha_distance < electron_distance else "ELECTRON_TYPICAL"


def build_feature_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    alpha = frame[frame.true_class == "Alpha"]
    electron = frame[frame.true_class == "Electron"]
    for sample_id in ERROR_IDS:
        record = frame.loc[frame.ground_truth_track_id == sample_id].iloc[0]
        for feature in MODEL_FEATURES:
            value = float(record[feature])
            p_alpha = percentile(alpha[feature], value)
            p_electron = percentile(electron[feature], value)
            rows.append({
                "ground_truth_track_id": sample_id,
                "true_class": record.true_class,
                "predicted_class": record.predicted_class,
                "feature": feature,
                "value": value,
                "alpha_percentile_midrank": p_alpha,
                "electron_percentile_midrank": p_electron,
                "descriptive_class": classify_percentiles(p_alpha, p_electron),
            })
    return pd.DataFrame(rows)


def choose_comparators(frame: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Nearest correct examples after dataset-wide z-standardization."""
    correct = frame[frame.correct].copy()
    means = frame[MODEL_FEATURES].mean()
    scales = frame[MODEL_FEATURES].std(ddof=1).replace(0.0, 1.0)
    correct_z = (correct[MODEL_FEATURES] - means) / scales
    rows: list[dict[str, Any]] = []
    for sample_id in ERROR_IDS:
        error = frame.loc[frame.ground_truth_track_id == sample_id].iloc[0]
        error_z = (error[MODEL_FEATURES].astype(float) - means) / scales
        distances = np.sqrt(((correct_z - error_z) ** 2).sum(axis=1))
        for class_name in ("Alpha", "Electron"):
            candidates = correct[correct.true_class == class_name]
            selected = distances.loc[candidates.index].sort_values(kind="mergesort").head(limit)
            for rank, (index, distance) in enumerate(selected.items(), start=1):
                comparator = frame.loc[index]
                rows.append({
                    "misclassified_track_id": sample_id,
                    "comparator_class": class_name,
                    "rank": rank,
                    "comparator_track_id": comparator.ground_truth_track_id,
                    "standardized_euclidean_distance": float(distance),
                    "comparator_source_dataset": comparator.source_dataset,
                    "comparator_roi_type": comparator.roi_type,
                    **{feature: float(comparator[feature]) for feature in MODEL_FEATURES},
                })
    return pd.DataFrame(rows)


def _profile_assessment(frame: pd.DataFrame, row: pd.Series, features: Iterable[str]) -> tuple[str, float, float]:
    selected = list(features)
    scales = frame[selected].std(ddof=1).replace(0.0, 1.0)
    medians = frame.groupby("true_class")[selected].median()
    values = row[selected].astype(float)
    alpha_distance = float(np.sqrt((((values - medians.loc["Alpha"]) / scales) ** 2).mean()))
    electron_distance = float(np.sqrt((((values - medians.loc["Electron"]) / scales) ** 2).mean()))
    nearer = min(alpha_distance, electron_distance)
    farther = max(alpha_distance, electron_distance)
    if farther <= 1e-12 or nearer / farther > 0.67:
        label = "overlapping"
    elif alpha_distance < electron_distance:
        label = "strongly Alpha-like"
    else:
        label = "strongly Electron-like"
    return label, alpha_distance, electron_distance


def _summary_stats(frame: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows = []
    scopes: list[tuple[str, pd.DataFrame]] = []
    for class_name in ("Alpha", "Electron"):
        scopes.append((f"{class_name} overall", frame[frame.true_class == class_name]))
    for (source, class_name), group in frame.groupby(["source_dataset", "true_class"], sort=True):
        scopes.append((f"{class_name} source={source}", group))
    for scope, group in scopes:
        for feature in features:
            values = group[feature]
            rows.append({"scope": scope, "feature": feature, "n": len(values), "mean": values.mean(),
                         "median": values.median(), "q1": values.quantile(.25), "q3": values.quantile(.75),
                         "min": values.min(), "max": values.max()})
    return pd.DataFrame(rows)


def build_diagnostic_summary(frame: pd.DataFrame) -> pd.DataFrame:
    source_class_counts = frame.groupby(["source_dataset", "true_class"]).size()
    roi_class_counts = frame.groupby(["roi_type", "true_class"]).size()
    rows = []
    for sample_id in ERROR_IDS:
        record = frame.loc[frame.ground_truth_track_id == sample_id].iloc[0]
        width_label, width_alpha, width_electron = _profile_assessment(frame, record, WIDTH_FEATURES)
        other_label, other_alpha, other_electron = _profile_assessment(frame, record, OTHER_FEATURES)
        source_n = int(source_class_counts.loc[(record.source_dataset, record.true_class)])
        roi_n = int(roi_class_counts.loc[(record.roi_type, record.true_class)])
        visual = VISUAL_ASSESSMENTS[sample_id]
        rows.append({
            "ground_truth_track_id": sample_id,
            "true_class": record.true_class,
            "predicted_class": record.predicted_class,
            "probability_alpha": record.probability_alpha,
            "probability_electron": record.probability_electron,
            "source_dataset": record.source_dataset,
            "roi_type": record.roi_type,
            "event_id": record.event_id,
            "module2_track_id": record.module2_track_id,
            "source_class_n": source_n,
            "roi_class_n": roi_n,
            "sparse_subgroup": "YES" if min(source_n, roi_n) < 5 else "NO",
            "visual_segmentation_codes": visual["visual_code"],
            "visual_finding": visual["visual_finding"],
            "mask_trustworthy": visual["mask_trustworthy"],
            "width_profile": width_label,
            "width_distance_to_alpha_median": width_alpha,
            "width_distance_to_electron_median": width_electron,
            "length_curvature_tortuosity_profile": other_label,
            "other_distance_to_alpha_median": other_alpha,
            "other_distance_to_electron_median": other_electron,
            "conflicting_feature_evidence": "YES" if width_label != other_label else "NO",
            "primary_diagnostic_category": visual["primary_category"],
            "action": visual["action"],
        })
    return pd.DataFrame(rows)


def _decode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _module2_paths(mask_path: str | Path) -> tuple[Path, Path, Path]:
    mask = Path(mask_path)
    root = mask.parent.parent
    stem = mask.stem.removesuffix("_mask")
    return root / "images" / f"{stem}.png", mask, root / "overlays" / f"{stem}_overlay.png"


def _find_source(raw_root: Path, filename: str) -> Path:
    matches = sorted(raw_root.rglob(filename), key=lambda path: path.as_posix().casefold())
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source image named {filename}, found {len(matches)}")
    return matches[0]


def _draw_manual_roi(source: np.ndarray, row: pd.Series) -> np.ndarray:
    shown = source.copy()
    x, y, width, height = (int(round(float(row[name]))) for name in ("x", "y", "width", "height"))
    cv2.rectangle(shown, (x, y), (x + width, y + height), (255, 255, 0), 6)
    return shown


def _save_diagnostic_image(row: pd.Series, raw_root: Path, output: Path) -> None:
    class_dir = "alpha" if row.true_class == "Alpha" else "electron"
    crop = _decode(Path(row.labels_root) / class_dir / row.crop_filename, cv2.IMREAD_UNCHANGED)
    source = _decode(_find_source(raw_root, row.source_image))
    selected_path, mask_path, overlay_path = _module2_paths(row.module2_mask_file)
    selected = _decode(selected_path, cv2.IMREAD_UNCHANGED)
    mask = _decode(mask_path, cv2.IMREAD_GRAYSCALE)
    overlay = _decode(overlay_path, cv2.IMREAD_UNCHANGED)
    geometry = extract_geometric_features(mask)
    if geometry.get("processing_status") != "success":
        raise ValueError(f"Could not derive diagnostic skeleton for {row.ground_truth_track_id}")
    skeleton = np.asarray(geometry["skeleton"], dtype=bool)
    skeleton_image = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    skeleton_image[skeleton] = (255, 255, 255)
    panels = [crop, _draw_manual_roi(source, row), selected, mask, overlay, skeleton_image]
    titles = ["Ground-truth crop", "Source + manual ROI", "Selected Module-2 ROI", "Production mask",
              "Production overlay", "Skeleton from saved mask"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for axis, panel, title in zip(axes.flat, panels, titles):
        axis.imshow(_rgb(panel)); axis.set_title(title); axis.axis("off")
    visual = VISUAL_ASSESSMENTS[row.ground_truth_track_id]
    fig.suptitle(f"{row.ground_truth_track_id} | visual check (RF prediction not used) | trustworthy={visual['mask_trustworthy']}")
    fig.text(.5, .015, visual["visual_finding"], ha="center", va="bottom", wrap=True, fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150); plt.close(fig)


def _mask_bbox(image: np.ndarray, mask: np.ndarray, padding: int = 12) -> tuple[np.ndarray, np.ndarray]:
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return image, mask
    x, y, width, height = cv2.boundingRect(points)
    y0, y1 = max(0, y - padding), min(mask.shape[0], y + height + padding)
    x0, x1 = max(0, x - padding), min(mask.shape[1], x + width + padding)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _save_comparator_montage(frame: pd.DataFrame, comparators: pd.DataFrame, sample_id: str, output: Path) -> None:
    error = frame.loc[frame.ground_truth_track_id == sample_id].iloc[0]
    selected = comparators[comparators.misclassified_track_id == sample_id]
    ids = [sample_id, *selected.comparator_track_id.tolist()]
    labels = ["ERROR", *[f"{row.comparator_class} #{int(row.rank)}\nd={row.standardized_euclidean_distance:.3f}"
                         for row in selected.itertuples(index=False)]]
    fig, axes = plt.subplots(2, len(ids), figsize=(3.2 * len(ids), 7))
    for column, (track_id, label) in enumerate(zip(ids, labels)):
        row = frame.loc[frame.ground_truth_track_id == track_id].iloc[0]
        image_path, mask_path, _ = _module2_paths(row.module2_mask_file)
        image = _decode(image_path, cv2.IMREAD_UNCHANGED)
        mask = _decode(mask_path, cv2.IMREAD_GRAYSCALE)
        image, cropped_mask = _mask_bbox(image, mask)
        axes[0, column].imshow(_rgb(image)); axes[1, column].imshow(cropped_mask, cmap="gray")
        axes[0, column].set_title(f"{label}\n{track_id} ({row.true_class})", fontsize=9)
        axes[1, column].set_title("saved production mask", fontsize=8)
        axes[0, column].axis("off"); axes[1, column].axis("off")
    fig.suptitle(f"{sample_id}: nearest correctly classified seven-feature profiles (global z scores)")
    fig.tight_layout(rect=(0, 0, 1, .94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140); plt.close(fig)


def _format_stats(stats: pd.DataFrame, features: Iterable[str]) -> list[str]:
    lines = []
    for feature in features:
        lines.append(f"  {feature}:")
        for row in stats[stats.feature == feature].itertuples(index=False):
            lines.append(f"    {row.scope}: n={row.n}, mean={row.mean:.6f}, median={row.median:.6f}, "
                         f"IQR={row.q1:.6f}-{row.q3:.6f}, range={row.min:.6f}-{row.max:.6f}")
    return lines


def build_report(frame: pd.DataFrame, profiles: pd.DataFrame, comparators: pd.DataFrame,
                 diagnostic: pd.DataFrame) -> str:
    width_stats = _summary_stats(frame, WIDTH_FEATURES)
    other_stats = _summary_stats(frame, OTHER_FEATURES)
    lines = [
        "PRODUCTION MODULE-2 RANDOM FOREST OOF ERROR DIAGNOSTIC", "",
        "DIAGNOSTIC ONLY. Saved OOF probabilities and saved production features were read exactly as generated.",
        "No classifier was fitted, no prediction was regenerated, and no label/mapping/mask/feature was changed.", "",
        "METHOD", "Percentiles are empirical midranks: 100 * (count below + 0.5 * count equal) / class n.",
        "Descriptive feature class: central=10th-90th percentile; central in both=OVERLAP; central in one=that class; "
        "same <=5th or >=95th tail in both=EXTREME; otherwise the percentile nearer 50 wins unless distances differ by <=10 points (OVERLAP).",
        "Comparator distances are Euclidean distances after standardizing all seven features by the full production dataset mean and sample SD. "
        "Only correctly classified saved OOF rows are eligible; standardization is diagnostic and does not alter the RF.",
        "Profile-level class likeness uses RMS standardized distance to each class median; distance ratio >0.67 is overlapping, otherwise the nearer class is strong.",
        "Sparse subgroup means source-by-class or ROI-by-class n<5. Source and ROI are descriptive context, not causal evidence.",
        "Width is image geometry in pixels, not physical particle diameter; pixel-scale calibration is unavailable.", "",
        "VISUAL SEGMENTATION CODES", "A=correct intended track; B=significant unrelated foreground/background; C=multiple tracks merged; "
        "D=partial intended track; E=otherwise segmentation-suspect.", "",
    ]
    for sample_id in ERROR_IDS:
        row = diagnostic.loc[diagnostic.ground_truth_track_id == sample_id].iloc[0]
        lines.extend([
            f"===== {sample_id} =====",
            f"Context: source_dataset={row.source_dataset}; roi_type={row.roi_type}; event_id={row.event_id}; module2_track_id={row.module2_track_id}",
            f"OOF: true={row.true_class}; predicted={row.predicted_class}; P(Alpha)={row.probability_alpha:.2f}; P(Electron)={row.probability_electron:.2f}",
            f"Sparse subgroup: {row.sparse_subgroup} (source/class n={row.source_class_n}; ROI/class n={row.roi_class_n})",
            f"Visual assessment: {row.visual_segmentation_codes}. {row.visual_finding}",
            "Seven-feature profile (value; Alpha percentile; Electron percentile; description):",
        ])
        for item in profiles[profiles.ground_truth_track_id == sample_id].itertuples(index=False):
            lines.append(f"  {item.feature}: {item.value:.9f}; {item.alpha_percentile_midrank:.6f}; "
                         f"{item.electron_percentile_midrank:.6f}; {item.descriptive_class}")
        lines.extend([
            f"Width profile: {row.width_profile} (distance Alpha={row.width_distance_to_alpha_median:.6f}, Electron={row.width_distance_to_electron_median:.6f})",
            f"Length/curvature/tortuosity profile: {row.length_curvature_tortuosity_profile} "
            f"(distance Alpha={row.other_distance_to_alpha_median:.6f}, Electron={row.other_distance_to_electron_median:.6f})",
            f"Conflicting feature evidence: {row.conflicting_feature_evidence}",
            "Comparators:",
        ])
        for comparator in comparators[comparators.misclassified_track_id == sample_id].itertuples(index=False):
            lines.append(f"  {comparator.comparator_class} #{comparator.rank}: {comparator.comparator_track_id}; "
                         f"standardized distance={comparator.standardized_euclidean_distance:.9f}")
        if sample_id == "A018":
            answer = "The 0.55 Electron vote was borderline, but it describes chamber-frame/background geometry from an untrustworthy mask, not the intended Alpha track."
        elif sample_id == "A024":
            answer = "The trustworthy short, narrow curved-track geometry lies in substantial class overlap and is nearer Electron comparators; the 0.63 result is a genuine overlap error."
        else:
            answer = "The 0.94 Alpha result is caused by the bad Module-2 component: its very long, tortuous chamber-frame geometry is strongly Alpha-like, but it is not the Electron track."
        lines.extend([f"Specific answer: {answer}", f"MASK TRUSTWORTHY: {row.mask_trustworthy}",
                      f"PRIMARY CAUSE: {row.primary_diagnostic_category}", f"ACTION: {row.action}", ""])
    lines.extend(["WIDTH REFERENCE DISTRIBUTIONS", *_format_stats(width_stats, WIDTH_FEATURES), "",
                  "LENGTH / CURVATURE / TORTUOSITY REFERENCE DISTRIBUTIONS", *_format_stats(other_stats, OTHER_FEATURES), "",
                  "SUBGROUP CONTEXT",
                  "AWAN Electron is sparse (n=2). Rotated Electron is sparse (n=1). These counts do not establish causality.", "",
                  "FINAL INTERPRETATION", "C. MIXED FAILURE MODES",
                  "A018 and E022 are Module-2 segmentation failures; A024 is genuine geometric class overlap with a trustworthy mask.",
                  "IS ANOTHER MODULE 3 ALGORITHM CHANGE JUSTIFIED NOW? NO",
                  "IS MORE / BETTER-BALANCED DATA THE PRIORITY? YES", "",
                  "Module 1 modified: NO", "Module 2 modified: NO", "RANDOM FOREST RETRAINED: NO",
                  "FINAL CLASSIFIER TRAINED: NO", "PRODUCTION CLASSIFIER MODIFIED: NO", ""])
    return "\n".join(lines)


def run_production_error_diagnostic(
    features_path: str | Path,
    predictions_path: str | Path,
    labels_path: str | Path,
    mapping_path: str | Path,
    raw_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    frame = load_diagnostic_frame(features_path, predictions_path, labels_path, mapping_path)
    frame["labels_root"] = str(Path(labels_path).parent)
    profiles = build_feature_profiles(frame)
    comparators = choose_comparators(frame)
    diagnostic = build_diagnostic_summary(frame)
    output = Path(output_dir)
    for sample_id in ERROR_IDS:
        row = frame.loc[frame.ground_truth_track_id == sample_id].iloc[0]
        _save_diagnostic_image(row, Path(raw_root), output / "diagnostic_images" / f"{sample_id}_diagnostic.png")
        _save_comparator_montage(frame, comparators, sample_id,
                                 output / "comparator_montages" / f"{sample_id}_comparators.png")
    report = build_report(frame, profiles, comparators, diagnostic)
    _atomic_csv(profiles, output / "misclassified_feature_profiles.csv")
    _atomic_csv(comparators, output / "misclassified_comparators.csv")
    _atomic_csv(diagnostic, output / "production_error_diagnostic.csv")
    _atomic_text(report, output / "production_error_diagnostic.txt")
    return {"frame": frame, "profiles": profiles, "comparators": comparators,
            "diagnostic": diagnostic, "report": report, "output_dir": output}
