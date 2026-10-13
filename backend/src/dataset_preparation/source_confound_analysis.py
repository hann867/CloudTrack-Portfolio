"""Diagnostic analysis of class/source confounding using existing geometry only."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cloudchamber-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "track_length_px",
    "mean_width_px",
    "median_width_px",
    "max_width_px",
    "mean_curvature",
    "max_curvature",
    "tortuosity",
]
SOURCE_ORDER = ["AWAN", "Lecturer", "Other"]
CLASS_ORDER = ["Alpha", "Electron"]
ROI_ORDER = ["axis_aligned", "rotated", "Other"]
SUMMARY_STATISTICS = ["count", "mean", "median", "standard_deviation", "min", "max"]


def _source_group(value: object) -> str:
    text = str(value).strip()
    return text if text in {"AWAN", "Lecturer"} else "Other"


def _roi_group(value: object) -> str:
    text = str(value).strip()
    return text if text in {"axis_aligned", "rotated"} else "Other"


def class_source_contingency(labels: pd.DataFrame) -> pd.DataFrame:
    """Build counts plus row/column percentages for class by source."""

    prepared = labels.copy()
    prepared["source_group"] = prepared["source_dataset"].map(_source_group)
    counts = pd.crosstab(prepared["source_group"], prepared["class"]).reindex(index=SOURCE_ORDER, columns=CLASS_ORDER, fill_value=0)
    rows: list[dict[str, Any]] = []
    column_totals = counts.sum(axis=0)
    for source in SOURCE_ORDER:
        alpha, electron = int(counts.loc[source, "Alpha"]), int(counts.loc[source, "Electron"])
        total = alpha + electron
        rows.append(
            {
                "source_dataset": source,
                "alpha_count": alpha,
                "electron_count": electron,
                "total": total,
                "alpha_row_pct": 100.0 * alpha / total if total else 0.0,
                "electron_row_pct": 100.0 * electron / total if total else 0.0,
                "alpha_column_pct": 100.0 * alpha / column_totals["Alpha"] if column_totals["Alpha"] else 0.0,
                "electron_column_pct": 100.0 * electron / column_totals["Electron"] if column_totals["Electron"] else 0.0,
            }
        )
    return pd.DataFrame(rows)


def categorical_association(contingency: pd.DataFrame) -> dict[str, float | None]:
    """Calculate Fisher's exact p and absolute phi for the AWAN/Lecturer 2x2 table."""

    indexed = contingency.set_index("source_dataset")
    table = np.asarray(
        [
            [indexed.loc["AWAN", "alpha_count"], indexed.loc["AWAN", "electron_count"]],
            [indexed.loc["Lecturer", "alpha_count"], indexed.loc["Lecturer", "electron_count"]],
        ],
        dtype=float,
    )
    if table.sum() == 0 or np.any(table.sum(axis=0) == 0) or np.any(table.sum(axis=1) == 0):
        return {"fisher_odds_ratio": None, "fisher_p_value": None, "phi_cramers_v": None}
    odds_ratio, p_value = fisher_exact(table.astype(int), alternative="two-sided")
    a, b, c, d = table.ravel()
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = abs((a * d - b * c) / denominator) if denominator else None
    return {
        "fisher_odds_ratio": float(odds_ratio),
        "fisher_p_value": float(p_value),
        "phi_cramers_v": float(phi) if phi is not None else None,
    }


def class_roi_contingency(labels: pd.DataFrame) -> pd.DataFrame:
    prepared = labels.copy()
    prepared["roi_group"] = prepared["roi_type"].map(_roi_group)
    counts = pd.crosstab(prepared["roi_group"], prepared["class"]).reindex(index=ROI_ORDER, columns=CLASS_ORDER, fill_value=0)
    rows = []
    column_totals = counts.sum(axis=0)
    for roi_type in ROI_ORDER:
        alpha, electron = int(counts.loc[roi_type, "Alpha"]), int(counts.loc[roi_type, "Electron"])
        total = alpha + electron
        rows.append(
            {
                "roi_type": roi_type,
                "alpha_count": alpha,
                "electron_count": electron,
                "total": total,
                "alpha_row_pct": 100.0 * alpha / total if total else 0.0,
                "electron_row_pct": 100.0 * electron / total if total else 0.0,
                "alpha_column_pct": 100.0 * alpha / column_totals["Alpha"] if column_totals["Alpha"] else 0.0,
                "electron_column_pct": 100.0 * electron / column_totals["Electron"] if column_totals["Electron"] else 0.0,
            }
        )
    return pd.DataFrame(rows)


def descriptive_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_values, group in frame.groupby(group_columns, dropna=False, sort=True):
        values_tuple = group_values if isinstance(group_values, tuple) else (group_values,)
        group_fields = dict(zip(group_columns, values_tuple))
        for feature in FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            rows.append(
                {
                    **group_fields,
                    "feature": feature,
                    "count": int(values.count()),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "median": float(values.median()) if len(values) else np.nan,
                    "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else np.nan),
                    "min": float(values.min()) if len(values) else np.nan,
                    "max": float(values.max()) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=group_columns + ["feature"] + SUMMARY_STATISTICS)


def identify_cross_source_samples(frame: pd.DataFrame) -> pd.DataFrame:
    """Return Lecturer Alpha and AWAN Electron records with unchanged feature values."""

    cross = frame[
        ((frame["class"] == "Alpha") & (frame["source_dataset"] == "Lecturer"))
        | ((frame["class"] == "Electron") & (frame["source_dataset"] == "AWAN"))
    ].copy()
    columns = ["track_id", "class", "source_dataset", "crop_filename"] + FEATURES
    return cross.loc[:, columns].sort_values("track_id").reset_index(drop=True)


def _add_exploratory_centroid_distances(frame: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    if cross.empty:
        return cross
    numeric = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    usable = numeric.notna().all(axis=1)
    if usable.sum() < 3:
        return cross.assign(
            distance_to_awan_alpha_centroid=np.nan,
            distance_to_lecturer_electron_centroid=np.nan,
            closer_primary_group="insufficient_data",
        )
    scaler = StandardScaler().fit(numeric.loc[usable])
    standardized = pd.DataFrame(scaler.transform(numeric.loc[usable]), index=frame.index[usable], columns=FEATURES)
    awan_alpha = standardized[(frame.loc[usable, "class"] == "Alpha") & (frame.loc[usable, "source_dataset"] == "AWAN")]
    lecturer_electron = standardized[(frame.loc[usable, "class"] == "Electron") & (frame.loc[usable, "source_dataset"] == "Lecturer")]
    if awan_alpha.empty or lecturer_electron.empty:
        return cross.assign(
            distance_to_awan_alpha_centroid=np.nan,
            distance_to_lecturer_electron_centroid=np.nan,
            closer_primary_group="insufficient_data",
        )
    alpha_centroid = awan_alpha.mean(axis=0).to_numpy()
    electron_centroid = lecturer_electron.mean(axis=0).to_numpy()
    enriched = cross.copy()
    alpha_distances, electron_distances, closer = [], [], []
    for _, row in cross.iterrows():
        point = scaler.transform(pd.DataFrame([[row[feature] for feature in FEATURES]], columns=FEATURES))[0]
        alpha_distance = float(np.linalg.norm(point - alpha_centroid))
        electron_distance = float(np.linalg.norm(point - electron_centroid))
        alpha_distances.append(alpha_distance)
        electron_distances.append(electron_distance)
        closer.append("AWAN Alpha" if alpha_distance < electron_distance else "Lecturer Electron")
    enriched["distance_to_awan_alpha_centroid"] = alpha_distances
    enriched["distance_to_lecturer_electron_centroid"] = electron_distances
    enriched["closer_primary_group"] = closer
    return enriched


def _plot_grouped_feature(frame: pd.DataFrame, feature: str, target: Path, title: str) -> None:
    groups = [(class_name, source) for class_name in CLASS_ORDER for source in ("AWAN", "Lecturer")]
    values = [pd.to_numeric(frame[(frame["class"] == class_name) & (frame["source_dataset"] == source)][feature], errors="coerce").dropna().to_numpy() for class_name, source in groups]
    fig, axis = plt.subplots(figsize=(9, 5.5))
    nonempty_positions = [index + 1 for index, group in enumerate(values) if len(group)]
    nonempty_values = [group for group in values if len(group)]
    if nonempty_values:
        axis.boxplot(nonempty_values, positions=nonempty_positions, widths=0.5, showfliers=False)
    colors = {"AWAN": "#2f6fbb", "Lecturer": "#e28743"}
    for position, ((_, source), group) in enumerate(zip(groups, values), start=1):
        if len(group):
            offsets = np.linspace(-0.12, 0.12, len(group)) if len(group) > 1 else np.zeros(1)
            axis.scatter(position + offsets, group, color=colors[source], s=34, alpha=0.8, label=source if position <= 2 else None)
    axis.set_xticks(range(1, len(groups) + 1), [f"{class_name}\n{source}" for class_name, source in groups])
    axis.set_ylabel(feature)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color, label=source) for source, color in colors.items()]
    axis.legend(handles=handles, title="Source")
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)


def _pca_plots(frame: pd.DataFrame, plots_dir: Path) -> tuple[float, float] | None:
    numeric = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    usable = numeric.notna().all(axis=1)
    if usable.sum() < 3:
        return None
    standardized = StandardScaler().fit_transform(numeric.loc[usable])
    pca = PCA(n_components=2)
    components = pca.fit_transform(standardized)
    metadata = frame.loc[usable].reset_index(drop=True)
    for field, categories, colors, filename in (
        ("class", CLASS_ORDER, {"Alpha": "#7b3294", "Electron": "#008837"}, "pca_by_class.png"),
        ("source_dataset", ["AWAN", "Lecturer"], {"AWAN": "#2f6fbb", "Lecturer": "#e28743"}, "pca_by_source.png"),
    ):
        fig, axis = plt.subplots(figsize=(7, 5.5))
        for category in categories:
            selected = metadata[field] == category
            axis.scatter(components[selected, 0], components[selected, 1], label=category, color=colors[category], s=42, alpha=0.8)
        axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
        axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
        axis.set_title(f"Exploratory PCA colored by {field}")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plots_dir / filename, dpi=160)
        plt.close(fig)
    return float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])


def _format_ids(frame: pd.DataFrame, class_name: str, source: str) -> str:
    ids = frame[(frame["class"] == class_name) & (frame["source_dataset"] == source)]["track_id"].tolist()
    return ", ".join(ids) if ids else "none"


def analyse_source_confound(
    labels_path: str | Path,
    features_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    create_plots: bool = True,
) -> dict[str, Any]:
    """Run diagnostic-only confound analysis without mutating either input CSV."""

    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    features = pd.read_csv(features_path)
    if labels["track_id"].duplicated().any() or features["track_id"].duplicated().any():
        raise ValueError("track_id must be unique in labels and feature inputs")
    missing_features = [column for column in FEATURES if column not in features.columns]
    if missing_features:
        raise ValueError(f"Feature CSV is missing columns: {', '.join(missing_features)}")
    label_columns = ["track_id", "class", "source_dataset", "roi_type", "ground_truth_quality", "crop_filename"]
    merged = labels[label_columns].merge(
        features[["track_id", "processing_status"] + FEATURES],
        on="track_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged[(merged["class"].isin(CLASS_ORDER)) & (merged["processing_status"] == "success")].copy()
    merged["source_dataset"] = merged["source_dataset"].map(_source_group)
    for feature in FEATURES:
        merged[feature] = pd.to_numeric(merged[feature], errors="coerce")

    source_table = class_source_contingency(labels)
    association = categorical_association(source_table)
    roi_table = class_roi_contingency(labels)
    by_class = descriptive_summary(merged, ["class"])
    by_source = descriptive_summary(merged, ["source_dataset"])
    by_class_source = descriptive_summary(merged, ["class", "source_dataset"])
    cross = _add_exploratory_centroid_distances(merged, identify_cross_source_samples(merged))

    quality = (
        labels.assign(source_dataset=labels["source_dataset"].map(_source_group))
        .groupby(["class", "source_dataset", "ground_truth_quality"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    pca_variance = None
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        source_table.to_csv(output / "class_source_contingency.csv", index=False)
        roi_table.to_csv(output / "class_roi_type_contingency.csv", index=False)
        by_class.to_csv(output / "feature_summary_by_class.csv", index=False)
        by_source.to_csv(output / "feature_summary_by_source.csv", index=False)
        by_class_source.to_csv(output / "feature_summary_by_class_and_source.csv", index=False)
        cross.to_csv(output / "cross_source_samples.csv", index=False)
        if create_plots:
            plots = output / "plots"
            _plot_grouped_feature(merged, "mean_width_px", plots / "width_by_class_source.png", "Mean width by class and source")
            _plot_grouped_feature(merged, "track_length_px", plots / "length_by_class_source.png", "Track length by class and source")
            _plot_grouped_feature(merged, "mean_curvature", plots / "curvature_by_class_source.png", "Mean curvature by class and source")
            _plot_grouped_feature(merged, "tortuosity", plots / "tortuosity_by_class_source.png", "Tortuosity by class and source")
            pca_variance = _pca_plots(merged, plots)

    phi = association["phi_cramers_v"]
    cross_minimum = min(
        int(source_table.set_index("source_dataset").loc["Lecturer", "alpha_count"]),
        int(source_table.set_index("source_dataset").loc["AWAN", "electron_count"]),
    )
    risk = "HIGH" if phi is not None and phi >= 0.7 and cross_minimum <= 2 else ("MODERATE" if phi is not None and phi >= 0.3 else "LOW")
    report_lines = [
        "SOURCE-CONFOUND DIAGNOSTIC REPORT",
        "",
        "This is a diagnostic of class-source association, not a particle classifier evaluation.",
        "No classifier was trained and no accuracy metrics were calculated.",
        "",
        "ACTUAL SOURCE DISTRIBUTION",
    ]
    indexed_source = source_table.set_index("source_dataset")
    for class_name, column in (("Alpha", "alpha_count"), ("Electron", "electron_count")):
        report_lines.append(f"{class_name}: AWAN={int(indexed_source.loc['AWAN', column])}, Lecturer={int(indexed_source.loc['Lecturer', column])}, Other={int(indexed_source.loc['Other', column])}")
    report_lines.extend(
        [
            "",
            "CLASS-SOURCE ASSOCIATION",
            f"Fisher exact odds ratio: {association['fisher_odds_ratio']}",
            f"Fisher exact two-sided p-value: {association['fisher_p_value']}",
            f"Absolute phi / Cramer's V: {association['phi_cramers_v']}",
            "The small-sample p-value and effect size diagnose dependence between provenance and label; they do not measure classifier performance.",
            "",
            "CROSS-SOURCE SAMPLES",
            f"Lecturer Alpha: {_format_ids(merged, 'Alpha', 'Lecturer')}",
            f"AWAN Electron: {_format_ids(merged, 'Electron', 'AWAN')}",
        ]
    )
    for _, row in cross.iterrows():
        report_lines.append(
            f"{row['track_id']}: standardized distance to AWAN Alpha centroid={row['distance_to_awan_alpha_centroid']:.4f}; "
            f"to Lecturer Electron centroid={row['distance_to_lecturer_electron_centroid']:.4f}; closer={row['closer_primary_group']}."
        )
    indexed_roi = roi_table.set_index("roi_type")
    report_lines.extend(
        [
            "",
            "ROI-TYPE DISTRIBUTION",
            f"Alpha: axis_aligned={int(indexed_roi.loc['axis_aligned', 'alpha_count'])}, rotated={int(indexed_roi.loc['rotated', 'alpha_count'])}",
            f"Electron: axis_aligned={int(indexed_roi.loc['axis_aligned', 'electron_count'])}, rotated={int(indexed_roi.loc['rotated', 'electron_count'])}",
            "",
            "GROUND-TRUTH QUALITY BY CLASS AND SOURCE",
        ]
    )
    report_lines.extend(
        f"{row['class']} / {row['source_dataset']} / {row['ground_truth_quality'] or 'missing'}: {int(row['count'])}"
        for _, row in quality.iterrows()
    )
    if pca_variance is not None:
        report_lines.extend(["", f"PCA variance: PC1={pca_variance[0] * 100:.2f}%, PC2={pca_variance[1] * 100:.2f}%."])
    report_lines.extend(
        [
            "",
            f"SOURCE-CONFOUND RISK: {risk}",
            "The class/source table is nearly diagonal and each cross-source combination has only one sample. Feature and PCA comparisons are exploratory because source and class cannot be disentangled reliably at this sample size.",
            "Pixel-based geometry may also retain acquisition-scale effects.",
            "",
            "RECOMMENDED NEXT DATA COLLECTION",
            f"Lecturer Alpha: current={int(indexed_source.loc['Lecturer', 'alpha_count'])}; first milestone=8 (+{max(0, 8-int(indexed_source.loc['Lecturer', 'alpha_count']))}); stretch=10.",
            f"AWAN Electron: current={int(indexed_source.loc['AWAN', 'electron_count'])}; first milestone=8 (+{max(0, 8-int(indexed_source.loc['AWAN', 'electron_count']))}); stretch=10.",
            "Collect independent events across additional acquisition sessions; a third source/setup would materially improve source-generalization assessment.",
            "With only two highly imbalanced sources, GroupShuffleSplit or leave-one-source-out evaluation is not currently credible. After balancing both classes within each source, use source-held-out checks cautiously; with at least three sources, prefer group-aware evaluation such as StratifiedGroupKFold using acquisition session/event groups.",
            "",
            "RANDOM FOREST TRAINED: NO",
            "READY FOR CREDIBLE CLASSIFICATION: NO",
        ]
    )
    report = "\n".join(report_lines) + "\n"
    if output is not None:
        (output / "source_confound_report.txt").write_text(report, encoding="utf-8")
    return {
        "source_contingency": source_table,
        "association": association,
        "roi_contingency": roi_table,
        "feature_summary_by_class": by_class,
        "feature_summary_by_source": by_source,
        "feature_summary_by_class_and_source": by_class_source,
        "cross_source_samples": cross,
        "quality_distribution": quality,
        "risk": risk,
        "pca_variance": pca_variance,
        "report": report,
    }

