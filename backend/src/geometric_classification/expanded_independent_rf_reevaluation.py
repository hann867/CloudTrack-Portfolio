"""Grouped exploratory RF re-evaluation for the finalized 34-row dataset."""

from __future__ import annotations

import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold

from .model_contract import AUTHORIZED_CLASS_NAMES, AUTHORIZED_FEATURE_NAMES


FEATURES = list(AUTHORIZED_FEATURE_NAMES)
CLASSES = list(AUTHORIZED_CLASS_NAMES)
RF_PARAMETERS = {"n_estimators": 100, "random_state": 42, "class_weight": "balanced"}
PRIOR = {"balanced_accuracy": 0.8976, "macro_f1": 0.8966, "mcc": 0.7952}
REQUIRED = [
    "ground_truth_track_id", "ground_truth_class", "source_dataset", "source_image",
    "source_video", "event_id", "roi_type", "physical_track_group_id", "dependency_group_id",
    "selected_candidate", "segmentation_source", "processing_status", *FEATURES,
]


def _atomic_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=index); os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except OSError: pass
        raise


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle: handle.write(text)
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except OSError: pass
        raise


def load_final_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = [column for column in REQUIRED if column not in frame]
    if missing: raise ValueError(f"Final dataset is missing required columns: {', '.join(missing)}")
    if len(frame) != 34: raise ValueError(f"Expected 34 finalized rows, found {len(frame)}.")
    if frame["ground_truth_track_id"].duplicated().any(): raise ValueError("Duplicate GT IDs are forbidden.")
    if frame["selected_candidate"].duplicated().any(): raise ValueError("Duplicate Module 2 masks are forbidden.")
    if "A008" in set(frame["ground_truth_track_id"]): raise ValueError("A008 must be excluded from independent modelling.")
    if set(frame["ground_truth_class"]) != set(CLASSES): raise ValueError("Both Alpha and Electron are required.")
    if Counter(frame["ground_truth_class"]) != Counter({"Alpha": 15, "Electron": 19}):
        raise ValueError("Final class distribution does not match the verified 15/19 contract.")
    if Counter(frame["source_dataset"]) != Counter({"AWAN": 6, "Lecturer": 28}):
        raise ValueError("Final source distribution does not match the verified 6/28 contract.")
    if int(((frame.source_dataset == "AWAN") & (frame.ground_truth_class == "Electron")).sum()) != 0:
        raise ValueError("Verified source limitation changed: AWAN Electron must equal zero.")
    if frame["dependency_group_id"].nunique() != 29: raise ValueError("Expected 29 dependency groups.")
    if frame["dependency_group_id"].str.strip().eq("").any(): raise ValueError("Blank dependency groups are forbidden.")
    if not frame["processing_status"].eq("SUCCESS").all(): raise ValueError("Every finalized row must be a successful extraction.")
    if not frame["segmentation_source"].eq("production_module2_human_selected").all():
        raise ValueError("Every row must come from a human-selected production Module 2 mask.")
    frame[FEATURES] = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all(): raise ValueError("Predictors contain NaN/infinity.")
    # Metadata dependencies that repeat must never map to different declared groups.
    for field in ("event_id", "source_image", "physical_track_group_id"):
        selected = frame[frame[field].astype(str).str.strip().ne("")]
        if any(group.dependency_group_id.nunique() > 1 for _, group in selected.groupby(field)):
            raise ValueError(f"Repeated {field} crosses declared dependency groups.")
    return frame.sort_values("ground_truth_track_id").reset_index(drop=True)


def choose_splits(frame: pd.DataFrame, maximum_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    x = frame[FEATURES].to_numpy(dtype=float); y = frame["ground_truth_class"].to_numpy()
    groups = frame["dependency_group_id"].to_numpy()
    class_group_limit = min(frame.loc[frame.ground_truth_class == name, "dependency_group_id"].nunique() for name in CLASSES)
    for folds in range(min(maximum_folds, class_group_limit), 1, -1):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
        splits = list(splitter.split(x, y, groups))
        if all(set(y[train]) == set(CLASSES) and set(y[test]) == set(CLASSES)
               and not (set(groups[train]) & set(groups[test])) for train, test in splits):
            return splits
    raise ValueError("No leakage-free grouped split with both classes in every train/test fold is feasible.")


def metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    matrix = confusion_matrix(true, pred, labels=CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(true, pred, labels=CLASSES, zero_division=0)
    return {
        "n": len(true), "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(true, pred)),
        "alpha_precision": float(precision[0]), "alpha_recall": float(recall[0]), "alpha_f1": float(f1[0]),
        "electron_precision": float(precision[1]), "electron_recall": float(recall[1]), "electron_f1": float(f1[1]),
        "alpha_support": int(support[0]), "electron_support": int(support[1]),
        "alpha_as_alpha": int(matrix[0,0]), "alpha_as_electron": int(matrix[0,1]),
        "electron_as_alpha": int(matrix[1,0]), "electron_as_electron": int(matrix[1,1]),
    }


def group_bootstrap(predictions: pd.DataFrame, repetitions: int = 5000) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    groups = np.asarray(sorted(predictions.dependency_group_id.unique()))
    index = {group: np.flatnonzero(predictions.dependency_group_id.to_numpy() == group) for group in groups}
    names = ("accuracy", "balanced_accuracy", "macro_f1", "mcc")
    samples = {name: [] for name in names}; valid = 0
    for _ in range(repetitions):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        subset = predictions.iloc[np.concatenate([index[group] for group in sampled])]
        if subset.true_class.nunique() < 2: continue
        result = metrics(subset.true_class, subset.predicted_class); valid += 1
        for name in names: samples[name].append(result[name])
    output: dict[str, Any] = {"bootstrap_requested": repetitions, "bootstrap_valid": valid, "bootstrap_unit": "dependency_group"}
    for name in names:
        output[f"{name}_ci_low"] = float(np.percentile(samples[name], 2.5))
        output[f"{name}_ci_high"] = float(np.percentile(samples[name], 97.5))
    return output


def subgroup_diagnostics(predictions: pd.DataFrame, field: str) -> pd.DataFrame:
    rows = []
    for value, group in predictions.groupby(field, sort=True, dropna=False):
        both = group.true_class.nunique() == 2
        if both:
            result = metrics(group.true_class, group.predicted_class)
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                group.true_class, group.predicted_class, labels=CLASSES, zero_division=0
            )
            result = {
                "accuracy": float(accuracy_score(group.true_class, group.predicted_class)),
                "alpha_precision": float(precision[0]), "alpha_recall": float(recall[0]), "alpha_f1": float(f1[0]),
                "electron_precision": float(precision[1]), "electron_recall": float(recall[1]), "electron_f1": float(f1[1]),
            }
        rows.append({
            "subgroup": value or "UNAVAILABLE", "n": len(group),
            "alpha": int((group.true_class == "Alpha").sum()), "electron": int((group.true_class == "Electron").sum()),
            "accuracy": result["accuracy"], "balanced_accuracy": result.get("balanced_accuracy", ""),
            "macro_f1": result.get("macro_f1", ""), "mcc": result.get("mcc", ""),
            "alpha_precision": result["alpha_precision"], "alpha_recall": result["alpha_recall"], "alpha_f1": result["alpha_f1"],
            "electron_precision": result["electron_precision"], "electron_recall": result["electron_recall"], "electron_f1": result["electron_f1"],
            "interpretation": "DESCRIPTIVE ONLY; both classes present" if both else "DESCRIPTIVE ONLY; single-class subgroup",
        })
    return pd.DataFrame(rows)


def _classification_report(result: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        {"class": "Alpha", "precision": result["alpha_precision"], "recall": result["alpha_recall"], "f1": result["alpha_f1"], "support": result["alpha_support"]},
        {"class": "Electron", "precision": result["electron_precision"], "recall": result["electron_recall"], "f1": result["electron_f1"], "support": result["electron_support"]},
        {"class": "macro_avg", "precision": (result["alpha_precision"]+result["electron_precision"])/2,
         "recall": result["balanced_accuracy"], "f1": result["macro_f1"], "support": result["n"]},
    ])


def run_grouped_reevaluation(input_csv: str | Path, output_dir: str | Path,
                             *, bootstrap_repetitions: int = 5000, permutation_repeats: int = 50) -> dict[str, Any]:
    frame = load_final_dataset(input_csv); splits = choose_splits(frame, maximum_folds=5)
    x = frame[FEATURES].to_numpy(dtype=float); y = frame.ground_truth_class.to_numpy()
    groups = frame.dependency_group_id.to_numpy(); masks = frame.selected_candidate.to_numpy()
    predicted = np.empty(len(frame), object); probabilities = np.full((len(frame), 2), np.nan)
    assignments = np.zeros(len(frame), int); coverage = np.zeros(len(frame), int)
    fold_audit = []; importance_rows = []; leakage = False
    for fold, (train, test) in enumerate(splits, 1):
        leaked_groups = set(groups[train]) & set(groups[test])
        leaked_gt = set(frame.iloc[train].ground_truth_track_id) & set(frame.iloc[test].ground_truth_track_id)
        leaked_masks = set(masks[train]) & set(masks[test])
        train_physical = set(frame.iloc[train].physical_track_group_id) - {""}
        test_physical = set(frame.iloc[test].physical_track_group_id) - {""}
        leaked_physical = train_physical & test_physical
        leakage = leakage or bool(leaked_groups or leaked_gt or leaked_masks or leaked_physical)
        model = RandomForestClassifier(**RF_PARAMETERS); model.fit(x[train], y[train])
        predicted[test] = model.predict(x[test]); fold_prob = model.predict_proba(x[test])
        for index, class_name in enumerate(model.classes_): probabilities[test, CLASSES.index(class_name)] = fold_prob[:, index]
        assignments[test] = fold; coverage[test] += 1
        fold_audit.append({
            "fold": fold, "train_n": len(train), "test_n": len(test),
            "train_alpha": int(np.sum(y[train]=="Alpha")), "train_electron": int(np.sum(y[train]=="Electron")),
            "test_alpha": int(np.sum(y[test]=="Alpha")), "test_electron": int(np.sum(y[test]=="Electron")),
            "train_groups": len(set(groups[train])), "test_groups": len(set(groups[test])),
            "gt_leakage_count": len(leaked_gt), "dependency_leakage_count": len(leaked_groups),
            "mask_leakage_count": len(leaked_masks), "physical_track_leakage_count": len(leaked_physical),
        })
        heldout = permutation_importance(model, x[test], y[test], scoring="balanced_accuracy",
                                         n_repeats=permutation_repeats, random_state=42+fold)
        for index, feature in enumerate(FEATURES):
            importance_rows.append({"fold": fold, "feature": feature,
                                    "importance": float(heldout.importances_mean[index]),
                                    "within_fold_std": float(heldout.importances_std[index]), "test_n": len(test)})
    if not np.all(coverage == 1) or not np.isfinite(probabilities).all(): raise RuntimeError("OOF coverage/probabilities are incomplete.")
    predictions = pd.DataFrame({
        "ground_truth_track_id": frame.ground_truth_track_id, "true_class": y, "predicted_class": predicted,
        "probability_alpha": probabilities[:,0], "probability_electron": probabilities[:,1], "fold": assignments,
        "dependency_group_id": frame.dependency_group_id, "event_id": frame.event_id,
        "source_dataset": frame.source_dataset, "source_video": frame.source_video.replace("", "UNAVAILABLE"),
        "roi_type": frame.roi_type, "source_image": frame.source_image,
        "module2_track_identity": frame.selected_candidate, "physical_track_group_id": frame.physical_track_group_id,
    })
    if predictions.ground_truth_track_id.duplicated().any() or not np.allclose(predictions[["probability_alpha","probability_electron"]].sum(axis=1), 1):
        raise RuntimeError("OOF identity/probability validation failed.")
    result = metrics(predictions.true_class, predictions.predicted_class); result.update(group_bootstrap(predictions, bootstrap_repetitions))
    importance_raw = pd.DataFrame(importance_rows); summaries=[]
    for feature, group in importance_raw.groupby("feature", sort=False):
        summaries.append({"feature": feature, "importance_mean": float(np.average(group.importance, weights=group.test_n)),
                          "importance_std_across_folds": float(group.importance.std(ddof=1)),
                          "mean_within_fold_std": float(group.within_fold_std.mean()), "folds": len(group)})
    importance = pd.DataFrame(summaries).sort_values(["importance_mean","feature"], ascending=[False,True]).reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance)+1))
    misclassified = predictions[predictions.true_class != predictions.predicted_class].copy()
    source = subgroup_diagnostics(predictions, "source_dataset")
    roi = subgroup_diagnostics(predictions, "roi_type")
    video = subgroup_diagnostics(predictions, "source_video")
    matrix = pd.DataFrame([[result["alpha_as_alpha"],result["alpha_as_electron"]],
                           [result["electron_as_alpha"],result["electron_as_electron"]]],
                          index=["true_alpha","true_electron"], columns=["predicted_alpha","predicted_electron"])
    fold_assignments = predictions[["ground_truth_track_id","fold","dependency_group_id","module2_track_identity","physical_track_group_id"]].copy()
    fold_assignments["test_coverage"] = 1
    ci_width = result["balanced_accuracy_ci_high"] - result["balanced_accuracy_ci_low"]
    if leakage or result["balanced_accuracy"] < .60 or result["macro_f1"] < .60 or result["mcc"] < .20 or ci_width > .55:
        decision = "C. TOO UNSTABLE — MORE CLEAN DATA NEEDED FIRST"
    else:
        # Severe source/class confounding (AWAN Electron=0) precludes artifact replacement from this evaluation.
        decision = "B. USABLE FOR FYP EXPLORATORY EVIDENCE, BUT DO NOT UPDATE ARTIFACT YET"

    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(predictions, output/"oof_predictions.csv")
    _atomic_csv(fold_assignments, output/"fold_assignments.csv")
    _atomic_csv(pd.DataFrame(fold_audit), output/"fold_audit.csv")
    _atomic_csv(matrix, output/"confusion_matrix.csv", index=True)
    _atomic_csv(_classification_report(result), output/"classification_report.csv")
    _atomic_csv(importance, output/"feature_importance.csv")
    miscols = ["ground_truth_track_id","true_class","predicted_class","probability_alpha","probability_electron",
               "source_dataset","roi_type","dependency_group_id","event_id","module2_track_identity"]
    _atomic_csv(misclassified[miscols], output/"misclassifications.csv")
    _atomic_csv(source, output/"source_diagnostics.csv")
    _atomic_csv(roi, output/"roi_diagnostics.csv")
    _atomic_csv(video, output/"source_video_diagnostics.csv")

    lines = [
        "EXPANDED INDEPENDENT GROUPED RF RE-EVALUATION", "Exploratory evidence only; not a production accuracy claim.", "",
        "DATASET", "Input: final_independent_module3_features.csv only", "Samples: 34; Alpha=15; Electron=19",
        "Sources: AWAN=6; Lecturer=28; AWAN Electron=0", "Predictors (exactly seven): " + ", ".join(FEATURES),
        f"RF settings: {RF_PARAMETERS}; no tuning", "", "GROUPED CROSS-VALIDATION",
        f"Dependency groups: {frame.dependency_group_id.nunique()}", f"CV folds: {len(splits)} (largest sensible cap=5)",
        "Splitter: StratifiedGroupKFold(shuffle=True, random_state=42)",
        *[f"Fold {r['fold']}: train A/E={r['train_alpha']}/{r['train_electron']}, test A/E={r['test_alpha']}/{r['test_electron']}, train/test groups={r['train_groups']}/{r['test_groups']}" for r in fold_audit],
        "OOF coverage: 34/34 exactly once", f"LEAKAGE DETECTED: {'YES' if leakage else 'NO'}", "",
        "OOF METRICS",
        f"Accuracy: {result['accuracy']:.6f} (95% group-bootstrap CI {result['accuracy_ci_low']:.6f}-{result['accuracy_ci_high']:.6f})",
        f"Balanced accuracy: {result['balanced_accuracy']:.6f} (95% CI {result['balanced_accuracy_ci_low']:.6f}-{result['balanced_accuracy_ci_high']:.6f})",
        f"Macro F1: {result['macro_f1']:.6f} (95% CI {result['macro_f1_ci_low']:.6f}-{result['macro_f1_ci_high']:.6f})",
        f"MCC: {result['mcc']:.6f} (95% CI {result['mcc_ci_low']:.6f}-{result['mcc_ci_high']:.6f})",
        f"Alpha precision/recall/F1: {result['alpha_precision']:.6f}/{result['alpha_recall']:.6f}/{result['alpha_f1']:.6f}",
        f"Electron precision/recall/F1: {result['electron_precision']:.6f}/{result['electron_recall']:.6f}/{result['electron_f1']:.6f}",
        f"Confusion matrix [[A->A,A->E],[E->A,E->E]]: [[{result['alpha_as_alpha']},{result['alpha_as_electron']}],[{result['electron_as_alpha']},{result['electron_as_electron']}]]",
        f"Bootstrap: {result['bootstrap_valid']}/{result['bootstrap_requested']} valid; unit=dependency group; fixed OOF predictions, models not refit", "",
        f"MISCLASSIFICATIONS: {len(misclassified)}",
        *[f"  {r.ground_truth_track_id}: {r.true_class}->{r.predicted_class}, P(A)={r.probability_alpha:.3f}, P(E)={r.probability_electron:.3f}, source={r.source_dataset}, ROI={r.roi_type}, group={r.dependency_group_id}, M2={r.module2_track_identity}" for r in misclassified.itertuples()], "",
        "HELD-OUT PERMUTATION IMPORTANCE",
        *[f"  {int(r.rank)}. {r.feature}: {r.importance_mean:.6f}" for r in importance.itertuples()],
        "Importance uses test-fold balanced-accuracy permutation only. Estimates are unstable with n=34.", "",
        "SOURCE / SUBGROUP LIMITATIONS",
        "AWAN Electron: 0. AWAN-specific Electron performance is undefined.",
        "No source-held-out AWAN evaluation was presented as meaningful for both classes.",
        "Source, ROI, and source-video diagnostics are descriptive only; small/single-class subgroups must not be over-interpreted.",
        "Source/class confounding remains severe. SOURCE GENERALISATION CLAIM: NOT SUPPORTED", "",
        "COMPARISON WITH PREVIOUS 29-ROW PRODUCTION PILOT",
        f"Previous balanced accuracy≈{PRIOR['balanced_accuracy']:.4f}; current={result['balanced_accuracy']:.4f}; descriptive difference={result['balanced_accuracy']-PRIOR['balanced_accuracy']:+.4f}",
        f"Previous macro F1≈{PRIOR['macro_f1']:.4f}; current={result['macro_f1']:.4f}; descriptive difference={result['macro_f1']-PRIOR['macro_f1']:+.4f}",
        f"Previous MCC≈{PRIOR['mcc']:.4f}; current={result['mcc']:.4f}; descriptive difference={result['mcc']-PRIOR['mcc']:+.4f}",
        "No improvement claim is valid: sample composition changed, stricter human mask adjudication was applied, AWAN Electron disappeared, and evaluations are not directly equivalent.", "",
        f"FINAL DECISION: {decision}", "RF EVALUATED: YES", "RF RETRAINED/PERSISTED: NO",
        "EXISTING MODEL ARTIFACT MODIFIED: NO",
    ]
    _atomic_text(output/"rf_reevaluation_summary.txt", "\n".join(lines)+"\n")
    return {"samples": len(frame), "alpha": 15, "electron": 19, "groups": frame.dependency_group_id.nunique(),
            "folds": len(splits), "metrics": result, "misclassifications": len(misclassified),
            "leakage_detected": leakage, "top_features": importance.feature.tolist(), "decision": decision}
