"""Read-only visual diagnostics for problematic production Module 2 mappings."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .ground_truth_candidate_matcher import manual_polygon
from .skeleton_analysis import NEIGHBOURS, principal_skeleton_path, skeletonise_mask


PROBLEM_TRACKS = ("A012", "A020", "A021", "A023", "A003", "A008", "A025", "A028", "E025")
DUPLICATE_PAIRS = (("A003", "A008"), ("A025", "A028"))
OUTPUT_COLUMNS = [
    "ground_truth_track_id", "issue_type", "current_module2_track_id", "diagnosis",
    "recommended_action", "manual_reselection_needed", "geometry_review_needed", "notes",
]
ALLOWED_ACTIONS = {
    "KEEP", "RESELECT_CANDIDATE", "EXCLUDE_NO_VALID_MODULE2_CANDIDATE",
    "REVIEW_GROUND_TRUTH", "REVIEW_MODULE3_GEOMETRY",
}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, output)
    except Exception:
        try: os.unlink(temporary)
        except OSError: pass
        raise


def _atomic_text(path: str | Path, text: str) -> None:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle: handle.write(text)
        os.replace(temporary, output)
    except Exception:
        try: os.unlink(temporary)
        except OSError: pass
        raise


def exact_file_index(roots: Sequence[str | Path], wanted: set[str]) -> dict[str, Path]:
    found: dict[str, list[Path]] = defaultdict(list)
    for value in roots:
        root = Path(value)
        if not root.is_dir(): continue
        for path in root.rglob("*"):
            if path.is_file() and path.name in wanted: found[path.name].append(path)
    return {name: paths[0] for name, paths in found.items() if len(paths) == 1}


def diagnostic_index(root: str | Path) -> dict[str, dict[str, Any]]:
    index = {}
    for manifest in Path(root).rglob("manifest.csv"):
        for row in _read_csv(manifest):
            track_id = row.get("track_id", "")
            if not track_id: continue
            if track_id in index: raise ValueError(f"Duplicate diagnostic track_id: {track_id}")
            item = dict(row); item["manifest"] = manifest
            item["mask_path"] = manifest.parent / row["diagnostic_mask_file"]
            item["overlay_path"] = manifest.parent / row["diagnostic_overlay_file"]
            index[track_id] = item
    return index


def _decode(path: Path | None, grayscale: bool = False) -> np.ndarray | None:
    if path is None or not path.is_file(): return None
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)


def _skeleton_nodes(skeleton: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    nodes = set(map(tuple, np.argwhere(skeleton)))
    endpoints, branches = [], []
    for node in nodes:
        y, x = node
        degree = sum((y + dy, x + dx) in nodes for dy, dx, _ in NEIGHBOURS)
        if degree == 1: endpoints.append(node)
        if degree >= 3: branches.append(node)
    return endpoints, branches


def mask_diagnostics(mask: np.ndarray, annotation: Mapping[str, str], diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    binary = mask > 0
    component_count, _, _, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    skeleton = skeletonise_mask(binary)
    endpoints, branches = _skeleton_nodes(skeleton)
    path_result = principal_skeleton_path(skeleton)
    polygon = manual_polygon(annotation).copy()
    polygon[:, 0] -= float(diagnostic["roi_x0"]); polygon[:, 1] -= float(diagnostic["roi_y0"])
    manual_mask = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(manual_mask, [np.rint(polygon).astype(np.int32)], 1)
    intersection = int(np.count_nonzero(binary & (manual_mask > 0)))
    area = int(binary.sum()); manual_area = int(manual_mask.sum())
    metrics = {
        "mask_area_pixels": area, "manifest_area_pixels": int(float(diagnostic.get("area_pixels") or 0)),
        "foreground_component_count": int(component_count - 1), "skeleton_pixels": int(skeleton.sum()),
        "skeleton_endpoint_count": len(endpoints), "skeleton_branch_pixel_count": len(branches),
        "mask_fraction_inside_manual_roi": intersection / area if area else 0.0,
        "manual_roi_foreground_fraction": intersection / manual_area if manual_area else 0.0,
        "principal_path_status": path_result.get("processing_status"),
        "principal_path_error": path_result.get("error_message") or "",
    }
    if path_result.get("processing_status") == "success":
        path = np.asarray(path_result["path"])
        length = float(path_result["track_length_px"])
        endpoint_distance = float(np.linalg.norm(path[-1] - path[0]))
        metrics.update({"principal_path_length_px": length, "principal_endpoint_distance_px": endpoint_distance,
                        "computed_tortuosity": length / endpoint_distance if endpoint_distance else math.inf})
    return {**metrics, "_skeleton": skeleton, "_endpoints": endpoints, "_branches": branches,
            "_path_result": path_result}


def _fit(image: np.ndarray | None, width: int, height: int, title: str) -> np.ndarray:
    panel = np.full((height, width, 3), 22, np.uint8)
    cv2.putText(panel, title[:75], (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    if image is None:
        cv2.putText(panel, "unavailable", (20, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150,150,150), 2)
        return panel
    if image.ndim == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = min(width/image.shape[1], (height-38)/image.shape[0])
    resized = cv2.resize(image, (max(1,int(image.shape[1]*scale)), max(1,int(image.shape[0]*scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    x=(width-resized.shape[1])//2; y=38+(height-38-resized.shape[0])//2
    panel[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    return panel


def _mask_skeleton_view(mask: np.ndarray, metrics: Mapping[str, Any]) -> np.ndarray:
    view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    skeleton = metrics["_skeleton"]
    view[skeleton] = (0, 0, 255)
    for y, x in metrics["_endpoints"]: cv2.circle(view, (x,y), 2, (255,0,0), -1)
    path_result = metrics["_path_result"]
    if path_result.get("processing_status") == "success":
        path = np.asarray(path_result["path"])
        view[path[:,0], path[:,1]] = (0,255,0)
        for y,x in (path[0],path[-1]): cv2.circle(view,(int(x),int(y)),5,(0,255,255),-1)
    return view


def _source_view(source: np.ndarray, annotation: Mapping[str,str], candidate: Mapping[str,Any], mask: np.ndarray,
                 alternatives: Sequence[Mapping[str,str]], diagnostics: Mapping[str,Mapping[str,Any]]) -> np.ndarray:
    view=source.copy(); polygon=np.rint(manual_polygon(annotation)).astype(np.int32)
    cv2.polylines(view,[polygon],True,(255,255,0),4,cv2.LINE_AA)
    colours=[(255,0,255),(0,255,0),(0,165,255)]
    selected_id=candidate["track_id"]
    for index, row in enumerate(alternatives):
        diagnostic=diagnostics.get(row["candidate_track_id"])
        if not diagnostic: continue
        x,y,w,h=[int(float(diagnostic[key])) for key in ("bbox_x","bbox_y","bbox_width","bbox_height")]
        colour=colours[index]
        cv2.rectangle(view,(x,y),(x+w,y+h),colour,2)
        marker=f"R{row['rank']}" + (" SELECTED" if row["candidate_track_id"]==selected_id else "")
        cv2.putText(view,marker,(x+3,max(22,y+22)),cv2.FONT_HERSHEY_SIMPLEX,0.6,colour,2,cv2.LINE_AA)
    x0,y0=int(float(candidate["roi_x0"])),int(float(candidate["roi_y0"]))
    y1=min(view.shape[0],y0+mask.shape[0]);x1=min(view.shape[1],x0+mask.shape[1])
    if x0<x1 and y0<y1:
        foreground=mask[:y1-y0,:x1-x0]>0; region=view[y0:y1,x0:x1]
        tint=np.empty_like(region);tint[:]=(0,0,255)
        region[foreground]=cv2.addWeighted(region[foreground],.55,tint[foreground],.45,0)
    return view


def render_track_diagnostic(item: Mapping[str,Any], source: np.ndarray|None, crop: np.ndarray|None,
                            diagnostics: Mapping[str,Mapping[str,Any]]) -> tuple[np.ndarray,dict[str,Any]]:
    annotation=item["annotation"]; selected=diagnostics[item["mapping"]["module2_track_id"]]
    mask=_decode(selected["mask_path"],True)
    if mask is None: raise OSError(f"Missing selected mask for {annotation['track_id']}")
    metrics=mask_diagnostics(mask,annotation,selected)
    canvas=np.full((1380,2000,3),15,np.uint8)
    cv2.putText(canvas,f"{annotation['track_id']} {annotation['class']} | issue {item['issue_type']}",(15,35),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),2)
    cv2.putText(canvas,f"selected: {selected['track_id']}",(15,70),cv2.FONT_HERSHEY_SIMPLEX,.62,(220,220,220),1)
    line=(f"mask area {metrics['mask_area_pixels']} | components {metrics['foreground_component_count']} | skeleton {metrics['skeleton_pixels']} | "
          f"endpoints {metrics['skeleton_endpoint_count']} | branch pixels {metrics['skeleton_branch_pixel_count']} | mask in manual ROI {metrics['mask_fraction_inside_manual_roi']:.4f}")
    cv2.putText(canvas,line,(15,102),cv2.FONT_HERSHEY_SIMPLEX,.55,(220,220,220),1)
    cv2.putText(canvas,f"source: {annotation['source_image']}",(15,132),cv2.FONT_HERSHEY_SIMPLEX,.52,(200,200,200),1)
    source_panel=_fit(_source_view(source,annotation,selected,mask,item["alternatives"],diagnostics) if source is not None else None,1180,700,"Original source: cyan manual ROI; red selected mask; ranked bboxes")
    crop_panel=_fit(crop,780,330,"Stored ground-truth crop")
    overlay_panel=_fit(_decode(selected["overlay_path"]),390,370,"Current selected Module 2 overlay")
    mask_panel=_fit(_mask_skeleton_view(mask,metrics),390,370,"Selected mask: red skeleton, blue endpoints, green path")
    canvas[155:855,15:1195]=source_panel;canvas[155:485,1205:1985]=crop_panel
    canvas[495:865,1205:1595]=overlay_panel;canvas[495:865,1595:1985]=mask_panel
    for index in range(3):
        row=item["alternatives"][index] if index<len(item["alternatives"]) else None
        image=None;title=f"Rank {index+1}: unavailable"
        if row:
            diag=diagnostics.get(row["candidate_track_id"]);image=_decode(diag["overlay_path"]) if diag else None
            title=f"Rank {row['rank']} IoU {float(row['iou']):.3f}: {row['candidate_track_id']}"
        canvas[875:1365,15+index*660:645+index*660]=_fit(image,630,490,title)
    return canvas,metrics


def render_duplicate_pair(pair: Sequence[Mapping[str,Any]], source: np.ndarray, crops: Sequence[np.ndarray|None],
                          diagnostics: Mapping[str,Mapping[str,Any]]) -> np.ndarray:
    selected=diagnostics[pair[0]["mapping"]["module2_track_id"]];mask=_decode(selected["mask_path"],True)
    if mask is None: raise OSError("Shared duplicate mask is missing.")
    view=source.copy();colours=[(255,255,0),(255,0,255)]
    for item,colour in zip(pair,colours):
        polygon=np.rint(manual_polygon(item["annotation"])).astype(np.int32);cv2.polylines(view,[polygon],True,colour,5)
        x,y=polygon[0];cv2.putText(view,item["annotation"]["track_id"],(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,.8,colour,2)
    x0,y0=int(float(selected["roi_x0"])),int(float(selected["roi_y0"]));y1=min(view.shape[0],y0+mask.shape[0]);x1=min(view.shape[1],x0+mask.shape[1])
    foreground=mask[:y1-y0,:x1-x0]>0;region=view[y0:y1,x0:x1];tint=np.empty_like(region);tint[:]=(0,0,255)
    region[foreground]=cv2.addWeighted(region[foreground],.55,tint[foreground],.45,0)
    canvas=np.full((1200,1800,3),15,np.uint8);ids=" / ".join(item["annotation"]["track_id"] for item in pair)
    cv2.putText(canvas,f"Duplicate mapping review: {ids}",(15,38),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
    cv2.putText(canvas,f"shared candidate: {selected['track_id']}",(15,75),cv2.FONT_HERSHEY_SIMPLEX,.6,(220,220,220),1)
    canvas[95:795,15:1115]=_fit(view,1100,700,"Shared original source: both manual polygons + shared mask")
    canvas[95:435,1125:1790]=_fit(crops[0],665,340,f"Stored crop {pair[0]['annotation']['track_id']}")
    canvas[445:785,1125:1790]=_fit(crops[1],665,340,f"Stored crop {pair[1]['annotation']['track_id']}")
    canvas[805:1190,15:885]=_fit(_decode(selected["overlay_path"]),870,385,"Shared Module 2 overlay")
    metrics=mask_diagnostics(mask,pair[0]["annotation"],selected)
    canvas[805:1190,900:1790]=_fit(_mask_skeleton_view(mask,metrics),890,385,"Shared mask and principal skeleton")
    return canvas


def build_adjudication_context(labels_csv: str|Path,mapping_csv: str|Path,matches_csv: str|Path,
                               diagnostics_root: str|Path) -> tuple[list[dict[str,Any]],dict[str,dict[str,Any]]]:
    labels={row["track_id"]:row for row in _read_csv(labels_csv)}
    mappings={row["ground_truth_track_id"]:row for row in _read_csv(mapping_csv)}
    grouped:dict[str,list[dict[str,str]]]=defaultdict(list)
    for row in _read_csv(matches_csv): grouped[row["labelled_track_id"]].append(row)
    diagnostics=diagnostic_index(diagnostics_root);items=[]
    for track_id in PROBLEM_TRACKS:
        if track_id not in labels or track_id not in mappings: raise ValueError(f"Missing problematic track input: {track_id}")
        issue="EXTRACTION_FAILURE" if track_id in {"A012","A020","A021","A023"} else ("EXTREME_GEOMETRY" if track_id=="E025" else "DUPLICATE_CANDIDATE")
        alternatives=sorted(grouped[track_id],key=lambda row:int(row["rank"]))[:3]
        items.append({"annotation":labels[track_id],"mapping":mappings[track_id],"alternatives":alternatives,"issue_type":issue})
    return items,diagnostics


def write_png(path: Path,image: np.ndarray) -> None:
    path.parent.mkdir(parents=True,exist_ok=True);ok,encoded=cv2.imencode(".png",image)
    if not ok: raise OSError(f"Could not encode {path}")
    encoded.tofile(str(path))


def render_all_diagnostics(labels_csv: str|Path,mapping_csv: str|Path,matches_csv: str|Path,
                           diagnostics_root: str|Path,source_roots:Sequence[str|Path],crop_root:str|Path,
                           output_dir:str|Path) -> dict[str,dict[str,Any]]:
    items,diagnostics=build_adjudication_context(labels_csv,mapping_csv,matches_csv,diagnostics_root)
    sources=exact_file_index(source_roots,{item["annotation"]["source_image"] for item in items})
    crops=exact_file_index([crop_root],{item["annotation"]["crop_filename"] for item in items})
    output=Path(output_dir);metrics={};by_id={item["annotation"]["track_id"]:item for item in items}
    for item in items:
        annotation=item["annotation"];track_id=annotation["track_id"]
        image,track_metrics=render_track_diagnostic(item,_decode(sources.get(annotation["source_image"])),_decode(crops.get(annotation["crop_filename"])),diagnostics)
        track_metrics["_module2_track_id"] = item["mapping"]["module2_track_id"]
        write_png(output/f"{track_id}_diagnostic.png",image);metrics[track_id]=track_metrics
    for first,second in DUPLICATE_PAIRS:
        pair=[by_id[first],by_id[second]];source=_decode(sources.get(pair[0]["annotation"]["source_image"]))
        if source is None: raise OSError(f"Missing duplicate-pair source for {first}/{second}")
        pair_crops=[_decode(crops.get(item["annotation"]["crop_filename"])) for item in pair]
        write_png(output/f"{first}_{second}_duplicate_pair.png",render_duplicate_pair(pair,source,pair_crops,diagnostics))
    return metrics


def write_adjudication(output_csv:str|Path,report_txt:str|Path,rows:Sequence[Mapping[str,Any]],report:str)->None:
    for row in rows:
        if row.get("recommended_action") not in ALLOWED_ACTIONS: raise ValueError(f"Invalid recommended action for {row.get('ground_truth_track_id')}")
    _atomic_csv(output_csv,rows);_atomic_text(report_txt,report)


def _metric_summary(metrics: Mapping[str, Any]) -> str:
    """Return the compact geometric evidence used by the adjudication report."""
    return (
        f"area={metrics['mask_area_pixels']}; components={metrics['foreground_component_count']}; "
        f"skeleton_endpoints={metrics['skeleton_endpoint_count']}; "
        f"branch_pixels={metrics['skeleton_branch_pixel_count']}; "
        f"candidate_mask_inside_manual_roi={metrics['mask_fraction_inside_manual_roi']:.6f}; "
        f"manual_roi_foreground_fraction={metrics['manual_roi_foreground_fraction']:.6f}"
    )


def build_adjudication_outputs(
    metrics: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, str]], str]:
    """Create evidence-based recommendations without changing mapping decisions.

    The conclusions in this function are deliberately scoped to the nine named
    records and to the visual panels produced from the current immutable inputs.
    They are recommendations for a later human mapping revision, not mutations.
    """
    missing = sorted(set(PROBLEM_TRACKS) - set(metrics))
    if missing:
        raise ValueError(f"Missing diagnostic metrics for: {', '.join(missing)}")

    current_ids = {
        track_id: str(metrics[track_id]["_module2_track_id"])
        for track_id in PROBLEM_TRACKS
    }

    failure_notes = {
        "A012": (
            "The stored crop is one long Alpha track, but the selected mask contains several adjacent "
            "parallel tracks and background. The target is not isolated. All five ranked alternatives "
            "are merged structures, unrelated regions, or partial track fragments."
        ),
        "A020": (
            "The stored crop is an isolated curved Alpha track, but the selected mask is the chamber "
            "border/background and has no foreground in the manual ROI. Rank 1 contains the track only "
            "inside a much larger background component; ranks 3-5 are small unrelated fragments."
        ),
        "A021": (
            "The stored crop is an isolated straight Alpha track, but the selected mask is the chamber "
            "border/background and has no foreground in the manual polygon. Other ranked candidates "
            "are broad noisy regions or unrelated fragments rather than an isolated track."
        ),
        "A023": (
            "The stored crop is one long Alpha track, while the selected mask is the chamber "
            "border/background with only negligible foreground in the manual polygon. The alternatives "
            "are background merges, a partial neighbouring structure, or unrelated fragments."
        ),
    }

    rows: list[dict[str, str]] = []
    for track_id in ("A012", "A020", "A021", "A023"):
        rows.append({
            "ground_truth_track_id": track_id,
            "issue_type": "EXTRACTION_FAILURE",
            "current_module2_track_id": current_ids[track_id],
            "diagnosis": (
                "MAPPING_REVIEW_REQUIRED: wrong selected candidate containing merged tracks and/or "
                "background (causes A, B, and C); not a Module 3 geometry failure on a valid mask."
            ),
            "recommended_action": "EXCLUDE_NO_VALID_MODULE2_CANDIDATE",
            "manual_reselection_needed": "NO",
            "geometry_review_needed": "NO",
            "notes": f"{_metric_summary(metrics[track_id])}. {failure_notes[track_id]}",
        })

    for track_id, partner in (("A003", "A008"), ("A008", "A003")):
        rows.append({
            "ground_truth_track_id": track_id,
            "issue_type": "DUPLICATE_CANDIDATE",
            "current_module2_track_id": current_ids[track_id],
            "diagnosis": (
                "The two ground-truth records are near-identical annotations of the same physical "
                "Alpha track (cause D); the shared Module 2 candidate spatially represents that track."
            ),
            "recommended_action": "REVIEW_GROUND_TRUTH",
            "manual_reselection_needed": "NO",
            "geometry_review_needed": "NO",
            "notes": (
                f"Paired with {partner}. The crops and manual polygons coincide visually. "
                f"{_metric_summary(metrics[track_id])}. Keep at most one as an independent sample, "
                "but this adjudication does not choose or remove either record."
            ),
        })

    for track_id, partner in (("A025", "A028"), ("A028", "A025")):
        rows.append({
            "ground_truth_track_id": track_id,
            "issue_type": "DUPLICATE_CANDIDATE",
            "current_module2_track_id": current_ids[track_id],
            "diagnosis": (
                "The two ground-truth records are near-identical annotations of the same physical "
                "Alpha track (cause D), and their shared selected candidate is also wrong: it is the "
                "chamber-border/background component (causes A and B)."
            ),
            "recommended_action": "REVIEW_GROUND_TRUTH",
            "manual_reselection_needed": "YES",
            "geometry_review_needed": "NO",
            "notes": (
                f"Paired with {partner}. {_metric_summary(metrics[track_id])}. Resolve the duplicate "
                "ground truth first. The reviewed top five candidates do not isolate the complete track, "
                "so the retained annotation should be excluded unless a later wider review finds a valid candidate."
            ),
        })

    e025 = metrics["E025"]
    rows.append({
        "ground_truth_track_id": "E025",
        "issue_type": "EXTREME_GEOMETRY",
        "current_module2_track_id": current_ids["E025"],
        "diagnosis": (
            "MAPPING_REVIEW_REQUIRED: tortuosity is produced by a wrong chamber-border/background mask "
            "with a highly branched, near-loop principal path (causes A and B), not genuine Electron morphology."
        ),
        "recommended_action": "EXCLUDE_NO_VALID_MODULE2_CANDIDATE",
        "manual_reselection_needed": "NO",
        "geometry_review_needed": "NO",
        "notes": (
            f"{_metric_summary(e025)}; principal_path_length_px={e025.get('principal_path_length_px', float('nan')):.6f}; "
            f"principal_endpoint_distance_px={e025.get('principal_endpoint_distance_px', float('nan')):.6f}; "
            f"computed_tortuosity={e025.get('computed_tortuosity', float('nan')):.6f}. The best spatial "
            "alternative covers only about 49% of the manual ROI and is a split/noisy partial candidate; "
            "none of the top five represents the complete labelled track."
        ),
    })

    report_lines = [
        "PRODUCTION MODULE 2 MAPPING ADJUDICATION",
        "========================================",
        "",
        "Scope: diagnostic-only review of A012, A020, A021, A023, A003, A008, A025, A028, and E025.",
        "Evidence: stored crops, original source frames/manual ROIs, current Module 2 masks/overlays,",
        "top-ranked spatial alternatives, and unchanged Module 3 skeleton diagnostics.",
        "No class features or Random Forest predictions influenced candidate assessment.",
        "",
        "EXTRACTION-FAILURE REVIEW",
        "-------------------------",
    ]
    for track_id in ("A012", "A020", "A021", "A023"):
        metric = metrics[track_id]
        report_lines.extend([
            f"{track_id}: MAPPING_REVIEW_REQUIRED",
            f"  {_metric_summary(metric)}",
            f"  principal_path_status={metric['principal_path_status']}; "
            f"principal_path_error={metric['principal_path_error']}",
            "  multiple_physical_tracks_merged=" + ("YES" if track_id == "A012" else "NO CLEAR MULTI-TRACK MERGE"),
            "  background_structures_included=YES",
            "  manually_labelled_track_isolated_in_mask=NO",
            f"  evidence: {failure_notes[track_id]}",
            "  recommendation=EXCLUDE_NO_VALID_MODULE2_CANDIDATE",
            "",
        ])

    report_lines.extend([
        "DUPLICATE-MAPPING REVIEW",
        "------------------------",
        "A003/A008:",
        "  diagnosis=the crops and near-coincident polygons show the same physical track (cause D).",
        "  current_candidate=spatially trustworthy for that physical track, but the two records are not independent.",
        "  recommendation=REVIEW_GROUND_TRUTH; retain at most one independent annotation after human review.",
        "",
        "A025/A028:",
        "  diagnosis=the crops and near-coincident polygons show the same physical track (cause D).",
        "  current_candidate=not trustworthy; it is the chamber-border/background component and has zero mask foreground in either manual polygon.",
        "  recommendation=REVIEW_GROUND_TRUTH first; top-five review found no complete isolated Module 2 candidate.",
        "",
        "E025 EXTREME TORTUOSITY",
        "-----------------------",
        f"  {_metric_summary(e025)}",
        f"  principal_path_length_px={e025.get('principal_path_length_px', float('nan')):.6f}",
        f"  principal_endpoint_distance_px={e025.get('principal_endpoint_distance_px', float('nan')):.6f}",
        f"  computed_tortuosity={e025.get('computed_tortuosity', float('nan')):.6f}",
        "  diagnosis=the selected frame-scale component forms a branched, near-loop route; the principal path",
        "  is very long while its endpoints are close together. This is not evidence of genuine extreme morphology.",
        "  recommendation=EXCLUDE_NO_VALID_MODULE2_CANDIDATE; no top-five candidate covers the complete track.",
        "",
        "ADJUDICATION SUMMARY",
        "--------------------",
        "Current successful unique masks: 31",
        "Expected unique masks after adjudication: 29 (estimate only)",
        "Estimate basis: four existing failures remain excluded; A003/A008 contribute one physical/unique mask;",
        "A025/A028 lose one invalid shared mask; E025 loses one invalid mask; no output data was revised here.",
        "Ready to revise confirmed mapping: YES",
        "Ready to train Random Forest: NO",
        "",
        "SAFETY",
        "------",
        "Module 1 modified: NO",
        "Module 2 modified: NO",
        "Existing Module 3 production code modified: NO",
        "labels.csv modified: NO",
        "manual_review_results.csv modified: NO",
        "confirmed mapping modified: NO",
        "production features regenerated: NO",
        "Random Forest trained: NO",
        "",
    ])
    return rows, "\n".join(report_lines)
