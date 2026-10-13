"""Connected-component analysis, logical trajectory grouping, and filtering."""

from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd

COMPONENT_COLUMNS = [
    "source_filename", "source_stem", "component_id", "area_pixels",
    "centroid_x", "centroid_y", "bbox_x", "bbox_y", "bbox_width",
    "bbox_height", "bbox_long_side", "bbox_short_side", "bbox_x2_exclusive",
    "bbox_y2_exclusive", "touches_boundary",
]

TRAJECTORY_COMPONENT_COLUMNS = COMPONENT_COLUMNS + [
    "source_component_ids", "fragment_count", "trajectory_linked",
    "trajectory_link_count", "trajectory_link_length_pixels",
    "trajectory_mean_link_gap_pixels", "trajectory_mean_local_linearity",
    "trajectory_max_link_alignment_degrees",
    "trajectory_max_tangent_change_degrees", "trajectory_step_ratio",
    "curve_fitted", "curve_model", "curve_score", "curve_contrast", "curve_axis_span_pixels",
    "curve_total_turn_degrees",
]


def analyse_components(mask, source_filename, connectivity=8):
    """Label foreground regions and return component measurements."""
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("analyse_components expects a 2-D uint8 mask.")
    binary = np.where(mask > 0, 1, 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity, ltype=cv2.CV_32S)
    height, width = mask.shape
    rows = []
    source_filename = str(source_filename)
    source_stem = Path(source_filename).stem
    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        box_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        x2, y2 = x + box_width, y + box_height
        rows.append({
            "source_filename": source_filename, "source_stem": source_stem,
            "component_id": int(component_id),
            "area_pixels": int(stats[component_id, cv2.CC_STAT_AREA]),
            "centroid_x": float(centroids[component_id, 0]),
            "centroid_y": float(centroids[component_id, 1]),
            "bbox_x": x, "bbox_y": y, "bbox_width": box_width,
            "bbox_height": box_height, "bbox_long_side": max(box_width, box_height),
            "bbox_short_side": min(box_width, box_height),
            "bbox_x2_exclusive": x2, "bbox_y2_exclusive": y2,
            "touches_boundary": bool(x == 0 or y == 0 or x2 >= width or y2 >= height),
        })
    return labels, pd.DataFrame(rows, columns=COMPONENT_COLUMNS)


def _passes_individual_filters(row, config):
    """Return whether one raw component already passes the base filters."""
    if row["area_pixels"] < config["min_component_area"]:
        return False
    max_area = config["max_component_area"]
    if max_area is not None and row["area_pixels"] > max_area:
        return False
    if row["bbox_long_side"] < config["min_bbox_long_side"]:
        return False
    if row["bbox_short_side"] < config["min_bbox_short_side"]:
        return False
    ratio_range = config["aspect_ratio_range"]
    if ratio_range is not None:
        ratio = row["bbox_long_side"] / max(row["bbox_short_side"], 1)
        if not (ratio_range[0] <= ratio <= ratio_range[1]):
            return False
    return True


def _singleton_trajectory_table(components):
    """Add no-link provenance fields while preserving raw component IDs exactly."""
    grouped = components.copy()
    grouped["source_component_ids"] = grouped["component_id"].astype(str)
    grouped["fragment_count"] = 1
    grouped["trajectory_linked"] = False
    grouped["trajectory_link_count"] = 0
    grouped["trajectory_link_length_pixels"] = 0.0
    grouped["trajectory_mean_link_gap_pixels"] = 0.0
    grouped["trajectory_mean_local_linearity"] = np.nan
    grouped["trajectory_max_link_alignment_degrees"] = 0.0
    grouped["trajectory_max_tangent_change_degrees"] = 0.0
    grouped["trajectory_step_ratio"] = 1.0
    grouped["curve_fitted"] = False
    grouped["curve_model"] = "none"
    grouped["curve_score"] = 0.0
    grouped["curve_contrast"] = 0.0
    grouped["curve_axis_span_pixels"] = 0.0
    grouped["curve_total_turn_degrees"] = 0.0
    return grouped.reindex(columns=TRAJECTORY_COMPONENT_COLUMNS)


def _curve_candidates(labels, work, config, protected):
    """Find smooth fragmented tracks using compatible Hough seeds and a quadratic fit."""
    if not config.get("curve_linking_enabled", False) or work.empty:
        return []
    areas = work["area_pixels"].to_numpy(dtype=float)
    eligible = (
        (areas >= int(config["curve_min_fragment_area"]))
        & (areas <= int(config["curve_max_fragment_area"]))
        & ~protected
    )
    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) < int(config["curve_min_components"]):
        return []
    coordinates = work[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    points = coordinates[eligible_indices]
    point_areas = areas[eligible_indices]
    mask = np.where(labels > 0, 255, 0).astype(np.uint8)
    lines = cv2.HoughLinesP(
        mask, 1, np.pi / 180.0,
        threshold=int(config["curve_hough_threshold"]),
        minLineLength=int(config["curve_hough_min_line_length"]),
        maxLineGap=int(config["curve_hough_max_line_gap"]),
    )
    if lines is None:
        return []
    lines = np.asarray(lines, dtype=float).reshape(-1, 4)
    lengths = np.linalg.norm(lines[:, 2:4] - lines[:, 0:2], axis=1)
    lines = lines[np.argsort(lengths)[::-1]][:int(config["curve_max_hough_lines"])]
    raw_candidates = []
    for first_index, first_line in enumerate(lines):
        first = first_line.reshape(2, 2)
        first_vector = first[1] - first[0]
        for second_line in lines[first_index + 1:]:
            second = second_line.reshape(2, 2)
            second_vector = second[1] - second[0]
            axis = 0 if (
                abs(first_vector[0]) + abs(second_vector[0])
                >= abs(first_vector[1]) + abs(second_vector[1])) else 1
            other_axis = 1 - axis
            if (abs(first_vector[axis]) < abs(first_vector[other_axis])
                    or abs(second_vector[axis]) < abs(second_vector[other_axis])):
                continue
            seed_points = np.vstack([first, second])
            seed_u = seed_points[:, axis]
            seed_v = seed_points[:, other_axis]
            if float(np.ptp(seed_u)) < float(config["curve_min_axis_span"]):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coefficients = np.polyfit(seed_u, seed_v, 2)
            seed_fit = np.polyval(coefficients, seed_u)
            seed_rmse = float(np.sqrt(np.mean(np.square(seed_v - seed_fit))))
            if seed_rmse > float(config["curve_max_seed_fit_rmse"]):
                continue
            compatible = True
            for segment, vector in ((first, first_vector), (second, second_vector)):
                midpoint = float(segment[:, axis].mean())
                derivative = float(2.0 * coefficients[0] * midpoint + coefficients[1])
                fitted_vector = (
                    np.array([1.0, derivative]) if axis == 0
                    else np.array([derivative, 1.0]))
                actual_vector = vector if vector[axis] >= 0 else -vector
                cosine = abs(float(np.dot(fitted_vector, actual_vector))) / (
                    np.linalg.norm(fitted_vector) * np.linalg.norm(actual_vector))
                angle = float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))
                if angle > float(config["curve_max_seed_tangent_error_degrees"]):
                    compatible = False
                    break
            if not compatible:
                continue
            u = points[:, axis]
            v = points[:, other_axis]
            derivative = 2.0 * coefficients[0] * u + coefficients[1]
            residual = np.abs(v - np.polyval(coefficients, u)) / np.sqrt(1.0 + derivative ** 2)
            seed_low = float(seed_u.min()) - 20.0
            seed_high = float(seed_u.max()) + 20.0
            member_local = np.flatnonzero(
                (u >= seed_low) & (u <= seed_high)
                & (residual <= float(config["curve_residual_tolerance"])))
            member_local = member_local[np.argsort(u[member_local])]
            if member_local.size == 0:
                continue
            cuts = np.where(
                np.diff(u[member_local]) > float(config["curve_max_component_gap"]))[0] + 1
            for run in np.split(member_local, cuts):
                if len(run) < int(config["curve_min_components"]):
                    continue
                axis_span = float(np.ptp(u[run]))
                if axis_span < float(config["curve_min_axis_span"]):
                    continue
                bin_size = float(config["curve_bin_size"])
                occupied_bins = len(np.unique(np.floor(u[run] / bin_size).astype(int)))
                expected_bins = max(1, int(np.ceil(axis_span / bin_size)))
                coverage = min(1.0, occupied_bins / expected_bins)
                if coverage < float(config["curve_min_bin_coverage"]):
                    continue
                run_low, run_high = float(u[run].min()), float(u[run].max())
                endpoint_slopes = (
                    2.0 * coefficients[0] * np.array([run_low, run_high]) + coefficients[1])
                total_turn = abs(float(np.degrees(
                    np.arctan(endpoint_slopes[1]) - np.arctan(endpoint_slopes[0]))))
                if total_turn < float(config["curve_min_total_turn_degrees"]):
                    continue
                within_run = (u >= run_low) & (u <= run_high)
                weights = np.sqrt(point_areas)
                narrow_weight = float(weights[
                    within_run & (residual <= float(config["curve_residual_tolerance"]))].sum())
                wide_weight = float(weights[
                    within_run & (residual <= float(config["curve_context_radius"]))].sum())
                contrast = narrow_weight / max(wide_weight, 1e-9)
                if contrast < float(config["curve_min_contrast"]):
                    continue
                run_residual = residual[run]
                support = float(np.sum(
                    weights[run] * np.maximum(
                        0.0, 1.0 - run_residual
                        / (float(config["curve_residual_tolerance"]) + 2.0))))
                base_score = (
                    axis_span + 12.0 * occupied_bins + 6.0 * support - 2.0 * len(run))
                curve_score = float(base_score * contrast)
                if curve_score < float(config["curve_min_score"]):
                    continue
                raw_candidates.append({
                    "model": "quadratic",
                    "score": curve_score,
                    "selection_score": curve_score * contrast,
                    "contrast": float(contrast),
                    "axis_span": axis_span,
                    "total_turn_degrees": total_turn,
                    "axis": axis,
                    "coefficients": coefficients,
                    "member_indices": eligible_indices[run].astype(int),
                    "ordered_member_indices": eligible_indices[run].astype(int),
                })
    selected = []
    selected_members = []
    for candidate in sorted(
            raw_candidates, key=lambda item: item["selection_score"], reverse=True):
        members = set(map(int, candidate["member_indices"]))
        duplicate = any(
            len(members & previous) / max(1, min(len(members), len(previous)))
            > float(config["curve_max_member_overlap"])
            for previous in selected_members)
        if duplicate:
            continue
        selected.append(candidate)
        selected_members.append(members)
        if len(selected) >= int(config["curve_max_candidates"]):
            break
    return selected


def _point_to_polyline_distances(points, path_points):
    """Return each point's shortest Euclidean distance to a polyline."""
    distances = np.full(len(points), np.inf, dtype=float)
    for start, stop in zip(path_points[:-1], path_points[1:]):
        segment = stop - start
        squared_length = float(segment @ segment)
        if squared_length <= 1e-9:
            continue
        projection = np.clip(((points - start) @ segment) / squared_length, 0.0, 1.0)
        closest = start + projection[:, None] * segment
        distances = np.minimum(distances, np.linalg.norm(points - closest, axis=1))
    return distances


def _polyline_candidates(labels, work, config):
    """Trace locally smooth droplet paths from short Hough-line seeds.

    Unlike one global polynomial/circle fit, this model updates its direction
    as it grows.  It can therefore follow asymmetric curves and is less likely
    to switch from one particle to another where two tracks cross.
    """
    if not config.get("polyline_linking_enabled", False) or work.empty:
        return []
    areas = work["area_pixels"].to_numpy(dtype=float)
    eligible = (
        (areas >= int(config["polyline_min_fragment_area"]))
        & (areas <= int(config["polyline_max_fragment_area"])))
    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) < int(config["polyline_min_components"]):
        return []
    points = work.loc[eligible, ["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    point_areas = areas[eligible_indices]
    mask = np.where(labels > 0, 255, 0).astype(np.uint8)
    lines = cv2.HoughLinesP(
        mask, 1, np.pi / 180.0,
        threshold=int(config["polyline_hough_threshold"]),
        minLineLength=int(config["polyline_hough_min_line_length"]),
        maxLineGap=int(config["polyline_hough_max_line_gap"]),
    )
    if lines is None:
        return []
    lines = np.asarray(lines, dtype=float).reshape(-1, 4)
    lengths = np.linalg.norm(lines[:, 2:4] - lines[:, 0:2], axis=1)
    lines = lines[np.argsort(lengths)[::-1]][:int(config["polyline_max_hough_lines"])]
    maximum_step = float(config["polyline_max_step"])
    maximum_turn = float(config["polyline_max_turn_degrees"])
    minimum_cosine = float(np.cos(np.radians(maximum_turn)))
    history_size = int(config["polyline_direction_history"])
    raw_candidates = []

    for line in lines:
        start, stop = line.reshape(2, 2)
        vector = stop - start
        line_length = float(np.linalg.norm(vector))
        if line_length <= 1e-9:
            continue
        direction = vector / line_length
        relative = points - start
        projection = relative @ direction
        perpendicular = np.abs(
            relative[:, 0] * direction[1] - relative[:, 1] * direction[0])
        seed_local = np.flatnonzero(
            (projection >= -float(config["polyline_seed_extension"]))
            & (projection <= line_length + float(config["polyline_seed_extension"]))
            & (perpendicular <= float(config["polyline_seed_tube_radius"])))
        if len(seed_local) < int(config["polyline_min_seed_components"]):
            continue
        seed_local = seed_local[np.argsort(projection[seed_local], kind="stable")]
        cuts = np.where(np.diff(projection[seed_local]) > maximum_step)[0] + 1
        seed_runs = [run for run in np.split(seed_local, cuts)
                     if len(run) >= int(config["polyline_min_seed_components"])]
        if not seed_runs:
            continue
        path = list(map(int, max(
            seed_runs,
            key=lambda run: float(np.ptp(projection[run])) + 3.0 * len(run))))
        used = set(path)

        def grow_one_end(at_end):
            changed = False
            while True:
                ordered = path if at_end else path[::-1]
                endpoint = points[ordered[-1]]
                anchor_position = max(0, len(ordered) - 1 - history_size)
                outward = endpoint - points[ordered[anchor_position]]
                norm = float(np.linalg.norm(outward))
                if norm <= 1e-9:
                    break
                outward /= norm
                offsets = points - endpoint
                gaps = np.linalg.norm(offsets, axis=1)
                forward = offsets @ outward
                cosine = forward / np.maximum(gaps, 1e-9)
                candidates = np.flatnonzero(
                    (gaps > 1e-6) & (gaps <= maximum_step)
                    & (forward > 0.0) & (cosine >= minimum_cosine))
                candidates = [int(index) for index in candidates if int(index) not in used]
                if not candidates:
                    break
                def continuation_cost(index):
                    angle = float(np.degrees(np.arccos(np.clip(cosine[index], -1.0, 1.0))))
                    return (float(gaps[index]) * (1.0 + angle / maximum_turn)
                            - 1.5 * np.sqrt(point_areas[index]))
                chosen = min(candidates, key=continuation_cost)
                if at_end:
                    path.append(chosen)
                else:
                    path.insert(0, chosen)
                used.add(chosen)
                changed = True
                if len(path) >= 300:
                    break
            return changed

        grow_one_end(True)
        grow_one_end(False)
        if len(path) < int(config["polyline_min_components"]):
            continue
        path_local = np.asarray(path, dtype=int)
        path_points = points[path_local]
        steps = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
        path_length = float(steps.sum())
        if path_length < float(config["polyline_min_length"]):
            continue
        cumulative = np.concatenate([[0.0], np.cumsum(steps)])
        bin_size = float(config["polyline_bin_size"])
        occupied_bins = len(np.unique(np.floor(cumulative / bin_size).astype(int)))
        expected_bins = max(1, int(np.ceil(path_length / bin_size)))
        coverage = min(1.0, occupied_bins / expected_bins)
        if coverage < float(config["polyline_min_bin_coverage"]):
            continue
        distances = _point_to_polyline_distances(points, path_points)
        weights = np.sqrt(point_areas)
        narrow_weight = float(weights[
            distances <= float(config["polyline_residual_tolerance"])].sum())
        context_weight = float(weights[
            distances <= float(config["polyline_context_radius"])].sum())
        contrast = narrow_weight / max(context_weight, 1e-9)
        if contrast < float(config["polyline_min_contrast"]):
            continue
        segment_vectors = np.diff(path_points, axis=0)
        segment_vectors /= np.maximum(np.linalg.norm(segment_vectors, axis=1)[:, None], 1e-9)
        if len(segment_vectors) >= 2:
            turn_cosines = np.sum(segment_vectors[:-1] * segment_vectors[1:], axis=1)
            total_turn = float(np.degrees(np.arccos(np.clip(turn_cosines, -1.0, 1.0))).sum())
        else:
            total_turn = 0.0
        turn_density = total_turn / max(path_length, 1e-9)
        if (turn_density > float(config["polyline_max_low_contrast_turn_density"])
                and contrast < float(config["polyline_high_turn_min_contrast"])):
            continue
        support = float(np.sqrt(point_areas[path_local]).sum())
        score = float((path_length + 12.0 * occupied_bins + 5.0 * support) * contrast)
        if score < float(config["polyline_min_score"]):
            continue
        selection_score = max(
            0.0,
            score - float(config["polyline_turn_penalty"]) * total_turn,
        ) * contrast
        raw_candidates.append({
            "model": "smooth_polyline", "score": score,
            "selection_score": float(selection_score), "contrast": contrast,
            "axis_span": path_length, "total_turn_degrees": total_turn,
            "axis": 0, "member_indices": eligible_indices[path_local].astype(int),
            "ordered_member_indices": eligible_indices[path_local].astype(int),
        })

    selected, member_sets = [], []
    for candidate in sorted(
            raw_candidates, key=lambda item: item["selection_score"], reverse=True):
        members = set(map(int, candidate["member_indices"]))
        duplicate = any(
            len(members & previous) / max(1, min(len(members), len(previous)))
            > float(config["curve_max_member_overlap"])
            for previous in member_sets)
        if duplicate:
            continue
        selected.append(candidate)
        member_sets.append(members)
        if len(selected) >= int(config["polyline_max_candidates"]):
            break
    return selected


def _join_polyline_fragments(base_candidates, extension_candidates, work, config):
    """Join a short end fragment that continues a confident polyline."""
    if not config.get("polyline_fragment_extension_enabled", False):
        return base_candidates
    coordinates = work[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    joined = []
    for base in base_candidates:
        if float(base["contrast"]) < float(config["polyline_fragment_min_base_contrast"]):
            joined.append(base)
            continue
        base_members = set(map(int, base["member_indices"]))
        base_order = np.asarray(base["ordered_member_indices"], dtype=int)
        base_points = coordinates[base_order]
        base_low, base_high = base_points.min(axis=0), base_points.max(axis=0)
        base_size = np.maximum(base_high - base_low, 1.0)
        options = []
        for extension in extension_candidates:
            extension_members = set(map(int, extension["member_indices"]))
            if base_members & extension_members:
                continue
            extension_order = np.asarray(
                extension["ordered_member_indices"], dtype=int)
            extension_points = coordinates[extension_order]
            endpoint_distances = np.linalg.norm(
                base_points[[0, -1], None, :]
                - extension_points[None, [0, -1], :],
                axis=2,
            )
            base_end, extension_end = np.unravel_index(
                int(np.argmin(endpoint_distances)), endpoint_distances.shape)
            endpoint_gap = float(endpoint_distances[base_end, extension_end])
            if endpoint_gap > float(config["polyline_fragment_max_endpoint_gap"]):
                continue

            direction_history = max(1, int(config["polyline_direction_history"]))
            base_inner = (
                min(direction_history, len(base_points) - 1)
                if base_end == 0 else
                max(0, len(base_points) - 1 - direction_history))
            extension_inner = (
                min(direction_history, len(extension_points) - 1)
                if extension_end == 0 else
                max(0, len(extension_points) - 1 - direction_history))
            base_direction = (
                base_points[base_end * (len(base_points) - 1)]
                - base_points[base_inner])
            extension_direction = (
                extension_points[extension_end * (len(extension_points) - 1)]
                - extension_points[extension_inner])
            direction_norm = float(
                np.linalg.norm(base_direction) * np.linalg.norm(extension_direction))
            if direction_norm <= 1e-9:
                continue
            join_turn = float(np.degrees(np.arccos(np.clip(
                -(base_direction @ extension_direction) / direction_norm,
                -1.0, 1.0))))
            if join_turn > float(config["polyline_fragment_max_join_turn_degrees"]):
                continue

            low, high = extension_points.min(axis=0), extension_points.max(axis=0)
            size = np.maximum(high - low, 1.0)
            for axis in (0, 1):
                other_axis = 1 - axis
                axis_gap = max(
                    float(low[axis] - base_high[axis]),
                    float(base_low[axis] - high[axis]),
                    0.0,
                )
                if axis_gap > float(config["polyline_fragment_max_axis_gap"]):
                    continue
                orthogonal_overlap = max(
                    0.0,
                    min(base_high[other_axis], high[other_axis])
                    - max(base_low[other_axis], low[other_axis]),
                ) / max(1.0, min(base_size[other_axis], size[other_axis]))
                if orthogonal_overlap < float(
                        config["polyline_fragment_min_orthogonal_overlap"]):
                    continue
                axis_extension = max(
                    float(base_low[axis] - low[axis]),
                    float(high[axis] - base_high[axis]),
                    0.0,
                )
                if axis_extension < float(config["polyline_fragment_min_axis_extension"]):
                    continue
                option_score = (
                    float(extension["selection_score"]) * orthogonal_overlap
                    / (1.0 + endpoint_gap))
                options.append((
                    option_score, extension, extension_order,
                    base_end, extension_end, endpoint_gap, join_turn))
        if not options:
            joined.append(base)
            continue
        (_, extension, extension_order, base_end, extension_end,
         endpoint_gap, join_turn) = max(options, key=lambda item: item[0])
        if base_end == 0:
            base_order = base_order[::-1]
        if extension_end == 1:
            extension_order = extension_order[::-1]
        combined_order = np.concatenate([base_order, extension_order])
        combined = base.copy()
        combined["model"] = "extended_polyline"
        combined["score"] = float(base["score"] + extension["score"])
        combined["selection_score"] = float(
            base["selection_score"] + extension["selection_score"])
        combined["axis_span"] = float(
            base["axis_span"] + extension["axis_span"]
            + endpoint_gap)
        combined["total_turn_degrees"] = float(
            base["total_turn_degrees"] + extension["total_turn_degrees"]
            + join_turn)
        combined["contrast"] = float(
            (base["contrast"] * base["axis_span"]
             + extension["contrast"] * extension["axis_span"])
            / max(base["axis_span"] + extension["axis_span"], 1e-9))
        combined["ordered_member_indices"] = combined_order.astype(int)
        combined["member_indices"] = np.unique(combined_order).astype(int)
        joined.append(combined)
    return joined


def _beam_polyline_candidates(labels, work, config):
    """Rescue smooth tracks when greedy polyline growth finds none.

    Hough seeds in dense chamber noise can have several plausible next
    fragments. This bounded search preserves a few alternatives and ranks the
    resulting paths by support, endpoint efficiency, and accumulated turning.
    """
    if not config.get("beam_polyline_enabled", False) or work.empty:
        return []
    areas = work["area_pixels"].to_numpy(dtype=float)
    eligible = (
        (areas >= int(config["polyline_min_fragment_area"]))
        & (areas <= int(config["polyline_max_fragment_area"])))
    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) < int(config["polyline_min_components"]):
        return []
    points = work.loc[eligible, ["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    point_areas = areas[eligible_indices]
    weights = np.sqrt(point_areas)
    mask = np.where(labels > 0, 255, 0).astype(np.uint8)
    raw_lines = cv2.HoughLinesP(
        mask, 1, np.pi / 180.0,
        threshold=int(config["polyline_hough_threshold"]),
        minLineLength=int(config["polyline_hough_min_line_length"]),
        maxLineGap=int(config["polyline_hough_max_line_gap"]),
    )
    if raw_lines is None:
        return []
    lines = np.asarray(raw_lines, dtype=float).reshape(-1, 4)

    seed_extension = float(config["polyline_seed_extension"])
    seed_radius = float(config["polyline_seed_tube_radius"])
    context_radius = float(config["polyline_context_radius"])
    line_scores = []
    for line in lines:
        start, stop = line.reshape(2, 2)
        vector = stop - start
        length = float(np.linalg.norm(vector))
        direction = vector / max(length, 1e-9)
        relative = points - start
        projection = relative @ direction
        perpendicular = np.abs(
            relative[:, 0] * direction[1] - relative[:, 1] * direction[0])
        within_axis = (
            (projection >= -seed_extension)
            & (projection <= length + seed_extension))
        narrow = within_axis & (perpendicular <= seed_radius)
        context = within_axis & (perpendicular <= context_radius)
        narrow_weight = float(weights[narrow].sum())
        contrast = narrow_weight / max(float(weights[context].sum()), 1e-9)
        line_scores.append(length * contrast + 5.0 * narrow_weight)
    lines = lines[np.argsort(line_scores)[::-1]][
        :int(config["beam_polyline_hough_seeds"])]

    pair_offsets = points[:, None, :] - points[None, :, :]
    pair_gaps = np.linalg.norm(pair_offsets, axis=2)
    maximum_step = float(config["beam_polyline_max_step"])
    neighbors = [
        np.flatnonzero((row > 1e-6) & (row <= maximum_step))
        for row in pair_gaps
    ]
    maximum_turn = float(config["beam_polyline_max_turn_degrees"])
    minimum_cosine = float(np.cos(np.radians(maximum_turn)))
    history_size = int(config["polyline_direction_history"])
    checkpoints = set(map(int, config["beam_polyline_checkpoint_steps"]))
    generated = []

    for line in lines:
        start, stop = line.reshape(2, 2)
        vector = stop - start
        line_length = float(np.linalg.norm(vector))
        if line_length <= 1e-9:
            continue
        direction = vector / line_length
        relative = points - start
        projection = relative @ direction
        perpendicular = np.abs(
            relative[:, 0] * direction[1] - relative[:, 1] * direction[0])
        seed = np.flatnonzero(
            (projection >= -seed_extension)
            & (projection <= line_length + seed_extension)
            & (perpendicular <= seed_radius))
        if len(seed) < int(config["polyline_min_seed_components"]):
            continue
        seed = seed[np.argsort(projection[seed], kind="stable")]
        cuts = np.where(
            np.diff(projection[seed]) > float(config["polyline_max_step"]))[0] + 1
        runs = [
            run for run in np.split(seed, cuts)
            if len(run) >= int(config["polyline_min_seed_components"])
        ]
        if not runs:
            continue
        initial = tuple(map(int, max(
            runs,
            key=lambda run: float(np.ptp(projection[run])) + 3.0 * len(run))))
        states = [(0.0, initial)]
        completed = []

        for depth in range(int(config["beam_polyline_growth_steps"])):
            expanded = []
            for cost, path in states:
                used = set(path)
                found_continuation = False
                for at_end in (False, True):
                    ordered = path if at_end else path[::-1]
                    endpoint_index = ordered[-1]
                    endpoint = points[endpoint_index]
                    anchor_index = ordered[max(0, len(ordered) - 1 - history_size)]
                    outward = endpoint - points[anchor_index]
                    norm = float(np.linalg.norm(outward))
                    if norm <= 1e-9:
                        continue
                    outward /= norm
                    options = []
                    for index in neighbors[endpoint_index]:
                        index = int(index)
                        if index in used:
                            continue
                        offset = points[index] - endpoint
                        gap = float(np.linalg.norm(offset))
                        forward = float(offset @ outward)
                        cosine = forward / max(gap, 1e-9)
                        if forward <= 0.0 or cosine < minimum_cosine:
                            continue
                        angle = float(np.degrees(np.arccos(
                            np.clip(cosine, -1.0, 1.0))))
                        transition_cost = (
                            gap * (1.0 + angle / maximum_turn)
                            - 1.5 * np.sqrt(point_areas[index]))
                        options.append((transition_cost, index))
                    for transition_cost, index in sorted(options)[
                            :int(config["beam_polyline_branch_factor"])]:
                        found_continuation = True
                        new_path = path + (index,) if at_end else (index,) + path
                        expanded.append((cost + transition_cost, new_path))
                if not found_continuation:
                    completed.append((cost, path))
            if not expanded:
                break
            deduplicated = {}
            for cost, path in sorted(expanded, key=lambda item: item[0]):
                key = (path[0], path[-1], len(path) // 4)
                if key not in deduplicated:
                    deduplicated[key] = (cost, path)
            states = list(deduplicated.values())[:int(config["beam_polyline_width"])]
            if depth in checkpoints:
                completed.extend(states)
        completed.extend(states)

        for _, path in completed:
            if len(path) < int(config["beam_polyline_min_components"]):
                continue
            path_local = np.asarray(path, dtype=int)
            path_points = points[path_local]
            segment_vectors = np.diff(path_points, axis=0)
            segment_lengths = np.linalg.norm(segment_vectors, axis=1)
            path_length = float(segment_lengths.sum())
            if path_length < float(config["beam_polyline_min_length"]):
                continue
            distances = _point_to_polyline_distances(points, path_points)
            narrow_weight = float(weights[
                distances <= float(config["polyline_residual_tolerance"])].sum())
            context_weight = float(weights[distances <= context_radius].sum())
            contrast = narrow_weight / max(context_weight, 1e-9)
            if contrast < float(config["beam_polyline_min_contrast"]):
                continue
            unit_vectors = segment_vectors / np.maximum(
                segment_lengths[:, None], 1e-9)
            total_turn = (
                float(np.degrees(np.arccos(np.clip(
                    np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1),
                    -1.0, 1.0))).sum())
                if len(unit_vectors) >= 2 else 0.0)
            displacement = float(np.linalg.norm(path_points[-1] - path_points[0]))
            efficiency = displacement / max(path_length, 1e-9)
            if efficiency < float(config["beam_polyline_min_efficiency"]):
                continue
            support = float(weights[path_local].sum())
            score = float((path_length + 5.0 * support) * contrast)
            quality = float(
                score * efficiency
                / (1.0 + total_turn / float(config["beam_polyline_turn_scale"])))
            if quality < float(config["beam_polyline_min_quality"]):
                continue
            generated.append({
                "model": "beam_polyline", "score": score,
                "selection_score": (
                    quality * float(config["beam_polyline_selection_multiplier"])),
                "contrast": contrast, "axis_span": path_length,
                "total_turn_degrees": total_turn, "axis": 0,
                "member_indices": eligible_indices[path_local].astype(int),
                "ordered_member_indices": eligible_indices[path_local].astype(int),
            })

    selected, selected_members, selected_boxes = [], [], []
    for candidate in sorted(
            generated, key=lambda item: item["selection_score"], reverse=True):
        members = set(map(int, candidate["member_indices"]))
        if any(
                len(members & previous) / max(1, min(len(members), len(previous)))
                > float(config["curve_max_member_overlap"])
                for previous in selected_members):
            continue
        member_points = points[np.asarray([
            np.flatnonzero(eligible_indices == index)[0] for index in sorted(members)
        ], dtype=int)]
        low, high = member_points.min(axis=0), member_points.max(axis=0)
        candidate_box = np.concatenate([low, high])
        candidate_area = float(np.prod(np.maximum(high - low, 1.0)))
        spatial_duplicate = False
        for previous_box in selected_boxes:
            intersection = np.maximum(
                0.0,
                np.minimum(high, previous_box[2:]) - np.maximum(low, previous_box[:2]),
            )
            intersection_area = float(np.prod(intersection))
            previous_area = float(np.prod(np.maximum(
                previous_box[2:] - previous_box[:2], 1.0)))
            if (intersection_area / max(1.0, min(candidate_area, previous_area))
                    > float(config["beam_polyline_max_bbox_overlap"])):
                spatial_duplicate = True
                break
        if spatial_duplicate:
            continue
        selected.append(candidate)
        selected_members.append(members)
        selected_boxes.append(candidate_box)
        if len(selected) >= int(config["beam_polyline_max_candidates"]):
            break
    return selected


def _fit_circle_least_squares(points):
    """Return the least-squares circle centre and radius for 2-D points."""
    matrix = np.column_stack((2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))))
    squared_radius = np.sum(points * points, axis=1)
    centre_x, centre_y, constant = np.linalg.lstsq(matrix, squared_radius, rcond=None)[0]
    radius_squared = constant + centre_x ** 2 + centre_y ** 2
    if not np.isfinite(radius_squared) or radius_squared <= 0:
        return None
    return np.array([centre_x, centre_y], dtype=float), float(np.sqrt(radius_squared))


def _arc_windows(points, point_areas, centre, radius, config):
    """Score continuous angular windows supported by components around one circle."""
    if not float(config["arc_min_radius"]) <= radius <= float(config["arc_max_radius"]):
        return []
    vectors = points - centre
    angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2.0 * np.pi)
    residual = np.abs(np.linalg.norm(vectors, axis=1) - radius)
    inliers = np.flatnonzero(residual <= float(config["arc_residual_tolerance"]))
    if len(inliers) < int(config["arc_min_components"]):
        return []
    ordered = inliers[np.argsort(angles[inliers])]
    extended_indices = np.concatenate([ordered, ordered])
    extended_angles = np.concatenate([angles[ordered], angles[ordered] + 2.0 * np.pi])
    weights = np.sqrt(point_areas)
    all_order = np.argsort(angles)
    all_angles = np.concatenate([angles[all_order], angles[all_order] + 2.0 * np.pi])
    narrow_values = np.concatenate([
        weights[all_order] * (residual[all_order] <= float(config["arc_residual_tolerance"])),
        weights[all_order] * (residual[all_order] <= float(config["arc_residual_tolerance"]))])
    context_values = np.concatenate([
        weights[all_order] * (residual[all_order] <= float(config["arc_context_radius"])),
        weights[all_order] * (residual[all_order] <= float(config["arc_context_radius"]))])
    narrow_prefix = np.concatenate([[0.0], np.cumsum(narrow_values)])
    context_prefix = np.concatenate([[0.0], np.cumsum(context_values)])
    candidates = []
    minimum = int(config["arc_min_components"])
    maximum = int(config["arc_max_components"])
    for start in range(len(ordered)):
        final_stop = min(start + maximum - 1, start + len(ordered) - 1)
        largest_gap = 0.0
        for stop in range(start + minimum - 1, final_stop + 1):
            run = extended_indices[start:stop + 1]
            run_angles = extended_angles[start:stop + 1]
            if stop == start + minimum - 1:
                largest_gap = float(np.max(np.diff(run_angles) * radius))
            else:
                largest_gap = max(
                    largest_gap,
                    float((extended_angles[stop] - extended_angles[stop - 1]) * radius))
            if largest_gap > float(config["arc_max_component_gap"]):
                break
            angular_span = float(run_angles[-1] - run_angles[0])
            turn_degrees = float(np.degrees(angular_span))
            if turn_degrees < float(config["arc_min_turn_degrees"]):
                continue
            if turn_degrees > float(config["arc_max_turn_degrees"]):
                break
            arc_length = radius * angular_span
            start_angle = float(np.mod(run_angles[0], 2.0 * np.pi))
            window_start = int(np.searchsorted(all_angles, start_angle, side="left"))
            window_stop = int(np.searchsorted(
                all_angles, start_angle + angular_span + 1e-9, side="right"))
            narrow_weight = float(narrow_prefix[window_stop] - narrow_prefix[window_start])
            context_weight = float(context_prefix[window_stop] - context_prefix[window_start])
            contrast = narrow_weight / max(context_weight, 1e-9)
            if contrast < float(config["arc_min_contrast"]):
                continue
            arc_positions = (run_angles - run_angles[0]) * radius
            bin_size = float(config["arc_bin_size"])
            occupied_bins = len(np.unique(np.floor(arc_positions / bin_size).astype(int)))
            expected_bins = max(1, int(np.ceil(arc_length / bin_size)))
            coverage = min(1.0, occupied_bins / expected_bins)
            if coverage < float(config["arc_min_bin_coverage"]):
                continue
            support = float(np.sum(
                weights[run] * np.maximum(
                    0.0, 1.0 - residual[run]
                    / (float(config["arc_residual_tolerance"]) + 2.0))))
            score = float((
                arc_length + 12.0 * occupied_bins + 6.0 * support - 2.0 * len(run))
                * contrast)
            if score < float(config["arc_min_score"]):
                continue
            selection_score = (
                score * contrast * np.sqrt(
                    float(config["arc_min_radius"]) / max(radius, 1e-9))
                if config.get("arc_prefer_compact", False)
                else score)
            candidates.append({
                "model": "circular_arc", "score": score,
                "selection_score": float(selection_score),
                "contrast": contrast, "axis_span": float(arc_length),
                "total_turn_degrees": turn_degrees, "axis": 0,
                "member_local_indices": run.astype(int),
                "ordered_member_local_indices": run.astype(int),
                "centre": centre.copy(), "radius": float(radius),
            })
    return sorted(
        candidates, key=lambda item: item["selection_score"], reverse=True
    )[:int(config["arc_max_windows_per_circle"])]


def _arc_candidates(labels, work, config, protected):
    """Detect compact U/C-shaped fragmented tracks using circular Hough seeds."""
    if not config.get("arc_linking_enabled", False) or work.empty:
        return []
    areas = work["area_pixels"].to_numpy(dtype=float)
    eligible = (
        (areas >= int(config["curve_min_fragment_area"]))
        & (areas <= int(config["curve_max_fragment_area"]))
        & ~protected)
    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) < int(config["arc_min_components"]):
        return []
    points = work.loc[eligible, ["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    point_areas = areas[eligible_indices]
    mask = np.where(labels > 0, 255, 0).astype(np.uint8)
    scale = float(config["arc_hough_scale"])
    dilation = int(config["arc_hough_dilation"])
    hough_mask = cv2.dilate(mask, np.ones((dilation, dilation), dtype=np.uint8))
    if scale != 1.0:
        hough_mask = cv2.resize(
            hough_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(
        hough_mask, (9, 9) if scale == 1.0 else (7, 7),
        sigmaX=2.0 if scale == 1.0 else 1.2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=float(config["arc_hough_dp"]),
        minDist=float(config["arc_hough_min_distance"]) * scale,
        param1=float(config["arc_hough_param1"]), param2=float(config["arc_hough_param2"]),
        minRadius=max(1, int(round(float(config["arc_min_radius"]) * scale))),
        maxRadius=max(1, int(round(float(config["arc_max_radius"]) * scale))),
    )
    if circles is None:
        return []
    hypotheses = (
        np.asarray(circles[0], dtype=float)[:int(config["arc_max_hough_circles"])] / scale)
    raw_candidates = []
    for centre_x, centre_y, radius in hypotheses:
        centre = np.array([centre_x, centre_y], dtype=float)
        initial = _arc_windows(points, point_areas, centre, float(radius), config)
        raw_candidates.extend(initial)
        for candidate in sorted(initial, key=lambda item: item["score"], reverse=True)[:3]:
            fitted = _fit_circle_least_squares(points[candidate["member_local_indices"]])
            if fitted is not None:
                raw_candidates.extend(_arc_windows(points, point_areas, fitted[0], fitted[1], config))
    ranked = []
    ranked_members = []
    for candidate in sorted(
            raw_candidates, key=lambda item: item["selection_score"], reverse=True):
        local_members = set(map(int, candidate["member_local_indices"]))
        duplicate = any(
            len(local_members & previous) / max(1, min(len(local_members), len(previous)))
            > float(config["curve_max_member_overlap"])
            for previous in ranked_members)
        if duplicate:
            continue
        ranked.append(candidate.copy())
        ranked_members.append(local_members)

    if not ranked:
        return []

    # A wide U-shaped track is often better represented by two adjacent circle
    # segments.  Extend the strongest segment only with a nearby segment whose
    # bounding box overlaps strongly on the orthogonal dimension.  This joins
    # the left and right halves of a U without following an accidental circle
    # upward through dense background texture.
    selected = ranked[0]
    if int(config["arc_max_candidates"]) > 1:
        first_order = np.asarray(selected["ordered_member_local_indices"], dtype=int)
        first_points = points[first_order]
        first_low, first_high = first_points.min(axis=0), first_points.max(axis=0)
        first_size = np.maximum(first_high - first_low, 1.0)
        continuation_options = []
        # Quadratic extension already handles tall tracks.  Piecewise circular
        # joining is reserved for horizontally spread U/C shapes, where two
        # side-by-side circle segments are needed to cover the complete track.
        for candidate in ranked[1:] if first_size[0] >= first_size[1] else []:
            order = np.asarray(candidate["ordered_member_local_indices"], dtype=int)
            candidate_points = points[order]
            low, high = candidate_points.min(axis=0), candidate_points.max(axis=0)
            horizontal_extension = max(
                float(first_low[0] - low[0]), float(high[0] - first_high[0]), 0.0)
            if horizontal_extension < float(config["arc_join_min_axis_extension"]):
                continue
            vertical_extension = max(
                float(first_low[1] - low[1]), float(high[1] - first_high[1]), 0.0)
            if (vertical_extension
                    > float(config["arc_join_max_orthogonal_extension_ratio"])
                    * first_size[1]):
                continue
            intersection = np.maximum(0.0, np.minimum(first_high, high) - np.maximum(first_low, low))
            candidate_size = np.maximum(high - low, 1.0)
            overlap = intersection / np.minimum(first_size, candidate_size)
            if float(np.max(overlap)) < float(config["arc_join_min_orthogonal_overlap"]):
                continue
            endpoint_distances = np.linalg.norm(
                first_points[[0, -1], None, :] - candidate_points[None, [0, -1], :], axis=2)
            endpoint_pair = np.unravel_index(
                int(np.argmin(endpoint_distances)), endpoint_distances.shape)
            endpoint_gap = float(endpoint_distances[endpoint_pair])
            if endpoint_gap > float(config["arc_join_max_endpoint_gap"]):
                continue
            continuation_options.append(
                (float(candidate["selection_score"]) / (1.0 + endpoint_gap),
                 candidate, endpoint_pair))
        if continuation_options:
            _, continuation, endpoint_pair = max(continuation_options, key=lambda item: item[0])
            first_order = np.asarray(selected["ordered_member_local_indices"], dtype=int)
            second_order = np.asarray(
                continuation["ordered_member_local_indices"], dtype=int)
            if endpoint_pair[0] == 0:
                first_order = first_order[::-1]
            if endpoint_pair[1] == 1:
                second_order = second_order[::-1]
            combined_order = []
            seen = set()
            for index in np.concatenate([first_order, second_order]):
                if int(index) not in seen:
                    combined_order.append(int(index))
                    seen.add(int(index))
            span_a, span_b = float(selected["axis_span"]), float(continuation["axis_span"])
            selected["model"] = "piecewise_circular_arc"
            selected["score"] = float(selected["score"] + continuation["score"])
            selected["selection_score"] = float(
                selected["selection_score"] + continuation["selection_score"])
            selected["contrast"] = float(
                (selected["contrast"] * span_a + continuation["contrast"] * span_b)
                / max(span_a + span_b, 1e-9))
            selected["axis_span"] = float(span_a + span_b)
            selected["total_turn_degrees"] = float(
                selected["total_turn_degrees"] + continuation["total_turn_degrees"])
            selected["member_local_indices"] = np.asarray(combined_order, dtype=int)
            selected["ordered_member_local_indices"] = np.asarray(combined_order, dtype=int)

    selected_candidates = [selected]
    selected_member_sets = [set(map(int, selected["member_local_indices"]))]
    for candidate in ranked[1:]:
        members = set(map(int, candidate["member_local_indices"]))
        duplicate = any(
            len(members & previous) / max(1, min(len(members), len(previous)))
            > float(config["curve_max_member_overlap"])
            for previous in selected_member_sets)
        if duplicate:
            continue
        selected_candidates.append(candidate)
        selected_member_sets.append(members)
        if len(selected_candidates) >= int(config["arc_max_candidates"]):
            break
    for candidate in selected_candidates:
        candidate["member_indices"] = eligible_indices[
            candidate["member_local_indices"]].astype(int)
        candidate["ordered_member_indices"] = eligible_indices[
            candidate["ordered_member_local_indices"]].astype(int)
    return selected_candidates


def _large_arc_candidates(labels, work, config, protected):
    """Run the stricter long-track arc mode without weakening compact arcs."""
    if not config.get("large_arc_linking_enabled", False):
        return []
    large_config = config.copy()
    large_config.update({
        "arc_linking_enabled": True,
        "arc_prefer_compact": False,
        "arc_min_radius": int(config["large_arc_min_radius"]),
        "arc_max_radius": int(config["large_arc_max_radius"]),
        "arc_max_hough_circles": int(config["large_arc_max_hough_circles"]),
        "arc_max_components": int(config["large_arc_max_components"]),
        "arc_min_turn_degrees": float(config["large_arc_min_turn_degrees"]),
        "arc_max_turn_degrees": float(config["large_arc_max_turn_degrees"]),
        "arc_min_contrast": float(config["large_arc_search_min_contrast"]),
        "arc_min_score": float(config["large_arc_min_score"]),
        "arc_max_candidates": int(config["large_arc_max_candidates"]),
    })
    candidates = _arc_candidates(labels, work, large_config, protected)
    accepted = []
    for candidate in candidates:
        if float(candidate["contrast"]) < float(config["large_arc_min_contrast"]):
            continue
        final_threshold = (
            float(config["large_arc_min_piecewise_score"])
            if candidate["model"] == "piecewise_circular_arc"
            else float(config["large_arc_min_final_score"]))
        if float(candidate["score"]) >= final_threshold:
            accepted.append(candidate)
    return accepted


def link_track_fragments(labels, components, config):
    """Group mutually aligned centroid chains without adding foreground pixels."""
    work = components.reset_index(drop=True).copy()
    diagnostics = {"eligible_fragments": 0, "accepted_mutual_links": 0,
                   "curve_candidates": 0, "trajectory_groups": len(work),
                   "linked_groups": 0}
    if work.empty:
        return labels.copy(), pd.DataFrame(columns=TRAJECTORY_COMPONENT_COLUMNS), [], diagnostics
    if not config["trajectory_linking_enabled"]:
        return labels.copy(), _singleton_trajectory_table(work), [], diagnostics

    coordinates = work[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    areas = work["area_pixels"].to_numpy(dtype=float)
    eligible = areas >= config["trajectory_min_fragment_area"]
    protected = np.zeros(len(work), dtype=bool)
    if config["trajectory_protect_individual_candidates"]:
        protected = np.array(
            [_passes_individual_filters(row, config) for _, row in work.iterrows()], dtype=bool)
        eligible &= ~protected
    eligible_count = int(np.count_nonzero(eligible))
    if len(work) > config["trajectory_max_pairwise_fragments"]:
        warnings.warn(
            f"Trajectory linking skipped: {len(work)} raw fragments exceed the safety cap "
            f"of {config['trajectory_max_pairwise_fragments']}.")
        diagnostics["eligible_fragments"] = eligible_count
        return labels.copy(), _singleton_trajectory_table(work), [], diagnostics

    delta = coordinates[None, :, :] - coordinates[:, None, :]
    distances = np.linalg.norm(delta, axis=2)
    tangents = np.full_like(coordinates, np.nan, dtype=float)
    local_linearity = np.full(len(work), np.nan, dtype=float)
    for index in range(len(work)):
        local = eligible & (distances[index] <= config["trajectory_tangent_radius"])
        if np.count_nonzero(local) < config["trajectory_min_local_points"]:
            continue
        points, weights = coordinates[local], np.sqrt(areas[local])
        centre = np.average(points, axis=0, weights=weights)
        centred = points - centre
        covariance = (centred * weights[:, None]).T @ centred / weights.sum()
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        total_variance = float(eigenvalues.sum())
        if total_variance <= 1e-9:
            continue
        linearity = float((eigenvalues[-1] - eigenvalues[0]) / total_variance)
        if linearity < config["trajectory_min_local_linearity"]:
            continue
        tangents[index], local_linearity[index] = eigenvectors[:, -1], linearity

    finite_tangent = np.isfinite(tangents).all(axis=1)
    selections = [set() for _ in range(len(work))]
    selection_details = {}
    max_gap = float(config["trajectory_max_centroid_gap"])
    max_alignment = float(config["trajectory_max_alignment_degrees"])
    max_tangent_change = float(config["trajectory_max_tangent_change_degrees"])
    for index in range(len(work)):
        if not eligible[index] or not finite_tangent[index]:
            continue
        candidates = np.where(
            eligible & finite_tangent & (distances[index] > 0) & (distances[index] <= max_gap))[0]
        best_by_direction = {1: None, -1: None}
        for candidate in candidates:
            unit = delta[index, candidate] / distances[index, candidate]
            alignment_a = float(np.degrees(np.arccos(np.clip(abs(unit @ tangents[index]), 0.0, 1.0))))
            alignment_b = float(np.degrees(np.arccos(np.clip(abs(unit @ tangents[candidate]), 0.0, 1.0))))
            tangent_change = float(np.degrees(np.arccos(np.clip(
                abs(tangents[index] @ tangents[candidate]), 0.0, 1.0))))
            if alignment_a > max_alignment or alignment_b > max_alignment:
                continue
            if tangent_change > max_tangent_change:
                continue
            direction = 1 if unit @ tangents[index] >= 0 else -1
            score = float(distances[index, candidate] * (
                1.0 + alignment_a / max_alignment + alignment_b / max_alignment))
            details = {"score": score, "candidate": int(candidate),
                       "gap_pixels": float(distances[index, candidate]),
                       "alignment_a_degrees": alignment_a,
                       "alignment_b_degrees": alignment_b,
                       "tangent_change_degrees": tangent_change}
            previous = best_by_direction[direction]
            if previous is None or score < previous["score"]:
                best_by_direction[direction] = details
        for details in best_by_direction.values():
            if details is not None:
                candidate = details["candidate"]
                selections[index].add(candidate)
                selection_details[index, candidate] = details

    edges = []
    for index, selected in enumerate(selections):
        for candidate in selected:
            if index < candidate and index in selections[candidate]:
                forward = selection_details[index, candidate]
                reverse = selection_details[candidate, index]
                edges.append({"index_a": int(index), "index_b": int(candidate),
                              "gap_pixels": forward["gap_pixels"],
                              "alignment_a_degrees": forward["alignment_a_degrees"],
                              "alignment_b_degrees": reverse["alignment_a_degrees"],
                              "tangent_change_degrees": forward["tangent_change_degrees"],
                              "curve_fitted": False})

    polyline_candidates = _polyline_candidates(labels, work, config)
    if polyline_candidates and config.get("extended_polyline_enabled", False):
        base_member_sets = [
            set(map(int, candidate["member_indices"]))
            for candidate in polyline_candidates
        ]
        extended_config = config.copy()
        extended_config.update({
            "polyline_max_step": float(config["extended_polyline_max_step"]),
            "polyline_max_turn_degrees": float(
                config["extended_polyline_max_turn_degrees"]),
        })
        extended_candidates = _polyline_candidates(labels, work, extended_config)
        for candidate in extended_candidates:
            members = set(map(int, candidate["member_indices"]))
            if any(
                    len(members & previous) / max(1, min(len(members), len(previous)))
                    >= float(config["extended_polyline_min_member_overlap"])
                    for previous in base_member_sets):
                polyline_candidates.append(candidate)
    if polyline_candidates and config.get("polyline_fragment_extension_enabled", False):
        fragment_config = config.copy()
        fragment_config.update({
            "polyline_min_components": int(
                config["polyline_fragment_min_components"]),
            "polyline_min_length": float(config["polyline_fragment_min_length"]),
            "polyline_min_score": float(config["polyline_fragment_min_score"]),
        })
        fragment_candidates = _polyline_candidates(labels, work, fragment_config)
        polyline_candidates = _join_polyline_fragments(
            polyline_candidates, fragment_candidates, work, config)
    if not polyline_candidates:
        polyline_candidates = _beam_polyline_candidates(labels, work, config)
    proposed_curves = (
        _curve_candidates(labels, work, config, protected)
        + polyline_candidates
        + _arc_candidates(labels, work, config, protected)
        + _large_arc_candidates(labels, work, config, protected))
    curve_candidates, curve_member_sets, curve_boxes = [], [], []
    protected_boxes = []
    for index in np.flatnonzero(protected):
        row = work.iloc[int(index)]
        protected_boxes.append(np.array([
            float(row["bbox_x"]), float(row["bbox_y"]),
            float(row["bbox_x2_exclusive"]), float(row["bbox_y2_exclusive"])]))
    maximum_curves = (
        int(config["curve_max_candidates"])
        + int(config["polyline_max_candidates"])
        + int(config["beam_polyline_max_candidates"])
        + int(config["arc_max_candidates"])
        + int(config["large_arc_max_candidates"]))
    for candidate in sorted(
            proposed_curves, key=lambda item: item["selection_score"], reverse=True):
        members = set(map(int, candidate["member_indices"]))
        if any(
                len(members & previous) / max(1, min(len(members), len(previous)))
                > float(config["curve_max_member_overlap"])
                for previous in curve_member_sets):
            continue
        # A raw connected component may lie at a genuine crossing or inside
        # two nearby hypotheses.  Giving that fragment to both candidates
        # would union the two trajectories into one oversized box later.
        # Keep it with the higher-scoring candidate and remove it from the
        # later one so every accepted trajectory has exclusive membership.
        already_claimed = set().union(*curve_member_sets) if curve_member_sets else set()
        if members & already_claimed:
            ordered = np.asarray(candidate.get("ordered_member_indices", []), dtype=int)
            if ordered.size == 0:
                ordered = np.asarray(candidate["member_indices"], dtype=int)
            ordered = np.asarray(
                [int(index) for index in ordered if int(index) not in already_claimed],
                dtype=int)
            if (ordered.size < int(config["polyline_min_components"])
                    or ordered.size / max(1, len(members))
                    < float(config["curve_min_unclaimed_fraction"])):
                continue
            candidate = candidate.copy()
            candidate["ordered_member_indices"] = ordered
            candidate["member_indices"] = ordered.copy()
            members = set(map(int, ordered))
        member_points = coordinates[np.asarray(sorted(members), dtype=int)]
        low, high = member_points.min(axis=0), member_points.max(axis=0)
        candidate_box = np.concatenate([low, high])
        candidate_area = float(np.prod(np.maximum(high - low, 1.0)))
        overlaps_protected = False
        for protected_box in protected_boxes:
            intersection_size = np.maximum(
                0.0, np.minimum(high, protected_box[2:]) - np.maximum(low, protected_box[:2]))
            intersection_area = float(np.prod(intersection_size))
            protected_area = float(np.prod(np.maximum(
                protected_box[2:] - protected_box[:2], 1.0)))
            if (intersection_area / max(1.0, min(candidate_area, protected_area))
                    > float(config["curve_max_protected_overlap"])):
                overlaps_protected = True
                break
        if overlaps_protected:
            continue
        spatial_duplicate = False
        for previous_box in curve_boxes:
            intersection_size = np.maximum(
                0.0, np.minimum(high, previous_box[2:]) - np.maximum(low, previous_box[:2]))
            intersection_area = float(np.prod(intersection_size))
            previous_area = float(np.prod(np.maximum(
                previous_box[2:] - previous_box[:2], 1.0)))
            if (intersection_area / max(1.0, min(candidate_area, previous_area))
                    > float(config["curve_max_bbox_overlap"])):
                spatial_duplicate = True
                break
        if spatial_duplicate:
            continue
        curve_candidates.append(candidate)
        curve_member_sets.append(members)
        curve_boxes.append(candidate_box)
        if len(curve_candidates) >= maximum_curves:
            break
    # Curve candidates already provide their own ordered links.  Generic local
    # graph links touching those members can otherwise pull unrelated branches
    # and dense background regions into the same bounding box.
    curved_members = {
        int(index)
        for candidate in curve_candidates
        for index in candidate["member_indices"]
    }
    edges = [
        edge for edge in edges
        if edge["index_a"] not in curved_members and edge["index_b"] not in curved_members
    ]
    existing_pairs = {
        tuple(sorted((edge["index_a"], edge["index_b"]))) for edge in edges
    }
    for candidate in curve_candidates:
        members = np.asarray(candidate["member_indices"], dtype=int)
        axis = int(candidate["axis"])
        ordered = np.asarray(candidate.get("ordered_member_indices", []), dtype=int)
        if ordered.size == 0:
            ordered = members[np.argsort(coordinates[members, axis], kind="stable")]
        for index_a, index_b in zip(ordered[:-1], ordered[1:]):
            pair = tuple(sorted((int(index_a), int(index_b))))
            if pair in existing_pairs:
                continue
            gap = float(np.linalg.norm(coordinates[index_b] - coordinates[index_a]))
            edges.append({
                "index_a": int(index_a), "index_b": int(index_b),
                "gap_pixels": gap, "alignment_a_degrees": 0.0,
                "alignment_b_degrees": 0.0, "tangent_change_degrees": 0.0,
                "curve_fitted": True,
            })
            existing_pairs.add(pair)

    parent = np.arange(len(work), dtype=int)
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)
    def union(index_a, index_b):
        root_a, root_b = find(index_a), find(index_b)
        if root_a != root_b:
            parent[root_b] = root_a
    for edge in edges:
        union(edge["index_a"], edge["index_b"])
    groups_by_root = {}
    for index in range(len(work)):
        groups_by_root.setdefault(find(index), []).append(index)
    ordered_groups = sorted(
        groups_by_root.values(),
        key=lambda members: min(int(work.iloc[i]["component_id"]) for i in members))

    index_to_group, group_rows = {}, []
    for group_id, members in enumerate(ordered_groups, start=1):
        for index in members:
            index_to_group[index] = group_id
        member_set = set(members)
        group_edges = [e for e in edges if e["index_a"] in member_set and e["index_b"] in member_set]
        matching_curves = [
            candidate for candidate in curve_candidates
            if set(map(int, candidate["member_indices"])).issubset(member_set)
        ]
        best_curve = max(matching_curves, key=lambda item: item["score"], default=None)
        member_rows = work.iloc[members]
        member_areas = member_rows["area_pixels"].to_numpy(dtype=float)
        x0, y0 = int(member_rows["bbox_x"].min()), int(member_rows["bbox_y"].min())
        x1 = int(member_rows["bbox_x2_exclusive"].max())
        y1 = int(member_rows["bbox_y2_exclusive"].max())
        width, height = x1 - x0, y1 - y0
        valid_linearity = local_linearity[members]
        valid_linearity = valid_linearity[np.isfinite(valid_linearity)]
        steps = np.array([edge["gap_pixels"] for edge in group_edges], dtype=float)
        if steps.size >= 2:
            low_step, high_step = np.quantile(steps, [0.1, 0.9])
            step_ratio = float(high_step / max(low_step, 1e-9))
        else:
            step_ratio = 1.0
        component_ids = [int(value) for value in member_rows["component_id"]]
        group_rows.append({
            "source_filename": member_rows.iloc[0]["source_filename"],
            "source_stem": member_rows.iloc[0]["source_stem"], "component_id": group_id,
            "area_pixels": int(member_areas.sum()),
            "centroid_x": float(np.average(member_rows["centroid_x"], weights=member_areas)),
            "centroid_y": float(np.average(member_rows["centroid_y"], weights=member_areas)),
            "bbox_x": x0, "bbox_y": y0, "bbox_width": width, "bbox_height": height,
            "bbox_long_side": max(width, height), "bbox_short_side": min(width, height),
            "bbox_x2_exclusive": x1, "bbox_y2_exclusive": y1,
            "touches_boundary": bool(member_rows["touches_boundary"].any()),
            "source_component_ids": ";".join(map(str, component_ids)),
            "fragment_count": len(members), "trajectory_linked": len(members) > 1,
            "trajectory_link_count": len(group_edges),
            "trajectory_link_length_pixels": float(steps.sum()),
            "trajectory_mean_link_gap_pixels": float(steps.mean()) if steps.size else 0.0,
            "trajectory_mean_local_linearity": float(valid_linearity.mean()) if valid_linearity.size else np.nan,
            "trajectory_max_link_alignment_degrees": max(
                (max(e["alignment_a_degrees"], e["alignment_b_degrees"]) for e in group_edges), default=0.0),
            "trajectory_max_tangent_change_degrees": max(
                (e["tangent_change_degrees"] for e in group_edges), default=0.0),
            "trajectory_step_ratio": step_ratio,
            "curve_fitted": best_curve is not None,
            "curve_model": str(best_curve["model"]) if best_curve is not None else "none",
            "curve_score": float(best_curve["score"]) if best_curve is not None else 0.0,
            "curve_contrast": float(best_curve["contrast"]) if best_curve is not None else 0.0,
            "curve_axis_span_pixels": float(best_curve["axis_span"]) if best_curve is not None else 0.0,
            "curve_total_turn_degrees": (
                float(best_curve["total_turn_degrees"]) if best_curve is not None else 0.0),
        })

    maximum_raw_label = int(labels.max()) if labels.size else 0
    label_lookup = np.zeros(maximum_raw_label + 1, dtype=np.int32)
    for index, row in work.iterrows():
        label_lookup[int(row["component_id"])] = index_to_group[index]
    grouped_labels = label_lookup[labels]
    link_records = []
    for edge in edges:
        index_a, index_b = edge["index_a"], edge["index_b"]
        link_records.append({
            "trajectory_group_id": index_to_group[index_a],
            "source_component_id_a": int(work.iloc[index_a]["component_id"]),
            "source_component_id_b": int(work.iloc[index_b]["component_id"]),
            "x_a": float(coordinates[index_a, 0]), "y_a": float(coordinates[index_a, 1]),
            "x_b": float(coordinates[index_b, 0]), "y_b": float(coordinates[index_b, 1]),
            "gap_pixels": edge["gap_pixels"],
            "alignment_a_degrees": edge["alignment_a_degrees"],
            "alignment_b_degrees": edge["alignment_b_degrees"],
            "tangent_change_degrees": edge["tangent_change_degrees"],
            "curve_fitted": bool(edge.get("curve_fitted", False)),
        })
    grouped = pd.DataFrame(group_rows, columns=TRAJECTORY_COMPONENT_COLUMNS)
    diagnostics = {"eligible_fragments": eligible_count,
                   "accepted_mutual_links": len(link_records),
                   "curve_candidates": len(curve_candidates),
                   "trajectory_groups": len(grouped),
                   "linked_groups": int((grouped["fragment_count"] > 1).sum())}
    return grouped_labels, grouped, link_records, diagnostics


def filter_components(labels, components, config):
    """Filter trajectory groups and build sequential accepted labels."""
    decisions = components.copy()
    decisions["aspect_ratio"] = (
        decisions[["bbox_width", "bbox_height"]].max(axis=1)
        / decisions[["bbox_width", "bbox_height"]].min(axis=1)
        if not decisions.empty else pd.Series(dtype=float))
    decisions["accepted"] = False
    decisions["rejection_reason"] = ""
    decisions["label_id"] = 0
    for index, row in decisions.iterrows():
        reasons = []
        if row["area_pixels"] < config["min_component_area"]:
            reasons.append(f"area<{config['min_component_area']}")
        max_area = config["max_component_area"]
        if max_area is not None and row["area_pixels"] > max_area:
            reasons.append(f"area>{max_area}")
        if row["bbox_long_side"] < config["min_bbox_long_side"]:
            reasons.append(f"long_side<{config['min_bbox_long_side']}")
        if row["bbox_short_side"] < config["min_bbox_short_side"]:
            reasons.append(f"short_side<{config['min_bbox_short_side']}")
        ratio_range = config["aspect_ratio_range"]
        if ratio_range is not None and not (ratio_range[0] <= row["aspect_ratio"] <= ratio_range[1]):
            reasons.append(f"aspect_not_in_{tuple(ratio_range)}")
        if int(row["fragment_count"]) > 1 and not bool(row["curve_fitted"]):
            if row["fragment_count"] < config["trajectory_min_group_fragments"]:
                reasons.append(f"fragments<{config['trajectory_min_group_fragments']}")
            if row["trajectory_link_count"] < row["fragment_count"] - 1:
                reasons.append("trajectory_not_connected")
            if row["trajectory_link_length_pixels"] < config["trajectory_min_group_link_length"]:
                reasons.append(f"link_length<{config['trajectory_min_group_link_length']}")
            if (not np.isfinite(row["trajectory_mean_link_gap_pixels"])
                    or row["trajectory_mean_link_gap_pixels"] > config["trajectory_max_mean_link_gap"]):
                reasons.append(f"mean_link_gap>{config['trajectory_max_mean_link_gap']}")
            if row["trajectory_mean_local_linearity"] < config["trajectory_min_local_linearity"]:
                reasons.append(f"linearity<{config['trajectory_min_local_linearity']}")
            if row["trajectory_max_tangent_change_degrees"] > config["trajectory_max_tangent_change_degrees"]:
                reasons.append(f"turn>{config['trajectory_max_tangent_change_degrees']}deg")
            if row["trajectory_step_ratio"] > config["trajectory_max_step_ratio"]:
                reasons.append(f"step_ratio>{config['trajectory_max_step_ratio']}")
        decisions.at[index, "accepted"] = not reasons
        decisions.at[index, "rejection_reason"] = "; ".join(reasons)
    accepted_indices = decisions.index[decisions["accepted"]].tolist()
    for sequential_label, index in enumerate(accepted_indices, start=1):
        decisions.at[index, "label_id"] = sequential_label
    decisions["label_id"] = decisions["label_id"].astype(int)
    maximum_raw_label = int(labels.max()) if labels.size else 0
    label_lookup = np.zeros(maximum_raw_label + 1, dtype=np.int32)
    for row in decisions.loc[accepted_indices].itertuples(index=False):
        label_lookup[int(row.component_id)] = int(row.label_id)
    accepted_labels = label_lookup[labels]
    final_mask = np.where(accepted_labels > 0, 255, 0).astype(np.uint8)
    accepted = decisions.loc[accepted_indices].copy().reset_index(drop=True)
    return decisions, accepted, accepted_labels, final_mask
