from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


KNOWLEDGE_FILE = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "disease_information.json"


# ============================================================
# ADVANCED LEAF OBSERVATION ENGINE
# ============================================================


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
        if not np.isfinite(val):
            return default
        return val
    except Exception:
        return default


def _soft_round(value: Any, digits: int = 3) -> float:
    value = _safe_float(value, 0.0)
    return round(value, digits)


def _safe_percent(value: Any) -> float:
    value = _safe_float(value, 0.0)
    return max(0.0, min(100.0, value))


def _to_rgb_array(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image.copy()
    return np.asarray(image)


def _normalize_mask(mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return np.zeros((0, 0), dtype=np.uint8)
    arr = np.asarray(mask)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if arr.dtype != np.uint8:
        arr = (arr > 0).astype(np.uint8) * 255
    return arr.astype(np.uint8)


def _connected_components(mask: np.ndarray, min_area: int = 16) -> List[np.ndarray]:
    if mask.size == 0:
        return []
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    contours = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            component = (labels == label).astype(np.uint8) * 255
            contours.append(component)
    return contours


def _component_metrics(component_mask: np.ndarray, leaf_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if component_mask.size == 0 or component_mask.sum() == 0:
        return {
            "area_pixels": 0,
            "area_percent_leaf": 0.0,
            "perimeter_pixels": 0.0,
            "bounding_box": {"x": 0, "y": 0, "w": 0, "h": 0},
            "centroid": {"x": 0, "y": 0},
            "aspect_ratio": 0.0,
            "circularity": None,
            "solidity": None,
            "extent": 0.0,
            "compactness": None,
        }

    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "area_pixels": 0,
            "area_percent_leaf": 0.0,
            "perimeter_pixels": 0.0,
            "bounding_box": {"x": 0, "y": 0, "w": 0, "h": 0},
            "centroid": {"x": 0, "y": 0},
            "aspect_ratio": 0.0,
            "circularity": None,
            "solidity": None,
            "extent": 0.0,
            "compactness": None,
        }

    contour = max(contours, key=cv2.contourArea)
    area_pixels = float(cv2.contourArea(contour))
    perimeter_pixels = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)

    M = cv2.moments(contour)
    cx = float(M["m10"] / M["m00"]) if M["m00"] > 0 else x + w / 2.0
    cy = float(M["m01"] / M["m00"]) if M["m00"] > 0 else y + h / 2.0

    bbox_area = float(w * h)
    extent = area_pixels / bbox_area if bbox_area > 0 else 0.0
    aspect_ratio = float(w / h) if h > 0 else 0.0
    circularity = (4.0 * math.pi * area_pixels / (perimeter_pixels ** 2)) if perimeter_pixels > 0 else None

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area_pixels / hull_area if hull_area > 0 else None
    compactness = (perimeter_pixels ** 2) / (4.0 * math.pi * area_pixels) if area_pixels > 0 else None

    leaf_pixels = float((leaf_mask > 0).sum()) if leaf_mask is not None and leaf_mask.size else 0.0
    area_percent_leaf = (area_pixels / leaf_pixels * 100.0) if leaf_pixels > 0 else 0.0

    return {
        "area_pixels": int(round(area_pixels)),
        "area_percent_leaf": _soft_round(area_percent_leaf, 3),
        "perimeter_pixels": _soft_round(perimeter_pixels, 3),
        "bounding_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "centroid": {"x": int(round(cx)), "y": int(round(cy))},
        "aspect_ratio": _soft_round(aspect_ratio, 3),
        "circularity": _soft_round(circularity, 4) if circularity is not None else None,
        "solidity": _soft_round(solidity, 4) if solidity is not None else None,
        "extent": _soft_round(extent, 4),
        "compactness": _soft_round(compactness, 4) if compactness is not None else None,
    }


def _derive_mask_from_inputs(image: Any, leaf_mask: Optional[np.ndarray], affected_mask: Optional[np.ndarray], gradcam: Optional[np.ndarray]) -> Tuple[np.ndarray, str]:
    rgb = _to_rgb_array(image)
    leaf = _normalize_mask(leaf_mask)
    affected = _normalize_mask(affected_mask)

    if affected is not None and affected.size and affected.max() > 0:
        return affected, "existing affected mask"

    if gradcam is not None and np.asarray(gradcam).size:
        grad = np.asarray(gradcam)
        if grad.ndim == 2:
            if grad.shape != rgb.shape[:2]:
                grad = cv2.resize(grad.astype(np.float32), (rgb.shape[1], rgb.shape[0]))
            grad_norm = (grad - np.nanmin(grad)) / (np.nanmax(grad) - np.nanmin(grad) + 1e-6)
            thresh = grad_norm >= 0.5
            mask = (thresh * 255).astype(np.uint8)
            if leaf.size and leaf.shape == mask.shape:
                mask = cv2.bitwise_and(mask, leaf)
            return mask, "derived from Grad-CAM"

    if leaf.size and leaf.shape[0] > 0 and leaf.shape[1] > 0:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.bitwise_and(thresh, leaf)
        if mask.sum() > 0:
            return mask, "derived from leaf-texture threshold"

    return np.zeros_like(rgb[:, :, 0], dtype=np.uint8), "not available"


def _compute_image_quality(image: Any, leaf_mask: Optional[np.ndarray], affected_mask: Optional[np.ndarray]) -> Dict[str, Any]:
    rgb = _to_rgb_array(image)
    if rgb.size == 0:
        return {
            "resolution": {"width": 0, "height": 0},
            "quality_score": 0.0,
            "classification": "Limited",
            "blur_estimate": "Not available",
            "brightness": "Not available",
            "contrast": "Not available",
            "background_interference": "Not available",
            "leaf_coverage": "Not available",
            "notes": "Image is unavailable for advanced visual analysis."
        }

    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    blur_score = float(np.std(lap))
    brightness = float(gray.mean())
    contrast = float(gray.std())

    leaf = _normalize_mask(leaf_mask)
    leaf_pixels = float((leaf > 0).sum()) if leaf.size else 0.0
    total_pixels = max(1.0, float(width * height))
    leaf_coverage = (leaf_pixels / total_pixels) * 100.0 if leaf_pixels > 0 else 0.0

    affected = _normalize_mask(affected_mask)
    affected_pixels = float((affected > 0).sum()) if affected.size else 0.0
    affected_ratio = (affected_pixels / total_pixels) * 100.0 if affected.size else 0.0

    if blur_score < 25:
        blur_label = "Low blur"
    elif blur_score < 60:
        blur_label = "Moderate blur"
    else:
        blur_label = "High blur"

    if brightness < 60:
        brightness_desc = "Dark"
    elif brightness < 180:
        brightness_desc = "Moderate"
    else:
        brightness_desc = "Bright"

    if contrast < 30:
        contrast_desc = "Low contrast"
    elif contrast < 60:
        contrast_desc = "Moderate contrast"
    else:
        contrast_desc = "High contrast"

    if leaf_coverage > 65:
        leaf_visibility = "High"
    elif leaf_coverage > 30:
        leaf_visibility = "Moderate"
    else:
        leaf_visibility = "Low"

    if affected_ratio < 10:
        background_interference = "Low"
    elif affected_ratio < 30:
        background_interference = "Moderate"
    else:
        background_interference = "High"

    quality_score = 0.0
    quality_score += 35 if leaf_visibility == "High" else 20 if leaf_visibility == "Moderate" else 8
    quality_score += 25 if brightness_desc in {"Moderate", "Bright"} else 10
    quality_score += 20 if contrast_desc != "Low contrast" else 8
    quality_score += 20 if blur_label != "High blur" else 5

    if quality_score >= 80:
        classification = "Good"
    elif quality_score >= 55:
        classification = "Moderate"
    else:
        classification = "Limited"

    return {
        "resolution": {"width": int(width), "height": int(height)},
        "quality_score": round(quality_score, 2),
        "classification": classification,
        "blur_estimate": blur_label,
        "brightness": brightness_desc,
        "contrast": contrast_desc,
        "background_interference": background_interference,
        "leaf_coverage": leaf_visibility,
        "notes": "Image suitability for visual analysis is based on leaf occupancy, brightness, contrast, and blur characteristics."
    }


def _compute_color_features(image: Any, mask: np.ndarray, healthy_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    rgb = _to_rgb_array(image)
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return {
            "mean_rgb": "Not available",
            "mean_hsv": "Not available",
            "contrast_to_healthy": "Not available",
            "description": "No abnormal regions were identified for color analysis."
        }

    pixels = rgb[binary > 0]
    if pixels.size == 0:
        return {
            "mean_rgb": "Not available",
            "mean_hsv": "Not available",
            "contrast_to_healthy": "Not available",
            "description": "No abnormal pixels were available for color statistics."
        }

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv[binary > 0]

    mean_rgb = [float(np.mean(pixels[:, i])) for i in range(3)]
    mean_hsv = [float(np.mean(hsv_pixels[:, i])) for i in range(3)]

    healthy_pixels = rgb[(healthy_mask > 0).astype(bool)] if healthy_mask is not None and np.array(healthy_mask).size else None
    if healthy_pixels is not None and healthy_pixels.size > 0:
        healthy_mean = np.mean(healthy_pixels, axis=0)
        contrast = np.linalg.norm(np.array(mean_rgb) - healthy_mean)
        contrast_desc = "High" if contrast > 80 else "Moderate" if contrast > 30 else "Low"
    else:
        contrast = None
        contrast_desc = "Not available"

    return {
        "mean_rgb": [round(float(x), 2) for x in mean_rgb],
        "mean_hsv": [round(float(x), 2) for x in mean_hsv],
        "contrast_to_healthy": contrast_desc,
        "description": "Detected abnormal regions show a measurable color contrast relative to surrounding apparently healthy leaf tissue."
    }


def _compute_texture_features(image: Any, mask: np.ndarray, healthy_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    rgb = _to_rgb_array(image)
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return {
            "texture_variation": "Not available",
            "edge_density": "Not available",
            "contrast": "Not available",
            "description": "No abnormal regions were available for texture analysis."
        }

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    affected = gray[binary > 0]
    mean_intensity = float(np.mean(affected))
    std_intensity = float(np.std(affected))
    local_var = float(np.var(gray[binary > 0]))

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges[binary > 0] > 0)) if binary.sum() else 0.0

    healthy = gray[(healthy_mask > 0).astype(bool)] if healthy_mask is not None and np.array(healthy_mask).size else None
    if healthy is not None and healthy.size:
        healthy_std = float(np.std(healthy))
        texture_level = "High" if std_intensity > 2 * healthy_std else "Moderate" if std_intensity > healthy_std else "Low"
    else:
        texture_level = "Moderate" if std_intensity > 25 else "Low"

    return {
        "texture_variation": texture_level,
        "edge_density": round(edge_density, 3),
        "contrast": round(std_intensity, 2),
        "description": "Detected abnormal regions show measurable texture variation relative to surrounding healthy tissue."
    }


def _compute_spatial_distribution(mask: np.ndarray, leaf_mask: Optional[np.ndarray] = None, canvas_shape: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    if mask.size == 0:
        return {
            "distribution": "Not available",
            "edge_dominance": "Not available",
            "center_dominance": "Not available",
            "clustering": "Not available",
            "description": "No abnormal region mask was available for spatial analysis."
        }

    h, w = mask.shape[:2] if mask.ndim >= 2 else (0, 0)
    if h == 0 or w == 0:
        return {
            "distribution": "Not available",
            "edge_dominance": "Not available",
            "center_dominance": "Not available",
            "clustering": "Not available",
            "description": "The mask size was invalid for spatial analysis."
        }

    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return {
            "distribution": "Not available",
            "edge_dominance": "Not available",
            "center_dominance": "Not available",
            "clustering": "Not available",
            "description": "No abnormal pixels were identified for spatial analysis."
        }

    cy, cx = h / 2.0, w / 2.0
    distances_to_center = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    distances_to_edge = np.minimum(xs, ys)
    distances_to_edge = np.minimum(distances_to_edge, np.minimum(w - xs, h - ys))

    edge_ratio = float(np.mean(distances_to_edge < 0.12 * min(w, h)))
    center_ratio = float(np.mean(distances_to_center < 0.35 * min(w, h)))

    if edge_ratio > 0.55 and center_ratio < edge_ratio:
        distribution = "Edge-dominant"
    elif center_ratio > 0.55 and center_ratio > edge_ratio:
        distribution = "Center-dominant"
    elif edge_ratio > 0.25 and center_ratio > 0.25:
        distribution = "Mixed"
    else:
        distribution = "Scattered"

    cluster_score = float(np.mean(np.bincount(np.clip((ys // max(1, h // 8)), 0, 7) * 8 + np.clip((xs // max(1, w // 8)), 0, 7))))
    clustering = "High" if cluster_score > 0.4 else "Moderate" if cluster_score > 0.2 else "Low"

    return {
        "distribution": distribution,
        "edge_dominance": round(edge_ratio * 100.0, 2),
        "center_dominance": round(center_ratio * 100.0, 2),
        "clustering": clustering,
        "description": "Abnormal regions were mapped across the leaf image to describe their spatial pattern and concentration."
    }


def _compute_connectivity(mask: np.ndarray) -> Dict[str, Any]:
    if mask.size == 0:
        return {
            "isolated_regions": 0,
            "touching_regions": 0,
            "merged_regions": 0,
            "largest_component_percentage": 0.0,
            "connectivity_level": "LOW CONNECTIVITY",
            "description": "No abnormal mask available for connectivity analysis."
        }

    comps = _connected_components(mask, min_area=16)
    if not comps:
        return {
            "isolated_regions": 0,
            "touching_regions": 0,
            "merged_regions": 0,
            "largest_component_percentage": 0.0,
            "connectivity_level": "LOW CONNECTIVITY",
            "description": "No meaningful connected abnormal regions were detected."
        }

    areas = []
    for comp in comps:
        areas.append(float((comp > 0).sum()))

    total = float((mask > 0).sum())
    largest_component = max(areas) if areas else 0.0
    largest_component_percentage = (largest_component / total * 100.0) if total > 0 else 0.0

    connectivity = "LOW CONNECTIVITY"
    if largest_component_percentage > 50.0:
        connectivity = "HIGH CONNECTIVITY"
    elif largest_component_percentage > 20.0:
        connectivity = "MODERATE CONNECTIVITY"

    return {
        "isolated_regions": len(comps),
        "touching_regions": max(0, len(comps) - 1),
        "merged_regions": int(len(comps) > 1),
        "largest_component_percentage": round(largest_component_percentage, 2),
        "connectivity_level": connectivity,
        "description": "Connectivity is based on the relative size of the largest connected abnormal region within the leaf mask."
    }


def _compute_vein_relationship(image: Any, mask: np.ndarray, vein_map: Optional[np.ndarray]) -> Dict[str, Any]:
    if mask.size == 0:
        return {
            "vein_relationship": "Not available",
            "proximity_percent": 0.0,
            "description": "No abnormal-region mask was available for vein relationship analysis."
        }
    if vein_map is None or np.asarray(vein_map).size == 0:
        return {
            "vein_relationship": "Not available",
            "proximity_percent": 0.0,
            "description": "No visible vein map was available for spatial comparison."
        }

    vein = _normalize_mask(vein_map)
    if vein.size == 0:
        return {
            "vein_relationship": "Not available",
            "proximity_percent": 0.0,
            "description": "The vein map was empty."
        }

    affected = (mask > 0).astype(bool)
    veins = (vein > 0).astype(bool)
    if affected.sum() == 0 or veins.sum() == 0:
        return {
            "vein_relationship": "Not available",
            "proximity_percent": 0.0,
            "description": "No overlapping pixels were available to compare abnormal regions and visible veins."
        }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated_veins = cv2.dilate(vein, kernel, iterations=1)
    proximity = cv2.bitwise_and((mask > 0).astype(np.uint8) * 255, dilated_veins)
    proximity_percent = (float(proximity.sum()) / max(1.0, float((mask > 0).sum()))) * 100.0

    if proximity_percent > 35:
        label = "Moderate"
    elif proximity_percent > 15:
        label = "Low"
    else:
        label = "Minimal"

    return {
        "vein_relationship": label,
        "proximity_percent": round(proximity_percent, 2),
        "description": "This is a spatial relationship measurement between abnormal regions and the visible veins; it is not a biological claim that the pathogen follows vein pathways."
    }


def _compute_damage_pattern(mask: np.ndarray) -> Dict[str, Any]:
    if mask.size == 0:
        return {"pattern": "Not available", "description": "No abnormal-region mask was available for damage-pattern classification."}

    comps = _connected_components(mask, min_area=16)
    if not comps:
        return {"pattern": "Not available", "description": "No meaningful connected abnormal region was detected."}

    total = float((mask > 0).sum())
    if total <= 0:
        return {"pattern": "Not available", "description": "The abnormal mask was empty."}

    largest = max(float((comp > 0).sum()) for comp in comps)
    ratio = largest / total
    if ratio > 0.6:
        dominant = "PATCH-DOMINANT"
        detail = "Several abnormal regions form a larger connected patch."
    elif len(comps) > 6:
        dominant = "SCATTERED"
        detail = "The abnormal regions are numerous and spread across the leaf surface."
    elif len(comps) <= 2:
        dominant = "CLUSTERED"
        detail = "The abnormal regions are concentrated in nearby parts of the leaf surface."
    else:
        dominant = "MIXED"
        detail = "The defect pattern is a combination of clustered and diffuse regions."

    return {"pattern": dominant, "description": detail}


def _compute_affected_area_geometry(mask: np.ndarray, leaf_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if mask.size == 0:
        return {
            "meaningful_abnormal_regions": 0,
            "largest_component_pct": 0.0,
            "median_component_pct": 0.0,
            "distribution": "Not available",
            "description": "No abnormal mask was available for geometry analysis."
        }

    comps = _connected_components(mask, min_area=16)
    if not comps:
        return {
            "meaningful_abnormal_regions": 0,
            "largest_component_pct": 0.0,
            "median_component_pct": 0.0,
            "distribution": "Not available",
            "description": "No meaningful abnormal regions were detected."
        }

    region_percentages = []
    for comp in comps:
        area = float((comp > 0).sum())
        if leaf_mask is not None and leaf_mask.size:
            leaf_total = float((leaf_mask > 0).sum())
            region_percentages.append((area / leaf_total) * 100.0 if leaf_total > 0 else 0.0)
        else:
            region_percentages.append(area)

    region_percentages = sorted(region_percentages)
    largest_component_pct = region_percentages[-1] if region_percentages else 0.0
    median_component_pct = float(np.median(region_percentages)) if region_percentages else 0.0

    distribution = "Clustered" if len(region_percentages) <= 3 else "Scattered" if len(region_percentages) > 8 else "Mixed"

    return {
        "meaningful_abnormal_regions": len(region_percentages),
        "largest_component_pct": round(largest_component_pct, 2),
        "median_component_pct": round(median_component_pct, 2),
        "distribution": distribution,
        "description": "Visible affected-area geometry describes how abnormal pixels are distributed across the leaf surface."
    }


def _compute_cross_analysis_consistency(gradcam: Optional[np.ndarray], mask: np.ndarray, affected_leaf_percent: float, prediction: str = "Unknown") -> Dict[str, Any]:
    if mask.size == 0 or gradcam is None:
        return {
            "consistency": "Limited",
            "description": "Not enough image evidence was available to assess cross-analysis consistency."
        }

    grad = np.asarray(gradcam)
    grad_2d = grad.reshape(-1)
    if grad_2d.size == 0:
        return {
            "consistency": "Limited",
            "description": "Grad-CAM evidence was missing."
        }

    mask_vector = (mask > 0).reshape(-1)
    overlap = float(np.mean((mask_vector > 0) & (grad_2d > 0.5))) if mask_vector.size else 0.0
    overlap_pct = overlap * 100.0
    if overlap_pct > 45:
        consistency = "High"
    elif overlap_pct > 20:
        consistency = "Moderate"
    else:
        consistency = "Low"

    return {
        "consistency": consistency,
        "overlap_percent": round(overlap_pct, 2),
        "description": f"Cross-analysis consistency compares the spatial overlap between the detected abnormal mask and the model attention map. This is not a diagnostic confirmation."
    }


def _load_disease_knowledge() -> Dict[str, Any]:
    try:
        if not KNOWLEDGE_FILE.exists():
            return {}
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _interpret_visual_observations(prediction: str, obs: Dict[str, Any]) -> str:
    knowledge = _load_disease_knowledge()
    disease = knowledge.get(prediction, {})
    disease_name = disease.get("condition") or prediction.replace("_", " ")

    spatial = obs.get("spatial_distribution", {})
    connectivity = obs.get("connectivity", {})
    damage = obs.get("damage_pattern", {})
    color = obs.get("color", {})
    texture = obs.get("texture", {})
    quality = obs.get("image_quality", {})

    distribution = spatial.get("distribution", "Not available")
    conn = connectivity.get("connectivity_level", "Not available")
    pattern = damage.get("pattern", "Not available")
    contrast = color.get("contrast_to_healthy", "Not available")
    texture_var = texture.get("texture_variation", "Not available")
    suitability = quality.get("classification", "Moderate")

    base = (
        f"The observed image contains {distribution.lower()} abnormal regions with {conn.lower()} connectivity. "
        f"The visible damage pattern is {pattern.lower()} and the affected tissue shows {contrast.lower()} color contrast with {texture_var.lower()} texture variation. "
        f"Image suitability for visual analysis is {suitability.lower()}. "
    )

    if disease and disease_name:
        return (
            f"{base}These measurable visual characteristics are compatible with the lesion pattern described for {disease_name}, but they are supporting image evidence and do not independently confirm the causal pathogen."
        )

    return (
        f"{base}These observations are derived from visible geometry, color, texture, and spatial distribution only and should not be interpreted as laboratory confirmation."
    )


def _make_summary_text(obs: Dict[str, Any]) -> str:
    quality = obs.get("image_quality", {})
    spatial = obs.get("spatial_distribution", {})
    connectivity = obs.get("connectivity", {})
    damage = obs.get("damage_pattern", {})
    color = obs.get("color", {})
    texture = obs.get("texture", {})

    affected = obs.get("affected_area_geometry", {}).get("largest_component_pct", 0.0)
    distribution = spatial.get("distribution", "Not available")
    connectivity_level = connectivity.get("connectivity_level", "Not available")
    damage_pattern = damage.get("pattern", "Not available")

    summary = (
        f"The uploaded leaf shows abnormal regions with a largest connected component of {affected:.2f}% of the visible leaf area. "
        f"The spatial distribution is {distribution}, connectivity is {connectivity_level}, and the observed damage pattern is {damage_pattern}. "
        f"The affected tissue shows {color.get('contrast_to_healthy', 'measurable')} color contrast and {texture.get('texture_variation', 'moderate')} texture variation. "
        f"Image suitability for visual analysis is {quality.get('classification', 'Moderate')}."
    )

    return summary


def analyze_leaf_observations(
    image: Any,
    leaf_mask: Optional[np.ndarray] = None,
    affected_mask: Optional[np.ndarray] = None,
    gradcam: Optional[np.ndarray] = None,
    vein_map: Optional[np.ndarray] = None,
    prediction: str = "Unknown",
    affected_leaf_percent: float = 0.0,
) -> Dict[str, Any]:
    """
    Build an advanced image-geometry observation report without replacing the
    existing disease-classification pipeline.

    This is a supportive visual-analysis module and must not be presented as a
    second diagnosis or independent disease classifier.
    """
    started = time.perf_counter()
    rgb = _to_rgb_array(image)

    if rgb.size == 0:
        return {
            "image_quality": {"classification": "Limited"},
            "morphology": {"status": "Not available"},
            "spatial_distribution": {"distribution": "Not available"},
            "connectivity": {"connectivity_level": "LOW CONNECTIVITY"},
            "color": {"description": "No image was available for analysis."},
            "texture": {"description": "No image was available for analysis."},
            "vein_relationship": {"vein_relationship": "Not available"},
            "damage_pattern": {"pattern": "Not available"},
            "affected_area_geometry": {"meaningful_abnormal_regions": 0},
            "cross_analysis": {"consistency": "Limited"},
            "summary": "No image data was available for advanced visual observation.",
            "processing_time_seconds": 0.0,
        }

    leaf = _normalize_mask(leaf_mask)
    if leaf.size == 0 or leaf.shape != rgb.shape[:2]:
        if leaf.size == 0:
            leaf = np.zeros_like(rgb[:, :, 0], dtype=np.uint8)
        else:
            leaf = cv2.resize(leaf, (rgb.shape[1], rgb.shape[0]))

    observed_mask, mask_source = _derive_mask_from_inputs(rgb, leaf, affected_mask, gradcam)
    actual_mask = _normalize_mask(observed_mask)

    if actual_mask.size == 0 or actual_mask.max() == 0:
        morphology = {
            "lesion_count": 0,
            "total_abnormal_area_percent": 0.0,
            "mean_lesion_area": 0.0,
            "median_lesion_area": 0.0,
            "largest_lesion_area": 0.0,
            "description": "No meaningful abnormal region was detected in the current image."
        }
    else:
        components = _connected_components(actual_mask, min_area=16)
        metrics = [_component_metrics(comp, leaf) for comp in components]
        areas = [m["area_pixels"] for m in metrics if m["area_pixels"] > 0]
        area_pct = [m["area_percent_leaf"] for m in metrics if m["area_percent_leaf"] > 0]
        circularities = [m["circularity"] for m in metrics if m["circularity"] is not None]
        solids = [m["solidity"] for m in metrics if m["solidity"] is not None]

        morphology = {
            "lesion_count": len(metrics),
            "total_abnormal_area_percent": round(float((actual_mask > 0).sum()) / max(1.0, float((leaf > 0).sum())) * 100.0 if leaf.size else 0.0, 2),
            "mean_lesion_area": round(float(np.mean(areas)) if areas else 0.0, 2),
            "median_lesion_area": round(float(np.median(areas)) if areas else 0.0, 2),
            "largest_lesion_area": round(float(np.max(areas)) if areas else 0.0, 2),
            "smallest_meaningful_lesion_area": round(float(np.min(areas)) if areas else 0.0, 2),
            "mean_circularity": round(float(np.mean(circularities)) if circularities else 0.0, 4),
            "mean_solidity": round(float(np.mean(solids)) if solids else 0.0, 4),
            "largest_region_percentage": round(max(area_pct) if area_pct else 0.0, 2),
            "mask_source": mask_source,
            "description": "Morphology metrics summarize the size and shape of the detected abnormal components."
        }

    healthy_mask = None
    if leaf.size and actual_mask.size:
        healthy_mask = cv2.bitwise_and(leaf, cv2.bitwise_not(actual_mask))

    image_quality = _compute_image_quality(rgb, leaf, actual_mask)
    color = _compute_color_features(rgb, actual_mask, healthy_mask)
    texture = _compute_texture_features(rgb, actual_mask, healthy_mask)
    spatial = _compute_spatial_distribution(actual_mask, leaf, rgb.shape[:2])
    connectivity = _compute_connectivity(actual_mask)
    damage_pattern = _compute_damage_pattern(actual_mask)
    affected_area_geometry = _compute_affected_area_geometry(actual_mask, leaf)
    vein = _compute_vein_relationship(rgb, actual_mask, vein_map)
    cross_analysis = _compute_cross_analysis_consistency(gradcam, actual_mask, affected_leaf_percent, prediction)

    summary_text = _make_summary_text({
        "image_quality": image_quality,
        "morphology": morphology,
        "spatial_distribution": spatial,
        "connectivity": connectivity,
        "damage_pattern": damage_pattern,
        "color": color,
        "texture": texture,
        "vein_relationship": vein,
        "cross_analysis": cross_analysis,
        "affected_area_geometry": affected_area_geometry,
    })

    processing_time = time.perf_counter() - started
    interpretation = _interpret_visual_observations(prediction, {
        "image_quality": image_quality,
        "morphology": morphology,
        "spatial_distribution": spatial,
        "connectivity": connectivity,
        "damage_pattern": damage_pattern,
        "color": color,
        "texture": texture,
        "vein_relationship": vein,
        "cross_analysis": cross_analysis,
        "affected_area_geometry": affected_area_geometry,
    })

    result = {
        "image_quality": image_quality,
        "morphology": morphology,
        "spatial_distribution": spatial,
        "connectivity": connectivity,
        "color": color,
        "texture": texture,
        "vein_relationship": vein,
        "damage_pattern": damage_pattern,
        "affected_area_geometry": affected_area_geometry,
        "cross_analysis": cross_analysis,
        "summary": summary_text,
        "scientific_interpretation": interpretation,
        "processing_time_seconds": round(processing_time, 4),
        "mask_source": mask_source,
        "observation_mask": actual_mask,
    }

    return result
