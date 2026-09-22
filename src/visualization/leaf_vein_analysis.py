"""
AgriVision AI - Leaf Vein / Vascular Structure Visualization & Real X-Ray Support
================================================================================

Classical Computer Vision module for extracting visible leaf vein networks
directly from RGB leaf photographs, and handling optional Real X-Ray image
registration.

SCIENTIFIC NOTICE:
- The leaf vein map is derived from visible optical features in the RGB photograph.
- It is NOT an X-ray, and must NEVER be labeled as "AI X-Ray".
- Real X-Ray images must strictly be supplied by the user as an independent modality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VEIN_REPORTS_DIR = PROJECT_ROOT / "data" / "reports" / "leaf_vein"
XRAY_REPORTS_DIR = PROJECT_ROOT / "data" / "reports" / "xray"

VEIN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
XRAY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SKELETONIZATION (PURE NUMPY/CV2)
# ============================================================

def _morphological_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Standard morphological skeletonization using iterative erosion and opening.
    Fast and robust on CPU without requiring external libraries like scikit-image.
    """
    img = binary_mask.copy()
    if img.max() > 1:
        img = (img > 0).astype(np.uint8)

    skeleton = np.zeros(img.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return (skeleton * 255).astype(np.uint8)


# ============================================================
# LEAF VEIN EXTRACTION (CLASSICAL CV)
# ============================================================

def extract_leaf_veins(
    image: np.ndarray | Image.Image,
    leaf_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Extract visible leaf vein structures from RGB image using classical CV.

    Pipeline:
        RGB -> Leaf mask suppression -> Grayscale -> CLAHE contrast enhancement ->
        Multi-scale morphological Black-Hat filtering ->
        Adaptive thresholding & morphological cleanup -> Skeletonization ->
        Vein structure map & metrics.
    """
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = image.copy()

    h, w = rgb.shape[:2]
    binary_leaf = (leaf_mask > 0).astype(np.uint8)
    total_leaf_pixels = int(np.sum(binary_leaf))

    if total_leaf_pixels == 0:
        empty = np.zeros((h, w), dtype=np.uint8)
        empty_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        return {
            "vein_mask": empty,
            "vein_skeleton": empty,
            "vein_map_rgb": empty_rgb,
            "metrics": {
                "detected_vein_pixels": 0,
                "vein_coverage_percent": 0.0,
                "skeleton_pixels": 0,
                "total_leaf_pixels": 0,
            }
        }

    # 1. Grayscale conversion & leaf background suppression
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_leaf = cv2.bitwise_and(gray, gray, mask=binary_leaf)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_leaf)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=binary_leaf)

    # 3. Multi-scale Black-Hat filtering to capture darker vascular channels (veins)
    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k_med   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    blackhat1 = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, k_small)
    blackhat2 = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, k_med)
    blackhat3 = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, k_large)

    # Combined multi-scale vein intensity
    vein_response = cv2.addWeighted(blackhat1, 0.5, blackhat2, 0.3, 0)
    vein_response = cv2.addWeighted(vein_response, 1.0, blackhat3, 0.2, 0)
    vein_response = cv2.bitwise_and(vein_response, vein_response, mask=binary_leaf)

    # 4. Bilateral filter to smooth noise while preserving vascular ridges
    smooth = cv2.bilateralFilter(vein_response, d=5, sigmaColor=50, sigmaSpace=50)

    # 5. Thresholding inside the leaf area
    leaf_vals = smooth[binary_leaf > 0]
    if len(leaf_vals) > 0 and leaf_vals.max() > 0:
        thresh_val = np.percentile(leaf_vals, 78)
        _, raw_vein = cv2.threshold(smooth, thresh_val, 255, cv2.THRESH_BINARY)
    else:
        raw_vein = np.zeros((h, w), dtype=np.uint8)

    # Clean small speckles
    clean_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned_vein = cv2.morphologyEx(raw_vein, cv2.MORPH_OPEN, clean_k)
    cleaned_vein = cv2.bitwise_and(cleaned_vein, cleaned_vein, mask=binary_leaf)

    # 6. Skeletonization of the vascular network
    skeleton = _morphological_skeleton(cleaned_vein)
    skeleton = cv2.bitwise_and(skeleton, skeleton, mask=binary_leaf)

    # 7. Scientific visualization: emerald/cyan vascular highlights over dimmed grayscale leaf
    base_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    base_dimmed = (base_gray * 0.45).astype(np.uint8)

    # Leaf contour boundary (green)
    leaf_contours, _ = cv2.findContours(binary_leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(base_dimmed, leaf_contours, -1, (40, 180, 70), 1, lineType=cv2.LINE_AA)

    # Draw vein network (high-contrast cyan/emerald glow)
    vein_map_rgb = base_dimmed.copy()
    vein_idx = cleaned_vein > 0
    vein_map_rgb[vein_idx] = (
        vein_map_rgb[vein_idx] * 0.25 + np.array([40, 230, 210]) * 0.75
    ).astype(np.uint8)

    # Skeleton overlay in bright white for centerlines
    skel_idx = skeleton > 0
    vein_map_rgb[skel_idx] = np.array([255, 255, 255], dtype=np.uint8)

    # Metrics
    vein_pixels = int(np.sum(cleaned_vein > 0))
    skel_pixels = int(np.sum(skeleton > 0))
    coverage_pct = round((vein_pixels / total_leaf_pixels * 100.0), 2) if total_leaf_pixels > 0 else 0.0

    metrics = {
        "detected_vein_pixels": vein_pixels,
        "vein_coverage_percent": coverage_pct,
        "skeleton_pixels": skel_pixels,
        "total_leaf_pixels": total_leaf_pixels,
    }

    return {
        "vein_mask": cleaned_vein,
        "vein_skeleton": skeleton,
        "vein_map_rgb": vein_map_rgb,
        "metrics": metrics,
    }


# ============================================================
# VEIN + INFECTED REGION OVERLAY
# ============================================================

def create_vein_overlay_with_regions(
    image: np.ndarray | Image.Image,
    leaf_mask: np.ndarray,
    vein_mask: np.ndarray,
    regions: List[Any],
    contours: List[np.ndarray],
) -> np.ndarray:
    """
    Produce a scientific overlay combining:
      - Dimmed leaf photograph
      - Detected leaf vein network (cyan)
      - Detected infected region outlines (red)
      - Region ID labels and leader lines
    """
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = image.copy()

    overlay = (rgb * 0.55).astype(np.uint8)
    binary_leaf = (leaf_mask > 0).astype(np.uint8)

    # 1. Draw leaf boundary
    leaf_contours, _ = cv2.findContours(binary_leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, leaf_contours, -1, (60, 200, 80), 1, lineType=cv2.LINE_AA)

    # 2. Draw vein network in glowing cyan
    v_idx = vein_mask > 0
    overlay[v_idx] = (overlay[v_idx] * 0.30 + np.array([30, 230, 220]) * 0.70).astype(np.uint8)

    # 3. Draw infected regions in clear red with badges
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = overlay.shape[:2]

    for reg, cnt in zip(regions, contours):
        # Outline in vivid red
        cv2.drawContours(overlay, [cnt], -1, (255, 40, 40), 2, lineType=cv2.LINE_AA)

        # Centroid
        if isinstance(reg, dict):
            cx = reg.get("centroid", {}).get("x", w // 2) if isinstance(reg.get("centroid"), dict) else reg.get("centroid", [w // 2, h // 2])[0]
            cy = reg.get("centroid", {}).get("y", h // 2) if isinstance(reg.get("centroid"), dict) else reg.get("centroid", [w // 2, h // 2])[1]
            reg_id = reg.get("region_id", 1)
        else:
            cx = reg.centroid["x"] if isinstance(reg.centroid, dict) else reg.centroid[0]
            cy = reg.centroid["y"] if isinstance(reg.centroid, dict) else reg.centroid[1]
            reg_id = reg.region_id

        # Centroid dot
        cv2.circle(overlay, (cx, cy), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        # Badge
        badge_text = f"R{reg_id}"
        (tw, th), _ = cv2.getTextSize(badge_text, font, 0.45, 1)
        bx = max(2, min(cx - tw // 2 - 4, w - tw - 12))
        by = max(2, min(cy - th - 8, h - th - 10))

        cv2.rectangle(overlay, (bx, by), (bx + tw + 8, by + th + 6), (20, 20, 20), cv2.FILLED)
        cv2.rectangle(overlay, (bx, by), (bx + tw + 8, by + th + 6), (255, 50, 50), 1, lineType=cv2.LINE_AA)
        cv2.putText(overlay, badge_text, (bx + 4, by + th + 2), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


# ============================================================
# REGION-SPECIFIC VEIN CROPS & VISIBILITY ASSESSMENT
# ============================================================

def crop_region_veins(
    image: np.ndarray | Image.Image,
    vein_mask: np.ndarray,
    regions: List[Any],
    padding: int = 8,
) -> List[Dict[str, Any]]:
    """
    Extract side-by-side original and vein crops for each detected infected region,
    evaluating vein visibility inside the lesion boundary.
    """
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = image.copy()

    h, w = rgb.shape[:2]
    results = []

    for reg in regions:
        if isinstance(reg, dict):
            bb = reg.get("bbox") or reg.get("bounding_box", {"x": 0, "y": 0, "w": w, "h": h})
            reg_id = reg.get("region_id", 1)
            disease = reg.get("friendly_disease", reg.get("disease", "Unknown"))
            area_pct = reg.get("area_percent_leaf", reg.get("area_percent_of_leaf", 0.0))
        else:
            bb = reg.bounding_box if hasattr(reg, "bounding_box") else reg.bbox
            reg_id = reg.region_id
            disease = getattr(reg, "friendly_disease", getattr(reg, "disease", "Unknown"))
            area_pct = getattr(reg, "area_percent_leaf", getattr(reg, "area_percent_of_leaf", 0.0))

        if isinstance(bb, list):
            bx, by, bw, bh = bb[0], bb[1], bb[2], bb[3]
        else:
            bx, by, bw, bh = bb["x"], bb["y"], bb["w"], bb["h"]

        x1 = max(0, bx - padding)
        y1 = max(0, by - padding)
        x2 = min(w, bx + bw + padding)
        y2 = min(h, by + bh + padding)

        orig_crop = rgb[y1:y2, x1:x2].copy()
        vein_sub = vein_mask[y1:y2, x1:x2]

        # Vein crop visualization (cyan overlay)
        vein_crop = (orig_crop * 0.40).astype(np.uint8)
        v_idx = vein_sub > 0
        vein_crop[v_idx] = np.array([40, 230, 215], dtype=np.uint8)

        # Outline
        cv2.rectangle(orig_crop, (0, 0), (orig_crop.shape[1] - 1, orig_crop.shape[0] - 1), (255, 60, 60), 2)
        cv2.rectangle(vein_crop, (0, 0), (vein_crop.shape[1] - 1, vein_crop.shape[0] - 1), (40, 220, 200), 2)

        # Visibility assessment based on vein pixel density inside region
        vein_px_inside = int(np.sum(v_idx))
        crop_total_px = max(1, (x2 - x1) * (y2 - y1))
        density = (vein_px_inside / crop_total_px) * 100.0

        if density >= 4.0:
            visibility = "visible"
        elif density >= 1.0:
            visibility = "limited"
        else:
            visibility = "not detected"

        results.append({
            "region_id": reg_id,
            "disease": disease,
            "area_percent_leaf": area_pct,
            "orig_crop": orig_crop,
            "vein_crop": vein_crop,
            "vein_density_percent": round(density, 2),
            "vein_visibility": visibility,
        })

    return results


# ============================================================
# REAL X-RAY REGISTRATION (IMAGE ALIGNMENT)
# ============================================================

def register_xray_to_rgb(
    rgb_image: np.ndarray | Image.Image,
    xray_image: np.ndarray | Image.Image,
) -> Tuple[Optional[np.ndarray], str, Optional[np.ndarray]]:
    """
    Attempt feature-based affine/homography registration between RGB leaf and Real X-Ray.
    """
    if isinstance(rgb_image, Image.Image):
        rgb = np.array(rgb_image.convert("RGB"))
    else:
        rgb = rgb_image.copy()

    if isinstance(xray_image, Image.Image):
        xray = np.array(xray_image.convert("RGB"))
    else:
        xray = xray_image.copy()

    h, w = rgb.shape[:2]
    xray_resized = cv2.resize(xray, (w, h))

    gray_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_xray = cv2.cvtColor(xray_resized, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray_rgb, None)
    kp2, des2 = orb.detectAndCompute(gray_xray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return xray_resized, "Alignment unavailable", None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    good_matches = [m for m in matches if m.distance < 60]

    if len(good_matches) < 8:
        return xray_resized, "Manual alignment required", None

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    try:
        M, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None or (inliers is not None and np.sum(inliers) < 6):
            return xray_resized, "Manual alignment required", None

        registered = cv2.warpPerspective(xray_resized, M, (w, h))
        overlay = cv2.addWeighted(rgb, 0.6, registered, 0.4, 0)
        return registered, "Aligned", overlay
    except Exception:
        return xray_resized, "Alignment unavailable", None


# ============================================================
# PERSISTENCE & REPORTS
# ============================================================

def save_leaf_vein_reports(
    image_name: str,
    vein_map_rgb: np.ndarray,
    vein_skeleton: np.ndarray,
    vein_regions_rgb: np.ndarray,
    metrics: Dict[str, Any],
    has_real_xray: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    out_dir = Path(output_dir) if output_dir else VEIN_REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    map_path = out_dir / "leaf_vein_map.png"
    skel_path = out_dir / "leaf_vein_skeleton.png"
    regions_path = out_dir / "leaf_vein_regions.png"
    json_path = out_dir / "leaf_vein_results.json"

    cv2.imwrite(str(map_path), cv2.cvtColor(vein_map_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(skel_path), vein_skeleton)
    cv2.imwrite(str(regions_path), cv2.cvtColor(vein_regions_rgb, cv2.COLOR_RGB2BGR))

    payload = {
        "image": image_name,
        "leaf_vein_analysis": True,
        "method": "computer_vision",
        "real_xray": has_real_xray,
        "detected_vein_pixels": metrics.get("detected_vein_pixels", 0),
        "vein_coverage_percent": metrics.get("vein_coverage_percent", 0.0),
        "skeleton_pixels": metrics.get("skeleton_pixels", 0),
        "disclaimer": (
            "Detected leaf vein structure is derived via classical computer vision "
            "from visible optical features in the RGB photograph. It is NOT an X-ray "
            "and may contain segmentation artifacts."
        )
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return {
        "map": map_path,
        "skeleton": skel_path,
        "regions": regions_path,
        "json": json_path,
    }


def save_xray_reports(
    image_name: str,
    xray_image: np.ndarray,
    output_dir: Optional[Path] = None,
) -> Path:
    out_dir = Path(output_dir) if output_dir else XRAY_REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"real_xray_{image_name}.png"
    cv2.imwrite(str(p), cv2.cvtColor(xray_image, cv2.COLOR_RGB2BGR))
    return p
