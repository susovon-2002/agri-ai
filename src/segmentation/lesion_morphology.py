"""
AgriVision AI - Lesion-Level Morphology Analysis
=================================================

Computes per-region shape, colour, and texture features for every detected
infected region derived from the existing CV pipeline.

IMPORTANT SCIENTIFIC DISCLAIMER
  Lesion morphology measurements are calculated from model-derived infected-region
  masks. They are image-based estimates and are not ground-truth pathological
  measurements.  Region disease labels inherit the image-level ResNet18 prediction.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR  = PROJECT_ROOT / "data" / "reports" / "lesion_morphology"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Consistent colour palette (RGB) for region IDs
REGION_PALETTE_RGB = [
    (255,  80,  80),   # R1 - coral red
    ( 80, 160, 255),   # R2 - sky blue
    ( 80, 220, 100),   # R3 - lime green
    (255, 200,  50),   # R4 - gold
    (200,  80, 255),   # R5 - violet
    ( 50, 220, 220),   # R6 - cyan
    (255, 140,  50),   # R7 - orange
]


# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class LesionRegion:
    region_id: int
    disease: str
    friendly_disease: str

    # Geometry
    area_pixels: int
    area_percent_leaf: float
    perimeter_pixels: float
    bounding_box: Dict[str, int]        # x, y, w, h
    centroid: Dict[str, int]            # x, y
    contour_area: float
    convex_hull_area: float

    # Shape descriptors
    circularity: Optional[float]        # 4pi*A/P^2  (None if P=0)
    solidity: Optional[float]           # A/hull_area (None if hull=0)
    extent: float                       # A/bbox_area
    aspect_ratio: float                 # w/h

    # Colour (RGB + HSV)
    mean_rgb: List[float]
    std_rgb:  List[float]
    mean_hsv: List[float]
    std_hsv:  List[float]

    # Texture
    texture_mean: float
    texture_std:  float
    local_variance: float


# ============================================================
# CORE ANALYSIS
# ============================================================

def analyze_lesion_morphology(
    image: Any,
    leaf_mask: np.ndarray,
    infected_mask: np.ndarray,
    contours: List[np.ndarray],
    predicted_disease: str,
    friendly_disease: str,
    confidence: float,
) -> Tuple[List[LesionRegion], Dict[str, Any]]:
    """
    Compute shape, colour, and texture features for each infected region.

    Returns
    -------
    regions : List[LesionRegion]
    summary : Dict[str, Any]
    """
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = image.copy()

    h, w = rgb.shape[:2]
    hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap  = cv2.Laplacian(gray, cv2.CV_32F) ** 2  # local variance proxy

    leaf_px = int(np.sum(leaf_mask > 0))

    regions: List[LesionRegion] = []
    sorted_ctrs = sorted(contours, key=cv2.contourArea, reverse=True)

    for idx, cnt in enumerate(sorted_ctrs, 1):
        # Geometry
        area_pix  = float(cv2.contourArea(cnt))
        perimeter = float(cv2.arcLength(cnt, True))
        bx, by, bw, bh = cv2.boundingRect(cnt)

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = bx + bw // 2, by + bh // 2

        hull      = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        bbox_area = float(bw * bh) if bw * bh > 0 else 1.0
        area_pct  = (area_pix / leaf_px * 100.0) if leaf_px > 0 else 0.0

        # Shape descriptors
        if perimeter > 0:
            circ = round(4 * math.pi * area_pix / (perimeter ** 2), 4)
        else:
            circ = None

        solidity   = round(area_pix / hull_area, 4) if hull_area > 0 else None
        extent     = round(area_pix / bbox_area, 4)
        asp_ratio  = round(bw / bh, 4) if bh > 0 else 1.0

        # Pixel mask for contour
        cmask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cmask, [cnt], -1, 255, cv2.FILLED)
        px = cmask > 0
        if np.sum(px) == 0:
            cmask[by:by+bh, bx:bx+bw] = 255
            px = cmask > 0

        # Colour
        rgb_roi  = rgb[px].astype(np.float32)
        hsv_roi  = hsv[px]
        mean_rgb = [round(float(rgb_roi[:, c].mean()), 2) for c in range(3)]
        std_rgb  = [round(float(rgb_roi[:, c].std()),  2) for c in range(3)]
        mean_hsv = [round(float(hsv_roi[:, c].mean()), 2) for c in range(3)]
        std_hsv  = [round(float(hsv_roi[:, c].std()),  2) for c in range(3)]

        # Texture
        gray_roi  = gray[px]
        tex_mean  = round(float(gray_roi.mean()), 2)
        tex_std   = round(float(gray_roi.std()),  2)
        local_var = round(float(lap[px].mean()),  4)

        regions.append(LesionRegion(
            region_id=idx,
            disease=predicted_disease,
            friendly_disease=friendly_disease,
            area_pixels=int(area_pix),
            area_percent_leaf=round(area_pct, 3),
            perimeter_pixels=round(perimeter, 2),
            bounding_box={"x": bx, "y": by, "w": bw, "h": bh},
            centroid={"x": cx, "y": cy},
            contour_area=round(area_pix, 2),
            convex_hull_area=round(hull_area, 2),
            circularity=circ,
            solidity=solidity,
            extent=extent,
            aspect_ratio=asp_ratio,
            mean_rgb=mean_rgb, std_rgb=std_rgb,
            mean_hsv=mean_hsv, std_hsv=std_hsv,
            texture_mean=tex_mean, texture_std=tex_std,
            local_variance=local_var,
        ))

    summary = _build_summary(regions, predicted_disease, friendly_disease, confidence)
    return regions, summary


def _build_summary(
    regions: List[LesionRegion],
    disease: str, friendly: str, confidence: float,
) -> Dict[str, Any]:
    if not regions:
        return {
            "disease": disease, "friendly_disease": friendly,
            "confidence": round(confidence, 2), "total_regions": 0,
            "total_region_area_pct_leaf": 0.0,
            "largest_region_percent": 0.0, "smallest_region_percent": 0.0,
            "mean_region_area": 0.0, "median_region_area": 0.0,
            "mean_circularity": None, "mean_solidity": None, "mean_aspect_ratio": None,
        }

    areas = [r.area_percent_leaf for r in regions]
    circs = [r.circularity  for r in regions if r.circularity is not None]
    sols  = [r.solidity     for r in regions if r.solidity    is not None]
    ars   = [r.aspect_ratio for r in regions]

    return {
        "disease": disease, "friendly_disease": friendly,
        "confidence": round(confidence, 2),
        "total_regions": len(regions),
        "total_region_area_pct_leaf": round(sum(areas), 3),
        "largest_region_percent":  round(max(areas), 3),
        "smallest_region_percent": round(min(areas), 3),
        "mean_region_area":   round(float(np.mean(areas)),   3),
        "median_region_area": round(float(np.median(areas)), 3),
        "mean_circularity":   round(float(np.mean(circs)), 4) if circs else None,
        "mean_solidity":      round(float(np.mean(sols)),  4) if sols  else None,
        "mean_aspect_ratio":  round(float(np.mean(ars)),   4),
    }


# ============================================================
# VISUALISATION
# ============================================================

def draw_morphology_overlay(
    image: Any,
    regions: List[LesionRegion],
    contours: List[np.ndarray],
) -> np.ndarray:
    """Semi-transparent region overlay with consistent palette and ID badges."""
    if isinstance(image, Image.Image):
        base = np.array(image.convert("RGB"))
    else:
        base = image.copy()

    overlay = base.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for reg, cnt in zip(regions, contours):
        colour = REGION_PALETTE_RGB[(reg.region_id - 1) % len(REGION_PALETTE_RGB)]
        colour_f = np.array(colour, dtype=np.float32)

        # Fill
        mask_tmp = np.zeros(base.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_tmp, [cnt], -1, 255, cv2.FILLED)
        px = mask_tmp > 0
        overlay[px] = np.clip(
            overlay[px] * 0.45 + colour_f * 0.55, 0, 255
        ).astype(np.uint8)

        # Outline
        cv2.drawContours(overlay, [cnt], -1, colour, 2, lineType=cv2.LINE_AA)

        # Badge
        cx, cy = reg.centroid["x"], reg.centroid["y"]
        label = f"R{reg.region_id}"
        sub   = f"{reg.area_percent_leaf:.1f}%"
        fs = 0.40
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        (tw2, _), _ = cv2.getTextSize(sub,   font, fs, 1)
        bw_box = max(tw, tw2) + 10
        bh_box = th * 2 + 14
        bx_b = max(2, min(cx - bw_box // 2, base.shape[1] - bw_box - 2))
        by_b = max(2, min(cy - bh_box - 10, base.shape[0] - bh_box - 2))
        cv2.rectangle(overlay, (bx_b, by_b), (bx_b+bw_box, by_b+bh_box), (15, 15, 15), cv2.FILLED)
        cv2.rectangle(overlay, (bx_b, by_b), (bx_b+bw_box, by_b+bh_box), colour, 1, lineType=cv2.LINE_AA)
        cv2.putText(overlay, label, (bx_b+5, by_b+th+4),      font, fs, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(overlay, sub,   (bx_b+5, by_b+th*2+9),    font, fs, colour,         1, cv2.LINE_AA)

    return overlay


def crop_region_images(
    image: Any,
    regions: List[LesionRegion],
    padding: int = 10,
) -> List[np.ndarray]:
    """Return one padded crop per region with a colour border."""
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = image.copy()

    h, w = rgb.shape[:2]
    crops: List[np.ndarray] = []
    for reg in regions:
        bb = reg.bounding_box
        x1 = max(0, bb["x"] - padding)
        y1 = max(0, bb["y"] - padding)
        x2 = min(w, bb["x"] + bb["w"] + padding)
        y2 = min(h, bb["y"] + bb["h"] + padding)
        crop = rgb[y1:y2, x1:x2].copy()
        col  = REGION_PALETTE_RGB[(reg.region_id - 1) % len(REGION_PALETTE_RGB)]
        cv2.rectangle(crop, (0,0), (crop.shape[1]-1, crop.shape[0]-1), col, 2)
        crops.append(crop)
    return crops


# ============================================================
# REPORTING
# ============================================================

def save_morphology_reports(
    regions: List[LesionRegion],
    summary: Dict[str, Any],
    region_crops: List[np.ndarray],
    morphology_overlay: np.ndarray,
    output_dir=None,
) -> Dict[str, Any]:
    out_dir = Path(output_dir) if output_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Region crops
    crop_paths = []
    for i, crop in enumerate(region_crops, 1):
        p = out_dir / f"lesion_{i:03d}.png"
        cv2.imwrite(str(p), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        crop_paths.append(p)

    # Overlay
    overlay_path = out_dir / "morphology_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(morphology_overlay, cv2.COLOR_RGB2BGR))

    # JSON
    json_data = {
        "summary": summary,
        "regions": [asdict(r) for r in regions],
        "scientific_disclaimer": (
            "Lesion morphology measurements are calculated from model-derived "
            "infected-region masks. They are image-based estimates and are not "
            "ground-truth pathological measurements. Region disease labels "
            "inherit the image-level ResNet18 prediction."
        ),
    }
    json_path = out_dir / "lesion_morphology_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    summary_path = out_dir / "lesion_morphology_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # CSV
    csv_path = out_dir / "lesion_morphology_results.csv"
    fieldnames = [
        "region_id","disease","area_pixels","area_percent_leaf","perimeter_pixels",
        "circularity","solidity","extent","aspect_ratio",
        "texture_mean","texture_std","local_variance",
        "mean_R","mean_G","mean_B","mean_H","mean_S","mean_V",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in regions:
            writer.writerow({
                "region_id": r.region_id, "disease": r.disease,
                "area_pixels": r.area_pixels, "area_percent_leaf": r.area_percent_leaf,
                "perimeter_pixels": r.perimeter_pixels,
                "circularity": r.circularity, "solidity": r.solidity,
                "extent": r.extent, "aspect_ratio": r.aspect_ratio,
                "texture_mean": r.texture_mean, "texture_std": r.texture_std,
                "local_variance": r.local_variance,
                "mean_R": r.mean_rgb[0], "mean_G": r.mean_rgb[1], "mean_B": r.mean_rgb[2],
                "mean_H": r.mean_hsv[0], "mean_S": r.mean_hsv[1], "mean_V": r.mean_hsv[2],
            })

    return {
        "json": str(json_path), "summary_json": str(summary_path),
        "csv": str(csv_path), "overlay": str(overlay_path),
        "crops": [str(p) for p in crop_paths],
    }
