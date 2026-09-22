"""
AgriVision AI - Infected Region Detection, Circle Marking & Disease Labeling
=============================================================================

Computer vision engine for detecting, delineating, circling, labeling, and
quantifying infected regions directly on the ORIGINAL leaf image.

Pipeline:
    Original RGB image + Leaf Mask + Grad-CAM attention
          ↓
    Infected-region binary mask (Restricted strictly to leaf tissue)
          ↓
    Connected components / contour extraction & noise filtering
          ↓
    Region metrics (centroid, bbox, enclosing circle, pixel area, % leaf area)
          ↓
    Visual Annotation (enclosing circles, contour outlines, leader lines, labels)
          ↓
    Structured JSON reporting & artifact persistence
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports" / "infected_regions"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Default configuration parameters
DEFAULT_CAM_THRESHOLD = 0.50
DEFAULT_MIN_REGION_AREA_PERCENT = 0.30  # % of total leaf area to filter out noise
DEFAULT_MIN_REGION_PIXELS = 15          # minimum connected component pixel area


@dataclass
class InfectedRegion:
    region_id: int
    disease: str
    friendly_disease: str
    crop: str
    pixel_area: int
    area_percent_of_leaf: float
    area_percent_of_image: float
    bbox: List[int]                     # [x, y, w, h]
    centroid: List[int]                 # [cx, cy]
    circle: Dict[str, Any]              # {"center": [cx, cy], "radius": int}
    mean_cam_attention: float
    confidence: float


def detect_infected_regions(
    image: np.ndarray | Image.Image,
    leaf_mask: np.ndarray,
    gradcam: np.ndarray,
    predicted_disease: str,
    friendly_disease: str,
    crop: str,
    confidence: float,
    cam_threshold: float = DEFAULT_CAM_THRESHOLD,
    min_region_area_percent: float = DEFAULT_MIN_REGION_AREA_PERCENT,
    min_region_pixels: int = DEFAULT_MIN_REGION_PIXELS,
) -> Tuple[List[InfectedRegion], np.ndarray, List[np.ndarray]]:
    """
    Generate binary infected-region mask and extract individual connected regions.

    Parameters
    ----------
    image : np.ndarray or PIL.Image
        RGB image of size (H, W, 3)
    leaf_mask : np.ndarray
        Binary or uint8 leaf segmentation mask (H, W)
    gradcam : np.ndarray
        Grad-CAM activation array (H, W) in range [0.0, 1.0]
    predicted_disease : str
        Global model prediction identifier
    friendly_disease : str
        Human-readable disease label
    crop : str
        Inferred crop species name
    confidence : float
        Image-level model confidence (e.g. 97.07)
    cam_threshold : float
        Activation threshold for disease attention (default 0.50)
    min_region_area_percent : float
        Minimum percentage of leaf area required for a valid region (noise rejection)
    min_region_pixels : int
        Minimum pixel count required for a connected component

    Returns
    -------
    regions : List[InfectedRegion]
        List of structured region records
    infected_mask : np.ndarray
        Binary uint8 mask (0 or 255) of all detected infected areas
    contours : List[np.ndarray]
        OpenCV contours corresponding to each valid region
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image.copy()

    h, w = image_np.shape[:2]
    total_image_pixels = h * w

    # 1. Normalize gradcam to [0.0, 1.0] if needed
    cam = gradcam.astype(np.float32)
    if cam.max() > 1.0:
        cam = cam / 255.0

    # 2. Strict leaf restriction: fused mask = (leaf_mask > 0) & (gradcam >= cam_threshold)
    leaf_binary = (leaf_mask > 0)
    total_leaf_pixels = int(np.sum(leaf_binary))

    cam_binary = (cam >= cam_threshold)
    fused_binary = leaf_binary & cam_binary
    infected_mask = (fused_binary.astype(np.uint8) * 255)

    if total_leaf_pixels == 0 or np.sum(infected_mask) == 0:
        return [], infected_mask, []

    # 3. Find connected contours
    raw_contours, _ = cv2.findContours(
        infected_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    regions: List[InfectedRegion] = []
    valid_contours: List[np.ndarray] = []

    # Sort raw contours by area descending
    sorted_contours = sorted(
        raw_contours,
        key=cv2.contourArea,
        reverse=True
    )

    region_counter = 1
    for cnt in sorted_contours:
        pixel_area = int(cv2.contourArea(cnt))
        if pixel_area < min_region_pixels:
            continue

        area_pct_leaf = (pixel_area / total_leaf_pixels) * 100.0 if total_leaf_pixels > 0 else 0.0
        area_pct_image = (pixel_area / total_image_pixels) * 100.0

        if area_pct_leaf < min_region_area_percent:
            continue

        # Bounding box
        bx, by, bw, bh = cv2.boundingRect(cnt)

        # Centroid calculation via moments
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = bx + bw // 2
            cy = by + bh // 2

        # Minimum enclosing circle
        (circle_cx, circle_cy), radius = cv2.minEnclosingCircle(cnt)
        radius = max(6, int(radius))

        # Mean CAM attention within region
        cnt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        mean_cam = float(np.mean(cam[cnt_mask > 0])) if np.sum(cnt_mask > 0) > 0 else float(cam_threshold)

        region = InfectedRegion(
            region_id=region_counter,
            disease=predicted_disease,
            friendly_disease=friendly_disease,
            crop=crop,
            pixel_area=pixel_area,
            area_percent_of_leaf=round(area_pct_leaf, 2),
            area_percent_of_image=round(area_pct_image, 2),
            bbox=[int(bx), int(by), int(bw), int(bh)],
            centroid=[int(cx), int(cy)],
            circle={
                "center": [int(circle_cx), int(circle_cy)],
                "radius": int(radius)
            },
            mean_cam_attention=round(mean_cam, 4),
            confidence=round(confidence, 2)
        )

        regions.append(region)
        valid_contours.append(cnt)
        region_counter += 1

    return regions, infected_mask, valid_contours


def annotate_infected_regions(
    image: np.ndarray | Image.Image,
    regions: List[InfectedRegion],
    contours: List[np.ndarray],
    draw_circles: bool = True,
    draw_contours: bool = True,
    draw_labels: bool = True,
) -> np.ndarray:
    """
    Draw circles/ellipses, contour outlines, leader lines, and disease labels
    directly onto the ORIGINAL RGB leaf image while keeping leaf details visible.

    Parameters
    ----------
    image : np.ndarray or PIL.Image
        RGB image of size (H, W, 3)
    regions : List[InfectedRegion]
        List of detected region records
    contours : List[np.ndarray]
        List of OpenCV contours corresponding to each region

    Returns
    -------
    annotated_rgb : np.ndarray
        Annotated RGB image
    """
    if isinstance(image, Image.Image):
        annotated = np.array(image.convert("RGB"))
    else:
        annotated = image.copy()

    h, w = annotated.shape[:2]

    if not regions:
        # If no regions, return clean image
        return annotated

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    font_thickness = 1

    for reg, cnt in zip(regions, contours):
        cx, cy = reg.centroid
        circle_cx, circle_cy = reg.circle["center"]
        radius = reg.circle["radius"]

        # 1. Draw contour boundary outline (bright red/coral)
        if draw_contours:
            cv2.drawContours(
                annotated,
                [cnt],
                -1,
                (255, 60, 60),
                2,
                lineType=cv2.LINE_AA
            )

        # 2. Draw bounding box (subtle amber/gold rectangle)
        bx, by, bw, bh = reg.bbox
        cv2.rectangle(
            annotated,
            (bx, by),
            (bx + bw, by + bh),
            (255, 190, 40),
            1,
            lineType=cv2.LINE_AA
        )

        # 3. Draw enclosing circle around the detected region
        if draw_circles:
            # Dual outline for maximum visibility against varied backgrounds
            cv2.circle(
                annotated,
                (circle_cx, circle_cy),
                radius + 3,
                (255, 255, 255),
                3,
                lineType=cv2.LINE_AA
            )
            cv2.circle(
                annotated,
                (circle_cx, circle_cy),
                radius + 3,
                (230, 30, 30),
                2,
                lineType=cv2.LINE_AA
            )

            # Centroid indicator dot
            cv2.circle(
                annotated,
                (cx, cy),
                3,
                (255, 255, 255),
                -1,
                lineType=cv2.LINE_AA
            )

        # 4. Draw readable label box with leader line
        if draw_labels:
            line1 = f"R{reg.region_id}"
            line2 = f"{reg.friendly_disease}"
            line3 = f"{reg.area_percent_of_leaf:.1f}% region area"

            (w1, h1), _ = cv2.getTextSize(line1, font, font_scale + 0.05, 1)
            (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, font_thickness)
            (w3, h3), _ = cv2.getTextSize(line3, font, font_scale, font_thickness)

            box_w = max(w1, w2, w3) + 12
            box_h = h1 + h2 + h3 + 18

            # Candidate placement: above circle if space allows, otherwise below
            target_box_x = circle_cx - box_w // 2
            target_box_y = circle_cy - radius - box_h - 10

            if target_box_y < 6:
                target_box_y = circle_cy + radius + 10

            # Clamp coordinates within image bounds
            box_x = max(4, min(w - box_w - 4, target_box_x))
            box_y = max(4, min(h - box_h - 4, target_box_y))

            # Leader line from label box to centroid
            anchor_x = box_x + box_w // 2
            anchor_y = box_y + box_h if box_y < cy else box_y

            cv2.line(
                annotated,
                (anchor_x, anchor_y),
                (cx, cy),
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA
            )
            cv2.line(
                annotated,
                (anchor_x, anchor_y),
                (cx, cy),
                (220, 20, 20),
                1,
                lineType=cv2.LINE_AA
            )

            # Draw background box (dark charcoal with red border)
            cv2.rectangle(
                annotated,
                (box_x, box_y),
                (box_x + box_w, box_y + box_h),
                (20, 20, 20),
                thickness=cv2.FILLED
            )
            cv2.rectangle(
                annotated,
                (box_x, box_y),
                (box_x + box_w, box_y + box_h),
                (230, 40, 40),
                thickness=1,
                lineType=cv2.LINE_AA
            )

            # Draw text
            text_x = box_x + 6
            text_y1 = box_y + h1 + 4
            text_y2 = text_y1 + h2 + 4
            text_y3 = text_y2 + h3 + 4

            cv2.putText(
                annotated,
                line1,
                (text_x, text_y1),
                font,
                font_scale + 0.05,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA
            )
            cv2.putText(
                annotated,
                line2,
                (text_x, text_y2),
                font,
                font_scale,
                (255, 220, 80),
                font_thickness,
                lineType=cv2.LINE_AA
            )
            cv2.putText(
                annotated,
                line3,
                (text_x, text_y3),
                font,
                font_scale,
                (100, 220, 255),
                font_thickness,
                lineType=cv2.LINE_AA
            )

    # Add subtle explanatory callout badge in corner
    callout_txt = "Detected Attention Region (Grad-CAM + Leaf Mask)"
    (cw, ch), _ = cv2.getTextSize(callout_txt, font, 0.32, 1)
    cv2.rectangle(annotated, (4, h - ch - 8), (cw + 12, h - 3), (15, 15, 15), cv2.FILLED)
    cv2.rectangle(annotated, (4, h - ch - 8), (cw + 12, h - 3), (80, 80, 80), 1, lineType=cv2.LINE_AA)
    cv2.putText(annotated, callout_txt, (8, h - 6), font, 0.32, (200, 200, 200), 1, lineType=cv2.LINE_AA)

    return annotated


# ============================================================
# FOUR-PANEL DETAILED AI VISUAL EXPLANATION
# ============================================================

def create_four_panel_explanation(
    original_image: np.ndarray | Image.Image,
    gradcam_image: np.ndarray | Image.Image,
    fusion_image: np.ndarray | Image.Image,
    annotated_image: np.ndarray | Image.Image,
    panel_size: int = 256,
) -> np.ndarray:
    """
    Create a composite 4-panel visual explanation image:
    ┌──────────────────────┬──────────────────────┐
    │ 1. Original Leaf     │ 2. Grad-CAM          │
    ├──────────────────────┼──────────────────────┤
    │ 3. Leaf + Attention  │ 4. AI Detected       │
    └──────────────────────┴──────────────────────┘
    """
    def _to_rgb(img, target_size=(256, 256)):
        if isinstance(img, Image.Image):
            arr = np.array(img.convert("RGB"))
        else:
            arr = img.copy()
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return cv2.resize(arr, target_size)

    orig_p = _to_rgb(original_image, (panel_size, panel_size))
    cam_p  = _to_rgb(gradcam_image, (panel_size, panel_size))
    fuse_p = _to_rgb(fusion_image, (panel_size, panel_size))
    ann_p  = _to_rgb(annotated_image, (panel_size, panel_size))

    header_h = 30
    def _add_header(panel, text):
        h, w = panel.shape[:2]
        canvas = np.zeros((h + header_h, w, 3), dtype=np.uint8)
        canvas[:header_h, :] = (25, 25, 25)
        canvas[header_h:, :] = panel
        cv2.putText(canvas, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (w - 1, h + header_h - 1), (60, 60, 60), 1)
        return canvas

    p1 = _add_header(orig_p, "1. Original Leaf Photograph")
    p2 = _add_header(cam_p,  "2. Grad-CAM Model Attention")
    p3 = _add_header(fuse_p, "3. Leaf Mask + Attention Fusion")
    p4 = _add_header(ann_p,  "4. Model-Highlighted Attention Regions")

    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    four_panel = np.vstack([top_row, bot_row])

    # Footer banner with scientific disclaimer
    footer_h = 26
    total_w = four_panel.shape[1]
    footer = np.zeros((footer_h, total_w, 3), dtype=np.uint8)
    footer[:] = (18, 18, 18)
    cv2.putText(
        footer,
        "AgriVision AI Visual Explanation: Attention regions derived from ResNet18 Grad-CAM & leaf mask.",
        (12, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )
    return np.vstack([four_panel, footer])


# ============================================================
# PERSISTENCE & REPORTS
# ============================================================

DETAILED_EXPLANATION_DIR = REPORTS_DIR / "detailed_explanation"
DETAILED_EXPLANATION_DIR.mkdir(parents=True, exist_ok=True)


def save_detailed_visual_explanation(
    four_panel_image: np.ndarray,
    annotated_image: np.ndarray,
    gradcam_image: np.ndarray,
    fusion_image: np.ndarray,
    segmentation_mask: np.ndarray,
    detailed_results: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Save the 4-panel visual explanation and associated scientific layers.
    """
    out_dir = Path(output_dir) if output_dir else DETAILED_EXPLANATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    det_path = out_dir / "detailed_explanation.png"
    ann_path = out_dir / "annotated_regions.png"
    cam_path = out_dir / "gradcam.png"
    fuse_path = out_dir / "fusion.png"
    mask_path = out_dir / "segmentation_mask.png"
    json_path = out_dir / "detailed_region_results.json"

    def _save_rgb(p, img):
        if isinstance(img, Image.Image):
            arr = np.array(img.convert("RGB"))
        else:
            arr = img
        cv2.imwrite(str(p), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    _save_rgb(det_path, four_panel_image)
    _save_rgb(ann_path, annotated_image)
    _save_rgb(cam_path, gradcam_image)
    _save_rgb(fuse_path, fusion_image)
    cv2.imwrite(str(mask_path), segmentation_mask)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    return {
        "detailed_explanation": det_path,
        "annotated_regions": ann_path,
        "gradcam": cam_path,
        "fusion": fuse_path,
        "segmentation_mask": mask_path,
        "detailed_json": json_path,
    }


def save_infected_region_reports(
    original_image: np.ndarray | Image.Image,
    annotated_image: np.ndarray,
    infected_mask: np.ndarray,
    summary_data: Dict[str, Any],
    output_dir: Optional[Path] = None,
    base_name: str = "leaf"
) -> Dict[str, Path]:
    """
    Save original image, binary infected mask, annotated image,
    and structured JSON results to disk.
    """
    out_dir = Path(output_dir) if output_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(original_image, Image.Image):
        orig_np = np.array(original_image.convert("RGB"))
    else:
        orig_np = original_image.copy()

    orig_path = out_dir / f"original_{base_name}.png"
    mask_path = out_dir / f"mask_{base_name}.png"
    annotated_path = out_dir / f"annotated_{base_name}.png"
    json_path = out_dir / "infected_region_results.json"

    # Save PNGs using cv2 (RGB -> BGR conversion)
    cv2.imwrite(str(orig_path), cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), infected_mask)
    cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Also save standard filename requested in specification
    spec_annotated_path = out_dir / "infected_regions_annotated.png"
    cv2.imwrite(str(spec_annotated_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    spec_mask_path = out_dir / "infected_region_mask.png"
    cv2.imwrite(str(spec_mask_path), infected_mask)

    spec_orig_path = out_dir / "original.png"
    cv2.imwrite(str(spec_orig_path), cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR))

    # Save JSON report
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    return {
        "original": orig_path,
        "mask": mask_path,
        "annotated": annotated_path,
        "json": json_path,
        "spec_annotated": spec_annotated_path
    }

