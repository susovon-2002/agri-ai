"""
AgriVision AI - Inference Core
==============================

Pure-Python inference module: no Streamlit dependency.

Exports:
    CLASS_NAMES       - list of 15 class labels
    FRIENDLY_NAMES    - human-readable label map
    IMAGE_SIZE        - 224
    CAM_THRESHOLD     - 0.50
    DEVICE            - torch.device (cuda / cpu)
    transform         - torchvision preprocessing pipeline
    load_model()      - returns a loaded ResNet18 (plain function, no @st.cache)
    segment_leaf()    - HSV-based leaf mask
    analyze_image()   - full inference pipeline -> result dict
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms, models
from src.gradcam_compat import GradCAM, ClassifierOutputTarget
from dataclasses import asdict

from src.segmentation.infected_region_detector import (
    detect_infected_regions,
    annotate_infected_regions,
    save_infected_region_reports,
    create_four_panel_explanation,
    save_detailed_visual_explanation,
)

from src.segmentation.lesion_morphology import (
    analyze_lesion_morphology,
    draw_morphology_overlay,
    crop_region_images,
    save_morphology_reports,
)

from src.visualization.leaf_vein_analysis import (
    extract_leaf_veins,
    create_vein_overlay_with_regions,
    crop_region_veins,
    save_leaf_vein_reports,
)

from src.vision.advanced_observation import analyze_leaf_observations
from src.knowledge.advanced_interpretation import generate_advanced_interpretation
from src.knowledge.farmer_report import generate_report as generate_knowledge_report


# ============================================================
# CONFIGURATION
# ============================================================

# This file lives at <project>/src/inference/inference_engine.py
# so parents[2] is the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = _PROJECT_ROOT / "models" / "classification" / "resnet18_best.pth"

IMAGE_SIZE = 224

CAM_THRESHOLD = 0.50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]


# ============================================================
# FRIENDLY NAMES
# ============================================================

FRIENDLY_NAMES = {
    "Pepper__bell___Bacterial_spot":        "Pepper Bacterial Spot",
    "Pepper__bell___healthy":               "Healthy Pepper",
    "Potato___Early_blight":                "Potato Early Blight",
    "Potato___Late_blight":                 "Potato Late Blight",
    "Potato___healthy":                     "Healthy Potato",
    "Tomato_Bacterial_spot":                "Tomato Bacterial Spot",
    "Tomato_Early_blight":                  "Tomato Early Blight",
    "Tomato_Late_blight":                   "Tomato Late Blight",
    "Tomato_Leaf_Mold":                     "Tomato Leaf Mold",
    "Tomato_Septoria_leaf_spot":            "Tomato Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato Two-Spotted Spider Mites",
    "Tomato__Target_Spot":                  "Tomato Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus":"Tomato Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus":          "Tomato Mosaic Virus",
    "Tomato_healthy":                       "Healthy Tomato"
}


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Build and return the ResNet18 model loaded from checkpoint.

    Plain function (no @st.cache_resource) so it can be called
    from outside a Streamlit context.
    The Streamlit app wraps this with @st.cache_resource in app.py.
    """
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, len(CLASS_NAMES))
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# LEAF SEGMENTATION
# ============================================================

def segment_leaf(image):
    """
    HSV-based leaf mask.

    Parameters
    ----------
    image : np.ndarray | PIL.Image.Image  (H, W, 3)  RGB uint8

    Returns
    -------
    np.ndarray  (H, W)  uint8 binary mask (0 or 255)
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    green_mask = cv2.inRange(
        hsv,
        np.array([20, 25, 20],  dtype=np.uint8),
        np.array([100, 255, 255], dtype=np.uint8)
    )

    saturation_mask = cv2.inRange(
        hsv,
        np.array([15, 20, 20],  dtype=np.uint8),
        np.array([110, 255, 255], dtype=np.uint8)
    )

    mask = cv2.bitwise_or(green_mask, saturation_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    image_area = IMAGE_SIZE * IMAGE_SIZE
    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] > image_area * 0.01:
            cleaned[labels == label] = 255

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    final_mask = np.zeros_like(cleaned)
    for contour in contours:
        if cv2.contourArea(contour) > image_area * 0.01:
            cv2.drawContours(
                final_mask, [contour], -1, 255, thickness=cv2.FILLED
            )

    return final_mask


# ============================================================
# CROP INFERENCE HELPER
# ============================================================

def infer_crop(prediction: str) -> str:
    """Infer crop name from disease class prediction string."""
    p = str(prediction).lower()
    if "pepper" in p:
        return "Pepper"
    if "potato" in p:
        return "Potato"
    if "tomato" in p:
        return "Tomato"
    return "Unknown"


# ============================================================
# ANALYSIS
# ============================================================

def analyze_image(image, model):
    """
    Full AgriVision inference pipeline on a PIL Image.

    Parameters
    ----------
    image : PIL.Image.Image
    model : torch.nn.Module  (ResNet18, eval mode)

    Returns
    -------
    dict:
        prediction, friendly_prediction, confidence, top3,
        leaf_area, attention_area, affected_leaf, severity,
        gradcam, disease_region
    """
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb   = np.array(image)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # ---- Prediction ------------------------------------------------
    with torch.no_grad():
        output        = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        top_values, top_indices = torch.topk(probabilities, k=3, dim=1)
        predicted_class = top_indices[0, 0].item()
        confidence      = top_values[0, 0].item() * 100

    predicted_name = CLASS_NAMES[predicted_class]

    # ---- Top-3 -----------------------------------------------------
    top3 = []
    for rank in range(3):
        class_id = top_indices[0, rank].item()
        score    = top_values[0, rank].item() * 100
        top3.append({
            "class":      CLASS_NAMES[class_id],
            "friendly":   FRIENDLY_NAMES.get(CLASS_NAMES[class_id], CLASS_NAMES[class_id]),
            "confidence": score
        })

    # ---- Grad-CAM --------------------------------------------------
    targets      = [ClassifierOutputTarget(predicted_class)]
    target_layers = [model.layer4[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # ---- Leaf mask -------------------------------------------------
    leaf_mask   = segment_leaf(rgb)
    leaf_binary = leaf_mask > 0

    # ---- Grad-CAM fusion -------------------------------------------
    print(
        f"GRAD-CAM DEBUG: min={grayscale_cam.min():.6f}, "
        f"max={grayscale_cam.max():.6f}, "
        f"mean={grayscale_cam.mean():.6f}, "
        f"threshold={CAM_THRESHOLD:.2f}"
    )

    cam_binary = grayscale_cam >= CAM_THRESHOLD
    fused      = leaf_binary & cam_binary

    # ---- Area ------------------------------------------------------
    leaf_pixels  = np.sum(leaf_binary)
    fused_pixels = np.sum(fused)
    total_pixels = IMAGE_SIZE * IMAGE_SIZE

    print(
        f"AREA DEBUG: leaf_pixels={leaf_pixels}, "
        f"fused_pixels={fused_pixels}, "
        f"leaf_ratio={(leaf_pixels / total_pixels) * 100:.2f}%"
    )

    leaf_area      = (leaf_pixels  / total_pixels) * 100
    attention_area = (fused_pixels / total_pixels) * 100
    affected_leaf  = (fused_pixels / leaf_pixels) * 100 if leaf_pixels > 0 else 0.0

    # ---- Severity --------------------------------------------------
    if "healthy" in predicted_name.lower():
        severity = "Healthy"
    elif affected_leaf < 10:
        severity = "Low"
    elif affected_leaf < 25:
        severity = "Moderate"
    else:
        severity = "High"

    # ---- Grad-CAM overlay ------------------------------------------
    heatmap = cv2.applyColorMap(
        np.uint8(grayscale_cam * 255), cv2.COLORMAP_JET
    )
    heatmap        = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    gradcam_overlay = cv2.addWeighted(rgb, 0.60, heatmap, 0.40, 0)

    # ---- Disease-region overlay ------------------------------------
    disease_mask    = fused.astype(np.uint8) * 255
    disease_heatmap = cv2.applyColorMap(disease_mask, cv2.COLORMAP_JET)
    disease_heatmap = cv2.cvtColor(disease_heatmap, cv2.COLOR_BGR2RGB)
    disease_overlay = cv2.addWeighted(rgb, 0.60, disease_heatmap, 0.40, 0)

    debug_info = {
        "leaf_pixels": int(leaf_pixels),
        "fused_pixels": int(fused_pixels),
        "total_pixels": int(total_pixels),
        "cam_min": float(grayscale_cam.min()),
        "cam_max": float(grayscale_cam.max()),
        "cam_mean": float(grayscale_cam.mean()),
        "cam_threshold": float(CAM_THRESHOLD),
    }

    # ---- Infected Region Detection & Circle Marking ----------------
    friendly_pred = FRIENDLY_NAMES.get(predicted_name, predicted_name)
    crop_name = infer_crop(predicted_name)

    regions, infected_mask, region_contours = detect_infected_regions(
        image=rgb,
        leaf_mask=leaf_mask,
        gradcam=grayscale_cam,
        predicted_disease=predicted_name,
        friendly_disease=friendly_pred,
        crop=crop_name,
        confidence=confidence,
        cam_threshold=CAM_THRESHOLD,
    )

    annotated_infected = annotate_infected_regions(
        image=rgb,
        regions=regions,
        contours=region_contours,
    )

    regions_list = [asdict(r) for r in regions]

    # Persist latest reports to data/reports/infected_regions/
    try:
        summary_report = {
            "crop": crop_name,
            "disease": predicted_name,
            "friendly_disease": friendly_pred,
            "confidence": round(confidence / 100.0, 4) if confidence > 1.0 else round(confidence, 4),
            "total_leaf_area_percent": round(leaf_area, 2),
            "affected_leaf_percent": round(affected_leaf, 2),
            "total_regions": len(regions),
            "regions": regions_list,
        }
        save_infected_region_reports(
            original_image=rgb,
            annotated_image=annotated_infected,
            infected_mask=infected_mask,
            summary_data=summary_report,
            base_name="latest",
        )
    except Exception as exc:
        print(f"Notice: infected region report save skipped: {exc}")

    # ---- Lesion Morphology Analysis --------------------------------
    morph_regions, morph_summary = analyze_lesion_morphology(
        image=rgb,
        leaf_mask=leaf_mask,
        infected_mask=infected_mask,
        contours=region_contours,
        predicted_disease=predicted_name,
        friendly_disease=friendly_pred,
        confidence=confidence,
    )

    morph_overlay = draw_morphology_overlay(
        image=rgb,
        regions=morph_regions,
        contours=region_contours,
    )

    morph_crops = crop_region_images(
        image=rgb,
        regions=morph_regions,
    )

    try:
        save_morphology_reports(
            regions=morph_regions,
            summary=morph_summary,
            region_crops=morph_crops,
            morphology_overlay=morph_overlay,
        )
    except Exception as exc:
        print(f"Notice: morphology report save skipped: {exc}")

    morph_regions_list = [asdict(r) for r in morph_regions]

    # ---- Four-Panel Detailed Visual Explanation --------------------
    four_panel_img = None
    try:
        four_panel_img = create_four_panel_explanation(
            original_image=rgb,
            gradcam_image=gradcam_overlay,
            fusion_image=disease_overlay,
            annotated_image=annotated_infected,
        )
        detailed_results = {
            "crop": crop_name,
            "disease": predicted_name,
            "friendly_disease": friendly_pred,
            "confidence": confidence,
            "total_regions": len(regions),
            "affected_leaf_percent": round(affected_leaf, 2),
        }
        save_detailed_visual_explanation(
            four_panel_image=four_panel_img,
            annotated_image=annotated_infected,
            gradcam_image=gradcam_overlay,
            fusion_image=disease_overlay,
            segmentation_mask=infected_mask,
            detailed_results=detailed_results,
        )
    except Exception as exc:
        print(f"Notice: four-panel explanation skipped: {exc}")

    # ---- Leaf Vein Analysis ----------------------------------------
    vein_map_rgb    = None
    vein_skeleton   = None
    vein_metrics    = {}
    vein_regions_overlay = None
    vein_region_crops    = []
    try:
        vein_result = extract_leaf_veins(rgb, leaf_mask)
        vein_map_rgb  = vein_result.get("vein_map_rgb")
        vein_skeleton = vein_result.get("vein_skeleton")
        vein_metrics  = vein_result.get("metrics", {})

        vein_regions_overlay = create_vein_overlay_with_regions(
            image=rgb,
            leaf_mask=leaf_mask,
            vein_mask=vein_result.get("vein_mask"),
            regions=regions,
            contours=region_contours,
        )

        vein_region_crops = crop_region_veins(
            image=rgb,
            vein_mask=vein_result.get("vein_mask"),
            regions=regions,
        )

        save_leaf_vein_reports(
            image_name="latest",
            vein_map_rgb=vein_map_rgb,
            vein_skeleton=vein_skeleton,
            vein_regions_rgb=vein_regions_overlay if vein_regions_overlay is not None else vein_map_rgb,
            metrics=vein_metrics,
        )
    except Exception as exc:
        print(f"Notice: leaf vein analysis skipped: {exc}")

    adv_obs = analyze_leaf_observations(
        image=rgb,
        leaf_mask=leaf_mask,
        affected_mask=infected_mask,
        gradcam=grayscale_cam,
        vein_map=vein_skeleton,
        prediction=predicted_name,
        affected_leaf_percent=affected_leaf,
    )

    advanced_interpretation = generate_advanced_interpretation(
        prediction=predicted_name,
        confidence=confidence,
        severity=severity,
        affected_area=affected_leaf,
        advanced_observation=adv_obs,
    )

    farmer_report = generate_knowledge_report({
        "predicted_class": predicted_name,
        "prediction": predicted_name,
        "confidence_percent": confidence,
        "confidence": confidence,
        "affected_leaf_percent": affected_leaf,
        "affected_leaf": affected_leaf,
        "severity": severity,
    })

    return {
        "prediction":               predicted_name,
        "friendly_prediction":      friendly_pred,
        "crop":                     crop_name,
        "confidence":               confidence,
        "top3":                     top3,
        "leaf_area":                leaf_area,
        "attention_area":           attention_area,
        "affected_leaf":            affected_leaf,
        "severity":                 severity,
        "gradcam":                  gradcam_overlay,
        "disease_region":           disease_overlay,
        "debug_info":               debug_info,
        "infected_mask":            infected_mask,
        "annotated_infected_image": annotated_infected,
        "infected_regions":         regions_list,
        "total_infected_regions":   len(regions),
        "morphology_regions":       morph_regions_list,
        "morphology_summary":       morph_summary,
        "morphology_overlay":       morph_overlay,
        "morphology_crops":         morph_crops,
        "four_panel_explanation":   four_panel_img,
        "vein_map_rgb":             vein_map_rgb,
        "vein_skeleton":            vein_skeleton,
        "vein_metrics":             vein_metrics,
        "vein_regions_overlay":     vein_regions_overlay,
        "vein_region_crops":        vein_region_crops,
        "advanced_observation":     adv_obs,
        "advanced_interpretation":  advanced_interpretation,
        "farmer_report":            farmer_report,
    }
