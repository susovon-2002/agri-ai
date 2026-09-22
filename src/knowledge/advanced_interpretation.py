from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parents[2]
KNOWLEDGE_FILE = BASE_DIR / "data" / "knowledge" / "disease_information.json"


def _load_disease_knowledge() -> Dict[str, Any]:
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            return default
        return numeric
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _as_title(raw: str) -> str:
    if not raw:
        return "Unknown condition"
    text = str(raw).replace("_", " ").replace("__", " ").replace("___", " ")
    text = " ".join(text.split())
    return text.strip()


def _pick_metric(source: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in source and source.get(key) is not None:
            return source.get(key)
    return None


def _safe_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _observation_reliability(advanced_observation: Dict[str, Any], confidence: float) -> str:
    quality = (advanced_observation or {}).get("image_quality", {})
    cross = (advanced_observation or {}).get("cross_analysis", {})
    quality_value = str(quality.get("classification", "Moderate")).lower()
    consistency = str(cross.get("consistency", "Moderate")).lower()

    score = 0
    if "good" in quality_value:
        score += 2
    elif "moderate" in quality_value:
        score += 1
    else:
        score += 0

    if "high" in consistency:
        score += 2
    elif "moderate" in consistency:
        score += 1
    else:
        score += 0

    if confidence >= 70:
        score += 1
    elif confidence >= 45:
        score += 0
    else:
        score -= 1

    if score >= 4:
        return "HIGH"
    if score >= 2:
        return "MODERATE"
    return "LIMITED"


def _key_observations(advanced_observation: Dict[str, Any]) -> List[str]:
    if not advanced_observation:
        return ["No advanced visual measurements were available for this image."]

    metrics: List[str] = []
    affected_geometry = advanced_observation.get("affected_area_geometry", {}) or {}
    morphology = advanced_observation.get("morphology", {}) or {}
    spatial = advanced_observation.get("spatial_distribution", {}) or {}
    connectivity = advanced_observation.get("connectivity", {}) or {}
    color = advanced_observation.get("color", {}) or {}
    texture = advanced_observation.get("texture", {}) or {}
    image_quality = advanced_observation.get("image_quality", {}) or {}

    largest_pct = _safe_float(affected_geometry.get("largest_component_pct"), 0.0)
    if largest_pct > 0:
        metrics.append(f"Large visible affected area: {largest_pct:.2f}% of the leaf area")

    lesion_count = _safe_int(morphology.get("lesion_count"), 0)
    if lesion_count:
        metrics.append(f"Meaningful lesion count: {lesion_count} visible abnormal components")

    dist = spatial.get("distribution")
    if dist and str(dist).lower() != "not available":
        metrics.append(f"Spatial distribution: {dist}")

    conn = connectivity.get("connectivity_level")
    if conn and str(conn).lower() != "not available":
        metrics.append(f"Connectivity: {conn}")

    color_contrast = color.get("contrast_to_healthy")
    if color_contrast and str(color_contrast).lower() != "not available":
        metrics.append(f"Color contrast: {color_contrast}")

    texture_var = texture.get("texture_variation")
    if texture_var and str(texture_var).lower() != "not available":
        metrics.append(f"Texture variation: {texture_var}")

    image_quality_class = image_quality.get("classification")
    if image_quality_class:
        metrics.append(f"Image suitability for visual analysis: {image_quality_class}")

    return metrics if metrics else ["The advanced observation engine detected a limited but non-zero image signal."]


def _healthy_response(prediction: str, advanced_observation: Dict[str, Any]) -> Dict[str, Any]:
    crop_name = prediction.replace("__", " ").replace("___", " ").replace("_", " ")
    visual = (
        "The visible leaf shows relatively uniform coloration and no substantial abnormal-region segmentation was identified by the current image-analysis pipeline. "
        "The measured abnormal tissue signal is minimal or absent in the analyzed frame. "
        "This is consistent with a healthy leaf appearance and no disease was detected by the model for this class."
    )
    compatibility = "This result is consistent with the healthy-class prediction in the current model and does not indicate a disease-specific lesion pattern in the uploaded image."
    environmental = "No disease-relevant environmental pattern is indicated by the current image alone. This result should still be interpreted alongside crop scouting and field context."
    progression = "No substantial visible progression signal was detected in the analyzed image. The current observation is consistent with a healthy-class phenotype rather than a progressive lesion pattern."
    limitations = "The healthy-class result reflects the uploaded image and the trained model classes. It does not guarantee that every plant in the field is free of stress, pests or other disorders."
    farmer = "The uploaded leaf appears largely healthy in the analyzed image. The current observation does not show a strong diseased-region pattern, but routine scouting is still important for any new lesions or stress symptoms."
    return {
        "visual_interpretation": visual,
        "disease_compatibility": compatibility,
        "key_observations": _key_observations(advanced_observation),
        "environmental_context": environmental,
        "progression_context": progression,
        "limitations": limitations,
        "observation_reliability": _observation_reliability(advanced_observation, 80.0),
        "crop_name": crop_name,
        "farmer_summary": farmer,
    }


def _build_late_blight_text(prediction: str, adv_obs: Dict[str, Any], disease_knowledge: Dict[str, Any]) -> Dict[str, Any]:
    affected = _safe_float((adv_obs.get("affected_area_geometry") or {}).get("largest_component_pct"), 0.0)
    distribution = (adv_obs.get("spatial_distribution") or {}).get("distribution", "Not available")
    connectivity = (adv_obs.get("connectivity") or {}).get("connectivity_level", "Not available")
    pattern = (adv_obs.get("damage_pattern") or {}).get("pattern", "Not available")
    color = (adv_obs.get("color") or {}).get("contrast_to_healthy", "Not available")
    texture = (adv_obs.get("texture") or {}).get("texture_variation", "Not available")
    quality = (adv_obs.get("image_quality") or {}).get("classification", "Moderate")

    visual = (
        f"The uploaded tomato leaf contains a {distribution.lower()} abnormality pattern with {connectivity.lower()} connectivity and a visible affected-area estimate of {affected:.2f}%. "
        f"The abnormal tissue is organized into a {pattern.lower()} pattern and shows {color.lower()} color contrast with {texture.lower()} texture variation. "
        f"This image-quality assessment is {quality.lower()} for visual analysis and is consistent with tomato late blight symptoms when the lesions coalesce and expand."
    )

    compatibility = (
        "These measured image characteristics are visually compatible with the lesion pattern described for tomato late blight. "
        "The combination of large connected abnormal tissue, irregularity and substantial tissue contrast is consistent with the predicted class and with the disease knowledge for this pathogen."
    )
    environmental = (
        "The observed lesion pattern may be particularly relevant where the crop has experienced cool, humid and prolonged wet conditions. "
        "If such conditions are present, the visual pattern becomes more relevant to the disease knowledge for late blight."
    )
    progression = (
        "The visible extent of abnormal tissue suggests substantial leaf involvement in the analyzed sample. "
        "For diseases known to progress rapidly under favorable weather, this pattern warrants closer monitoring and attention to nearby foliage."
    )
    limitations = (
        "These observations are derived from the uploaded image and provide supporting visual evidence only. "
        "They do not independently confirm Phytophthora infestans or replace field and laboratory diagnosis."
    )
    farmer = (
        "The uploaded leaf shows a large amount of visibly abnormal tissue. The affected regions are partly connected and clustered in a way that is compatible with the predicted late-blight pattern. "
        "If the crop is currently experiencing cool, humid or wet conditions, inspect nearby plants carefully for similar symptoms."
    )
    return {
        "visual_interpretation": visual,
        "disease_compatibility": compatibility,
        "key_observations": _key_observations(adv_obs),
        "environmental_context": environmental,
        "progression_context": progression,
        "limitations": limitations,
        "observation_reliability": _observation_reliability(adv_obs, _safe_float((adv_obs.get("cross_analysis") or {}).get("overlap_percent", 0.0))),
        "crop_name": (disease_knowledge or {}).get("crop", "Tomato"),
        "farmer_summary": farmer,
    }


def _build_early_blight_text(prediction: str, adv_obs: Dict[str, Any], disease_knowledge: Dict[str, Any]) -> Dict[str, Any]:
    affected = _safe_float((adv_obs.get("affected_area_geometry") or {}).get("largest_component_pct"), 0.0)
    morphology = (adv_obs.get("morphology") or {})
    lesion_count = _safe_int(morphology.get("lesion_count"), 0)
    circularity = morphology.get("mean_circularity")
    distribution = (adv_obs.get("spatial_distribution") or {}).get("distribution", "Not available")
    pattern = (adv_obs.get("damage_pattern") or {}).get("pattern", "Not available")
    color = (adv_obs.get("color") or {}).get("contrast_to_healthy", "Not available")
    quality = (adv_obs.get("image_quality") or {}).get("classification", "Moderate")

    ring_note = "Target-like concentric morphology could not be reliably established from this image." if circularity is None or _safe_float(circularity, 0.0) < 0.25 else "The measured lesion geometry is consistent with a relatively circular or target-like lesion structure."

    visual = (
        f"The visible abnormal regions occupy approximately {affected:.2f}% of the leaf area and appear as {lesion_count} measurable lesion components. "
        f"The spatial pattern is {distribution.lower()} and the damage pattern is {pattern.lower()}. {ring_note} "
        f"The affected tissue shows {color.lower()} color contrast and the image quality is {quality.lower()} for visual interpretation."
    )
    compatibility = (
        "Observed lesion morphology is compatible with the lesion-development pattern described for tomato early blight. "
        "When the measured image pattern and the disease knowledge align, the observations support the predicted class without independently proving the causal organism."
    )
    environmental = (
        "This lesion pattern is most relevant when the crop has experienced warm, humid or wet conditions and the canopy remains dense. "
        "If those conditions are present, the observed morphology becomes more consistent with early blight knowledge."
    )
    progression = (
        "The visible lesion pattern indicates active tissue damage on the analyzed leaf. In a field context, lesion expansion under warm wet weather may contribute to a progressively larger affected canopy."
    )
    limitations = (
        "Image morphology alone cannot establish the causal pathogen. The interpretation is therefore supportive visual evidence rather than laboratory confirmation."
    )
    farmer = (
        "The leaf shows a lesion pattern that is compatible with the predicted early-blight type. The visible abnormalities are concentrated in a way that can match common early-blight symptoms, especially in warm, humid field conditions."
    )
    return {
        "visual_interpretation": visual,
        "disease_compatibility": compatibility,
        "key_observations": _key_observations(adv_obs),
        "environmental_context": environmental,
        "progression_context": progression,
        "limitations": limitations,
        "observation_reliability": _observation_reliability(adv_obs, 70.0),
        "crop_name": (disease_knowledge or {}).get("crop", "Tomato"),
        "farmer_summary": farmer,
    }


def _build_bacterial_spot_text(prediction: str, adv_obs: Dict[str, Any], disease_knowledge: Dict[str, Any]) -> Dict[str, Any]:
    affected = _safe_float((adv_obs.get("affected_area_geometry") or {}).get("largest_component_pct"), 0.0)
    distribution = (adv_obs.get("spatial_distribution") or {}).get("distribution", "Not available")
    connectivity = (adv_obs.get("connectivity") or {}).get("connectivity_level", "Not available")
    lesion_count = _safe_int((adv_obs.get("morphology") or {}).get("lesion_count"), 0)
    color = (adv_obs.get("color") or {}).get("contrast_to_healthy", "Not available")
    quality = (adv_obs.get("image_quality") or {}).get("classification", "Moderate")

    visual = (
        f"The abnormal tissue occupies {affected:.2f}% of the visible leaf area with {distribution.lower()} spatial organization and {connectivity.lower()} connectivity. "
        f"Several small to medium lesion components were measured ({lesion_count} visible abnormal components), and the affected tissue shows {color.lower()} color contrast. "
        f"The image quality is {quality.lower()} for detailed lesion interpretation."
    )
    compatibility = (
        "This pattern is compatible with bacterial-spot style lesions because the measured abnormalities are small-to-medium, irregular and distributed in a leaf-affected pattern consistent with warm, wet disease pressure. "
        "However, this remains supporting visual evidence and does not confirm the pathogen."
    )
    environmental = (
        "The visual pattern is most relevant when the crop has been exposed to warm humidity, prolonged leaf wetness and splashing water. "
        "If such conditions are present, the lesion pattern becomes more consistent with the disease knowledge for bacterial spot."
    )
    progression = (
        "The lesion pattern indicates active damage in the observed leaf and may reflect ongoing disease pressure if the canopy remains wet and the crop is under bacterial-spot-favorable conditions."
    )
    limitations = (
        "Small irregular lesions alone do not prove bacterial infection. A single image can support the predicted class but cannot independently confirm the causal organism."
    )
    farmer = (
        "The leaf shows small to moderate visible spots and a damage pattern that is compatible with the predicted bacterial-spot class. This is not a lab confirmation, but it is consistent with the kind of symptoms that often develop under warm, wet crop conditions."
    )
    return {
        "visual_interpretation": visual,
        "disease_compatibility": compatibility,
        "key_observations": _key_observations(adv_obs),
        "environmental_context": environmental,
        "progression_context": progression,
        "limitations": limitations,
        "observation_reliability": _observation_reliability(adv_obs, 65.0),
        "crop_name": (disease_knowledge or {}).get("crop", "Tomato or pepper"),
        "farmer_summary": farmer,
    }


def _build_leaf_mold_text(prediction: str, adv_obs: Dict[str, Any], disease_knowledge: Dict[str, Any]) -> Dict[str, Any]:
    affected = _safe_float((adv_obs.get("affected_area_geometry") or {}).get("largest_component_pct"), 0.0)
    spatial = (adv_obs.get("spatial_distribution") or {}).get("distribution", "Not available")
    color = (adv_obs.get("color") or {}).get("contrast_to_healthy", "Not available")
    texture = (adv_obs.get("texture") or {}).get("texture_variation", "Not available")
    quality = (adv_obs.get("image_quality") or {}).get("classification", "Moderate")
    visual = (
        f"The analyzed leaf presents a {spatial.lower()} abnormality pattern with approximately {affected:.2f}% affected visible area. "
        f"The lesion tissue shows {color.lower()} color contrast and {texture.lower()} texture variation, which is consistent with a lesion pattern viewed under humid canopy conditions. "
        f"Image suitability is {quality.lower()} for this type of interpretation."
    )
    compatibility = (
        "The measured pattern is compatible with leaf-mold style disease expression in a humid canopy, especially when the disease knowledge emphasizes upper/lower leaf surface differences and high humidity. "
        "This remains visual support rather than a confirmed laboratory diagnosis."
    )
    environmental = (
        "The interpretation becomes more relevant when humidity remains high and the canopy has poor ventilation or prolonged leaf wetness. "
        "If those conditions are present, the observed pattern is more aligned with the disease knowledge for leaf mold."
    )
    progression = (
        "The visible abnormal tissue suggests ongoing leaf involvement and supports the interpretation that the canopy may be under persistent humid conditions."
    )
    limitations = "Upper- and lower-surface fungal growth cannot be confirmed from the current image unless the visual analysis explicitly measures it; the interpretation remains image-based and supportive."
    farmer = "The leaf shows a pattern that can be compatible with a humid-canopy fungal problem. In crop conditions with poor ventilation and repeated humidity, this kind of symptom pattern is worth checking on nearby plants."
    return {
        "visual_interpretation": visual,
        "disease_compatibility": compatibility,
        "key_observations": _key_observations(adv_obs),
        "environmental_context": environmental,
        "progression_context": progression,
        "limitations": limitations,
        "observation_reliability": _observation_reliability(adv_obs, 68.0),
        "crop_name": (disease_knowledge or {}).get("crop", "Tomato"),
        "farmer_summary": farmer,
    }


def _build_septoria_text(prediction: str, adv_obs: Dict[str, Any], disease_knowledge: Dict[str, Any]) -> Dict[str, Any]:
    affected = _safe_float((adv_obs.get("affected_area_geometry") or {}).get("largest_component_pct"), 0.0)
    lesion_count = _safe_int((adv_obs.get("morphology") or {}).get("lesion_count"), 0)
    distribution = (adv_obs.get("spatial_distribution") or {}).get("distribution", "Not available")
    connectivity = (adv_obs.get("connectivity") or {}).get("connectivity_level", "Not available")
    color = (adv_obs.get("color") or {}).get("contrast_to_healthy", "Not available")
    quality = (adv_obs.get("image_quality") or {}).get("classification", "Moderate")
    visual = (
        f"The lesion distribution is {distribution.lower()} and there are {lesion_count} measurable abnormal regions with {connectivity.lower()} connectivity. "
        f"The visible affected area is approximately {affected:.2f}% and the abnormal tissue shows {color.lower()} color contrast. "
        f"The image quality is {quality.lower()} for lesion-pattern interpretation and this pattern can overlap with late blight in a dense canopy."
    )
    compatibility = (
        "This pattern is compatible with the lesion pattern described for Septoria leaf spot, particularly when the damage is distributed across multiple lesion components rather than dominated by a few isolated large lesions. "
        "It may also resemble late blight under dense, humid conditions, but the interpretation remains visual support only and does not confirm the pathogen."
    )
    environmental = "The observed lesion pattern is most relevant when the crop has experienced periods of warm humidity, dense foliage and prolonged leaf wetness."
    progression = "The measured lesion distribution indicates visible leaf involvement across the sampled image and can be consistent with a progressive disease cycle under favorable conditions."
    limitations = "Spot size, lesion center and fruiting-body level cannot be confirmed from image geometry alone. The interpretation therefore remains provisional and image-based."
    farmer = "The leaf shows several visible lesions distributed across the surface, which is compatible with the predicted Septoria-type pattern. If the crop has been wet and humid, nearby foliage should be checked for the same pattern."
    return {
        "visual_interpretation": visual,
        "disease_compatibility": compatibility,
        "key_observations": _key_observations(adv_obs),
        "environmental_context": environmental,
        "progression_context": progression,
        "limitations": limitations,
        "observation_reliability": _observation_reliability(adv_obs, 66.0),
        "crop_name": (disease_knowledge or {}).get("crop", "Tomato"),
        "farmer_summary": farmer,
    }


def _build_spider_mite_text(prediction: str, adv_obs: Dict[str, Any], disease_knowledge: Dict[str, Any]) -> Dict[str, Any]:
    affected = _safe_float((adv_obs.get("affected_area_geometry") or {}).get("largest_component_pct"), 0.0)
    distribution = (adv_obs.get("spatial_distribution") or {}).get("distribution", "Not available")
    color = (adv_obs.get("color") or {}).get("contrast_to_healthy", "Not available")
    texture = (adv_obs.get("texture") or {}).get("texture_variation", "Not available")
    quality = (adv_obs.get("image_quality") or {}).get("classification", "Moderate")
    visual = (
        f"The leaf shows a {distribution.lower()} pattern of abnormalities with a visible affected-area estimate of about {affected:.2f}%. "
        f"The tissue color and texture differ from surrounding healthy tissue ({color.lower()} color contrast and {texture.lower()} texture variation), which can be consistent with spider mite feeding damage rather than a necrotic fungal lesion pattern. "
        f"Image analysis quality is {quality.lower()}."
    )
    compatibility = (
        "This pattern is more compatible with stippling or diffuse feeding damage than with a bacterial or fungal lesion pattern. The observation supports the predicted spider mite or mite-damage class but does not prove the pest is the causal agent."
    )
    environmental = "The pattern is particularly relevant in hot, dry and dusty conditions where mite populations can increase rapidly and feeding damage becomes visually prominent."
    progression = "The visible damage pattern suggests the leaf is under ongoing feeding pressure rather than a single lesion event, which is consistent with escalating mite pressure in dry conditions."
    limitations = "Fine stippling and diffuse discoloration can overlap with other abiotic stress symptoms, so the image should be interpreted alongside field observations and pest scouting."
    farmer = "The leaf shows a diffuse, stippling-like pattern that is compatible with mite feeding damage rather than a classic fungal or bacterial lesion pattern. In hot, dry conditions, this should be checked carefully for mites on the undersides of leaves."
    return {
        "visual_interpretation": visual,
        "disease_compatibility": compatibility,
        "key_observations": _key_observations(adv_obs),
        "environmental_context": environmental,
        "progression_context": progression,
        "limitations": limitations,
        "observation_reliability": _observation_reliability(adv_obs, 63.0),
        "crop_name": (disease_knowledge or {}).get("crop", "Tomato"),
        "farmer_summary": farmer,
    }


def _build_target_spot_text(prediction: str, adv_obs: Dict[str, Any], disease_knowledge: Dict[str, Any]) -> Dict[str, Any]:
    affected = _safe_float((adv_obs.get("affected_area_geometry") or {}).get("largest_component_pct"), 0.0)
    pattern = (adv_obs.get("damage_pattern") or {}).get("pattern", "Not available")
    similarity = "Target-like morphology could not be reliably established from this image." if pattern in ("Not available", "MIXED") else f"The lesion architecture is consistent with a {pattern.lower()} damage pattern, which may be relevant to target-spot-like disease expression."
    visual = (
        f"The lesion pattern covers approximately {affected:.2f}% of the visible leaf area and is described as {pattern.lower()}. {similarity} "
        "The observed pattern is evaluated only from the visible image geometry and must not be treated as laboratory confirmation."
    )
    compatibility = "The observed lesion pattern can be compatible with target-spot disease knowledge when the morphology and distribution support a concentric or target-like lesion program. If the lesions are not target-like in the measured image, the interpretation is more cautious."
    environmental = "Warm, humid and densely canopied conditions make this interpretation more relevant when the crop remains wet for long periods."
    progression = "The observed lesion distribution indicates visible tissue involvement and may reflect active disease pressure if the canopy remains favorable for fungal spread."
    limitations = "The image analysis does not guarantee concentric rings are truly present. Target-like morphology must be supported by actual measured lesion structure, not inference alone."
    farmer = "The leaf shows a lesion pattern that may be consistent with a target-spot type, but the final interpretation depends on whether the visible lesion structure is truly target-like in this image."
    return {
        "visual_interpretation": visual,
        "disease_compatibility": compatibility,
        "key_observations": _key_observations(adv_obs),
        "environmental_context": environmental,
        "progression_context": progression,
        "limitations": limitations,
        "observation_reliability": _observation_reliability(adv_obs, 60.0),
        "crop_name": (disease_knowledge or {}).get("crop", "Tomato"),
        "farmer_summary": farmer,
    }


def _build_viral_text(prediction: str, adv_obs: Dict[str, Any], disease_knowledge: Dict[str, Any]) -> Dict[str, Any]:
    affected = _safe_float((adv_obs.get("affected_area_geometry") or {}).get("largest_component_pct"), 0.0)
    distribution = (adv_obs.get("spatial_distribution") or {}).get("distribution", "Not available")
    color = (adv_obs.get("color") or {}).get("contrast_to_healthy", "Not available")
    texture = (adv_obs.get("texture") or {}).get("texture_variation", "Not available")
    quality = (adv_obs.get("image_quality") or {}).get("classification", "Moderate")
    visual = (
        f"The image shows a {distribution.lower()} pattern of visible color and tissue change across the leaf, with an affected-area estimate of approximately {affected:.2f}%. "
        f"The channel differences suggest {color.lower()} contrast and {texture.lower()} texture variation, which can be compatible with viral-type color disruption or mosaic-like appearance, but this is not a viral confirmation. "
        f"Image suitability is {quality.lower()} for this interpretation."
    )
    compatibility = "The measured color and tissue pattern is consistent with the predicted viral class and with the broader knowledge that viral diseases may cause diffuse symptom expression, yellowing or deformity. This is supportive, not definitive."
    environmental = "The interpretation is most relevant when whitefly pressure or mechanical spread risks are present. The visual pattern alone cannot identify the transmission route or confirm the causal virus."
    progression = "The visible distribution suggests broad leaf involvement and may be consistent with a systemic disease pattern, but a single image cannot establish the exact time course of the disease."
    limitations = "Image morphology cannot independently confirm a virus. The stored disease knowledge supports interpretation, but not a causal diagnosis from the photo alone."
    farmer = "The leaf shows broader color disruption and a pattern that is compatible with the predicted virus class. This is supportive visual evidence, but it does not prove the virus is the cause without field and diagnostic confirmation."
    return {
        "visual_interpretation": visual,
        "disease_compatibility": compatibility,
        "key_observations": _key_observations(adv_obs),
        "environmental_context": environmental,
        "progression_context": progression,
        "limitations": limitations,
        "observation_reliability": _observation_reliability(adv_obs, 60.0),
        "crop_name": (disease_knowledge or {}).get("crop", "Tomato"),
        "farmer_summary": farmer,
    }


def _looks_healthy_from_observation(advanced_observation: Dict[str, Any]) -> bool:
    if not advanced_observation:
        return False
    geometry = advanced_observation.get("affected_area_geometry") or {}
    morphology = advanced_observation.get("morphology") or {}
    color = advanced_observation.get("color") or {}
    texture = advanced_observation.get("texture") or {}
    largest_pct = _safe_float(geometry.get("largest_component_pct"), 0.0)
    lesion_count = _safe_int(morphology.get("lesion_count"), 0)
    color_contrast = str(color.get("contrast_to_healthy", "Not available")).lower()
    texture_var = str(texture.get("texture_variation", "Not available")).lower()
    return (
        largest_pct < 8.0
        and lesion_count <= 1
        and (color_contrast in {"low", "not available"} or "not available" in color_contrast)
        and (texture_var in {"low", "not available"} or "not available" in texture_var)
    )


def _looks_mite_like(advanced_observation: Dict[str, Any]) -> bool:
    if not advanced_observation:
        return False
    geometry = advanced_observation.get("affected_area_geometry") or {}
    spatial = advanced_observation.get("spatial_distribution") or {}
    color = advanced_observation.get("color") or {}
    texture = advanced_observation.get("texture") or {}
    damage = advanced_observation.get("damage_pattern") or {}
    largest_pct = _safe_float(geometry.get("largest_component_pct"), 0.0)
    distribution = str(spatial.get("distribution", "")).lower()
    contrast = str(color.get("contrast_to_healthy", "Not available")).lower()
    texture_var = str(texture.get("texture_variation", "Not available")).lower()
    pattern = str(damage.get("pattern", "")).lower()
    return (
        largest_pct > 15.0
        and distribution in {"center-dominant", "mixed", "scattered"}
        and contrast in {"low", "moderate"}
        and texture_var in {"high", "moderate"}
        and pattern in {"patch-dominant", "mixed", "clustered"}
    )


def _looks_late_blight_like(advanced_observation: Dict[str, Any]) -> bool:
    if not advanced_observation:
        return False
    geometry = advanced_observation.get("affected_area_geometry") or {}
    spatial = advanced_observation.get("spatial_distribution") or {}
    connectivity = advanced_observation.get("connectivity") or {}
    damage = advanced_observation.get("damage_pattern") or {}
    color = advanced_observation.get("color") or {}
    largest_pct = _safe_float(geometry.get("largest_component_pct"), 0.0)
    distribution = str(spatial.get("distribution", "")).lower()
    conn = str(connectivity.get("connectivity_level", "")).lower()
    pattern = str(damage.get("pattern", "")).lower()
    contrast = str(color.get("contrast_to_healthy", "Not available")).lower()
    return (
        largest_pct > 25.0
        and distribution in {"center-dominant", "mixed"}
        and "high" in conn
        and pattern in {"patch-dominant", "mixed", "clustered"}
        and contrast in {"low", "moderate"}
    )


def generate_advanced_interpretation(
    prediction: str,
    confidence: float,
    severity: Any,
    affected_area: Any,
    advanced_observation: Dict[str, Any],
    disease_knowledge: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Create an interpretation layer that explains the disease-specific meaning of the
    observed image measurements while keeping the underlying classifier unchanged.
    """
    if disease_knowledge is None:
        disease_knowledge = _load_disease_knowledge()

    prediction_key = str(prediction or "Unknown").strip()
    knowledge = disease_knowledge.get(prediction_key, {})
    observed = advanced_observation or {}
    profiled_confidence = _safe_float(confidence, 0.0)

    if _looks_healthy_from_observation(observed):
        result = _healthy_response(prediction_key, observed)
        result["disease_name"] = knowledge.get("condition") or _as_title(prediction_key)
        result["model_confidence"] = profiled_confidence
        result["severity"] = str(severity or "Healthy")
        result["affected_area_percent"] = _safe_float(affected_area, _safe_float((observed.get("affected_area_geometry") or {}).get("largest_component_pct"), 0.0))
        return result

    if _looks_mite_like(observed):
        result = _build_spider_mite_text(prediction_key, observed, knowledge)
    elif _looks_late_blight_like(observed):
        result = _build_late_blight_text(prediction_key, observed, knowledge)
    elif "healthy" in prediction_key.lower():
        result = _healthy_response(prediction_key, observed)
    elif "Tomato_Late_blight" in prediction_key or "Late_blight" in prediction_key:
        result = _build_late_blight_text(prediction_key, observed, knowledge)
    elif "Tomato_Early_blight" in prediction_key or "Early_blight" in prediction_key:
        result = _build_early_blight_text(prediction_key, observed, knowledge)
    elif "Bacterial_spot" in prediction_key:
        result = _build_bacterial_spot_text(prediction_key, observed, knowledge)
    elif "Tomato_Leaf_Mold" in prediction_key:
        result = _build_leaf_mold_text(prediction_key, observed, knowledge)
    elif "Tomato_Septoria_leaf_spot" in prediction_key:
        result = _build_septoria_text(prediction_key, observed, knowledge)
    elif "Spider_mites" in prediction_key or "spider" in prediction_key.lower():
        result = _build_spider_mite_text(prediction_key, observed, knowledge)
    elif "Target_Spot" in prediction_key:
        result = _build_target_spot_text(prediction_key, observed, knowledge)
    elif "YellowLeaf_Curl_Virus" in prediction_key or "mosaic_virus" in prediction_key.lower() or "virus" in prediction_key.lower():
        result = _build_viral_text(prediction_key, observed, knowledge)
    else:
        result = {
            "visual_interpretation": "The observed visual pattern is compatible with the predicted class based on image geometry and tissue appearance, but the single-image interpretation remains supportive and non-diagnostic.",
            "disease_compatibility": "The interpretation is of the predicted class only and should not be interpreted as laboratory confirmation.",
            "key_observations": _key_observations(observed),
            "environmental_context": "Relevant environmental context for this class is available in the local disease knowledge database, but the current image does not provide field weather information.",
            "progression_context": "The visual pattern indicates visible tissue involvement and may be relevant to disease progression under suitable conditions, but time-dependent spread cannot be inferred from a single image.",
            "limitations": "This interpretation is based on a single uploaded image and does not independently confirm the pathogen. It should be read as supportive visual evidence only.",
            "observation_reliability": _observation_reliability(observed, profiled_confidence),
            "crop_name": knowledge.get("crop", "Unknown crop"),
            "farmer_summary": "The visible leaf pattern is compatible with the predicted disease class. This is supportive image evidence and should be reviewed alongside crop scouting and expert advice.",
        }

    result.setdefault("disease_name", knowledge.get("condition") or _as_title(prediction_key))
    result.setdefault("model_confidence", profiled_confidence)
    result.setdefault("severity", str(severity or "Unknown"))
    result.setdefault("affected_area_percent", _safe_float(affected_area, _safe_float((observed.get("affected_area_geometry") or {}).get("largest_component_pct"), 0.0)))
    result.setdefault("observation_reliability", _observation_reliability(observed, profiled_confidence))
    result.setdefault("key_observations", _key_observations(observed))
    return result
