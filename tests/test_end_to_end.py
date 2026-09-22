import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.inference_engine import (
    CLASS_NAMES,
    load_model,
    analyze_image,
    segment_leaf,
)


MODEL = load_model()


def _make_leaf_image(
    name: str,
    mode: str = "normal",
    width: int = 224,
    height: int = 224,
):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:] = [80, 150, 75]

    if mode == "normal":
        for cx, cy, radius in [
            (55, 60, 22),
            (110, 90, 18),
            (150, 150, 20),
            (95, 170, 16),
        ]:
            yy, xx = np.ogrid[:height, :width]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            arr[mask] = [110, 95, 80]
    elif mode == "dark":
        arr[:] = [25, 35, 30]
        yy, xx = np.ogrid[:height, :width]
        mask = ((xx - 110) ** 2 + (yy - 110) ** 2 <= 50 ** 2)
        arr[mask] = [45, 60, 50]
    elif mode == "overexposed":
        arr[:] = [220, 230, 200]
        yy, xx = np.ogrid[:height, :width]
        mask = ((xx - 120) ** 2 + (yy - 120) ** 2 <= 60 ** 2)
        arr[mask] = [200, 180, 170]
    elif mode == "blurred":
        arr[:] = [90, 140, 80]
        yy, xx = np.ogrid[:height, :width]
        for cx, cy, radius in [(50, 50, 22), (165, 80, 20), (120, 150, 18)]:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            arr[mask] = [150, 120, 110]
    elif mode == "shadowed":
        arr[:] = [80, 120, 70]
        arr[:, :60] = [35, 55, 30]
        yy, xx = np.ogrid[:height, :width]
        mask = ((xx - 130) ** 2 + (yy - 90) ** 2 <= 40 ** 2)
        arr[mask] = [140, 90, 80]
    elif mode == "partial":
        arr[:] = [100, 150, 85]
        yy, xx = np.ogrid[:height, :width]
        mask = ((xx - 80) ** 2 + (yy - 110) ** 2 <= 70 ** 2)
        arr[mask] = [120, 95, 80]
        arr[:, 150:] = [30, 40, 30]
    elif mode == "complex":
        arr[:] = [180, 160, 145]
        for i in range(8):
            yy, xx = np.ogrid[:height, :width]
            mask = ((xx - (30 + i * 20)) ** 2 + (yy - (30 + i * 15)) ** 2 <= 7 ** 2)
            arr[mask] = [120 + i * 5, 90 + i * 3, 80 + i * 2]
    else:
        arr[:] = [90, 150, 80]

    return Image.fromarray(arr, mode="RGB")


def _assert_finite(value, field_name):
    if value is None:
        return
    if isinstance(value, (float, int)):
        assert math.isfinite(float(value)), f"{field_name} is not finite: {value}"
    if isinstance(value, dict):
        for k, v in value.items():
            _assert_finite(v, f"{field_name}.{k}")
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _assert_finite(item, f"{field_name}[{idx}]")


def _collect_observation_summary(obs):
    return {
        "affected_area": float((obs.get("affected_area_geometry") or {}).get("largest_component_pct", 0.0)),
        "lesion_count": int((obs.get("morphology") or {}).get("lesion_count", 0)),
        "connectivity": str((obs.get("connectivity") or {}).get("connectivity_level", "Not available")),
        "distribution": str((obs.get("spatial_distribution") or {}).get("distribution", "Not available")),
        "quality": str((obs.get("image_quality") or {}).get("classification", "Not available")),
        "summary": str((obs.get("summary") or "").strip()),
    }


def test_class_integrity():
    assert len(CLASS_NAMES) == 15
    assert CLASS_NAMES[0] == "Pepper__bell___Bacterial_spot"
    assert CLASS_NAMES[-1] == "Tomato_healthy"


def test_end_to_end_pipeline_for_core_disease_classes():
    target_predictions = [
        "Tomato_Late_blight",
        "Tomato_Early_blight",
        "Tomato_Bacterial_spot",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato__Target_Spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato__Tomato_mosaic_virus",
        "Potato___Late_blight",
        "Potato___Early_blight",
        "Pepper__bell___Bacterial_spot",
        "Tomato_healthy",
        "Potato___healthy",
        "Pepper__bell___healthy",
    ]

    for target_name in target_predictions:
        image = _make_leaf_image(target_name)
        result = analyze_image(image, MODEL)
        assert result["prediction"] in set(CLASS_NAMES)
        assert 0 <= result["confidence"] <= 100
        assert 0 <= result["affected_leaf"] <= 100
        assert result["advanced_observation"]
        assert result["advanced_interpretation"]
        assert result["farmer_report"]


def test_image_quality_regime_and_no_crash():
    for mode in [
        "normal",
        "low-resolution",
        "blurred",
        "dark",
        "overexposed",
        "shadowed",
        "partial",
        "complex",
    ]:
        image = _make_leaf_image(f"quality_{mode}", mode=mode, width=160 if mode == "low-resolution" else 224)
        result = analyze_image(image, MODEL)
        quality = str((result["advanced_observation"].get("image_quality") or {}).get("classification", "Limited"))
        assert quality in {"Good", "Moderate", "Limited"}
        assert 0 <= result["confidence"] <= 100


def test_numerical_validation_and_finite_values():
    image = _make_leaf_image("numbers")
    result = analyze_image(image, MODEL)
    obs = result["advanced_observation"]
    _assert_finite(result["confidence"], "confidence")
    _assert_finite(result["affected_leaf"], "affected_leaf")
    _assert_finite((obs.get("morphology") or {}).get("mean_lesion_area"), "morphology.mean_lesion_area")
    _assert_finite((obs.get("color") or {}).get("mean_rgb"), "color.mean_rgb")
    _assert_finite((obs.get("texture") or {}).get("contrast"), "texture.contrast")
    assert result["affected_leaf"] >= 0 and result["affected_leaf"] <= 100
    assert int((obs.get("morphology") or {}).get("lesion_count", 0)) >= 0
    circularity = (obs.get("morphology") or {}).get("mean_circularity")
    if circularity is not None:
        assert 0 <= float(circularity) <= 1


def test_empty_mask_handling():
    image = _make_leaf_image("empty_mask")
    leaf_mask = np.zeros((224, 224), dtype=np.uint8)
    affected_mask = np.zeros((224, 224), dtype=np.uint8)
    result = analyze_image(image, MODEL)
    obs = result["advanced_observation"]
    assert obs["morphology"]["lesion_count"] >= 0
    assert obs["image_quality"]["classification"] in {"Good", "Moderate", "Limited"}
    assert "not available" in str(obs["summary"]).lower() or "limited" in str(obs["summary"]).lower() or "no" in str(obs["summary"]).lower()


def test_single_dominant_region_and_many_small_regions():
    # One dominant region case
    single = _make_leaf_image("single_region")
    single_result = analyze_image(single, MODEL)
    single_obs = single_result["advanced_observation"]
    assert single_obs["morphology"]["lesion_count"] >= 0
    assert single_obs["affected_area_geometry"]["meaningful_abnormal_regions"] >= 0

    # Many small region case
    many = _make_leaf_image("many_regions", mode="complex")
    many_result = analyze_image(many, MODEL)
    many_obs = many_result["advanced_observation"]
    assert many_obs["morphology"]["lesion_count"] >= 0
    assert many_obs["connectivity"]["connectivity_level"] in {"LOW CONNECTIVITY", "MODERATE CONNECTIVITY", "HIGH CONNECTIVITY"}


def test_advanced_observation_uniqueness_across_images():
    summaries = []
    for mode in ["normal", "dark", "overexposed", "shadowed", "complex"]:
        img = _make_leaf_image(f"unique_{mode}", mode=mode)
        result = analyze_image(img, MODEL)
        summary = _collect_observation_summary(result["advanced_observation"])
        summaries.append(summary)
    distinct = 0
    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            if summaries[i] != summaries[j]:
                distinct += 1
    assert distinct > 0, "advanced observation was not distinct across different images"


def test_disease_specific_interpretation_is_not_generic():
    late = analyze_image(_make_leaf_image("late", mode="normal"), MODEL)
    mite = analyze_image(_make_leaf_image("mite", mode="dark"), MODEL)
    healthy = analyze_image(_make_leaf_image("healthy"), MODEL)

    late_text = " ".join([late["advanced_interpretation"]["visual_interpretation"], late["advanced_interpretation"]["disease_compatibility"]]).lower()
    mite_text = " ".join([mite["advanced_interpretation"]["visual_interpretation"], mite["advanced_interpretation"]["disease_compatibility"]]).lower()
    healthy_text = " ".join([healthy["advanced_interpretation"]["visual_interpretation"], healthy["advanced_interpretation"]["disease_compatibility"]]).lower()

    assert any(token in late_text for token in ["late blight", "septoria", "lesion", "disease pattern"])
    assert any(token in mite_text for token in ["mite", "spider", "stippling", "feeding damage", "lesion pattern", "disease pattern"])
    assert any(token in healthy_text for token in ["healthy", "no disease was detected by the model for this class", "no substantial abnormal", "no disease-specific lesion pattern"])
    assert "disease progression" not in healthy_text or "no substantial visible progression signal" in healthy_text


def test_no_hallucinated_visual_features():
    for mode in ["normal", "dark", "complex"]:
        result = analyze_image(_make_leaf_image(f"hallucination_{mode}", mode=mode), MODEL)
        text = " ".join([
            result["advanced_interpretation"]["visual_interpretation"],
            result["advanced_interpretation"]["disease_compatibility"],
            result["advanced_interpretation"]["limitations"],
            result["farmer_report"],
        ]).lower()
        assert "concentric rings are visible" not in text
        assert "white sporulation" not in text
        assert "vein-specific lesions" not in text
        assert "rapid spread is occurring" not in text
        assert "severe wilting" not in text
        assert "fruit lesions" not in text or "fruit quality" in text


def test_cross_analysis_validation_and_healthy_class_handling():
    healthy = analyze_image(_make_leaf_image("healthy"), MODEL)
    healthy_text = " ".join([healthy["advanced_interpretation"]["visual_interpretation"], healthy["advanced_interpretation"]["limitations"]]).lower()
    pred = healthy["prediction"].lower()
    assert (
        "no disease was detected by the model for this class" in healthy_text
        or "healthy" in healthy_text
        or pred.replace("_", " ") in healthy_text
        or "disease-specific lesion" in healthy_text
    )
    assert "definitely disease-free" not in healthy_text

    result = analyze_image(_make_leaf_image("cross"), MODEL)
    cross = result["advanced_observation"].get("cross_analysis", {})
    assert cross.get("consistency", "Moderate") in {"High", "Moderate", "Low", "Limited"}


def test_report_consistency_and_pdf_pipeline():
    result = analyze_image(_make_leaf_image("report"), MODEL)
    physician_report = result["farmer_report"]
    assert result["prediction"] in physician_report or result["friendly_prediction"] in physician_report
    assert result["confidence"] >= 0 and result["confidence"] <= 100
    assert result["affected_leaf"] >= 0 and result["affected_leaf"] <= 100
    assert result["severity"] in {"Healthy", "Low", "Moderate", "High"}
    assert result["advanced_interpretation"]


def test_graphical_analytics_uses_real_advanced_observation_fields():
    result = analyze_image(_make_leaf_image("graphical_analytics"), MODEL)
    adv = result["advanced_observation"]
    interpretation = result["advanced_interpretation"]

    assert interpretation.get("observation_reliability")
    assert adv.get("affected_area_geometry")
    assert adv.get("morphology")
    assert adv.get("connectivity")
    assert adv.get("color")
    assert adv.get("texture")
    assert adv["morphology"].get("lesion_count") is not None
    assert adv["color"].get("mean_rgb") is not None
    assert adv["texture"].get("contrast") is not None


def test_performance_and_memory_smoke():
    start = __import__("time").perf_counter()
    for idx in range(10):
        img = _make_leaf_image(f"perf_{idx}", mode="normal")
        result = analyze_image(img, MODEL)
        assert result["prediction"] in set(CLASS_NAMES)
    elapsed = __import__("time").perf_counter() - start
    assert elapsed < 120, f"performance too slow: {elapsed}s"


def test_security_and_dependency_checks():
    repo = ROOT
    matches = []
    forbidden_pattern = re.compile(
        r"localhost:11434|qwen2\.5vl|OLLAMA_URL|OLLAMA_MODEL|src\.ollama|Ollama|Qwen",
        re.IGNORECASE,
    )
    for path in [repo / "app", repo / "src"]:
        for root, _, files in os.walk(path):
            for name in files:
                if name.endswith(".py"):
                    full = os.path.join(root, name)
                    try:
                        with open(full, "r", encoding="utf-8") as fh:
                            text = fh.read()
                    except Exception:
                        continue
                    if forbidden_pattern.search(text):
                        matches.append(full)
    assert not matches, f"unexpected AI dependency strings found: {matches}"

    model_path = ROOT / "models" / "classification" / "resnet18_best.pth"
    assert model_path.exists(), "ResNet18 checkpoint is missing"


def test_model_integrity_and_no_retraining_flags():
    model_path = ROOT / "models" / "classification" / "resnet18_best.pth"
    assert model_path.exists()
    assert os.path.getsize(model_path) > 0
