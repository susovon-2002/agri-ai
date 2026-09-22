from pathlib import Path
import json
import base64
import requests


# ============================================================
# AgriVision AI - Qwen2.5-VL Farmer Report Generator
# ============================================================

MODEL = "qwen2.5vl:3b"

OLLAMA_URL = "http://localhost:11434/api/chat"

SEVERITY_FILE = Path(
    "data/reports/severity/severity_results.json"
)

OUTPUT_DIR = Path(
    "data/reports/ollama"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD MODEL RESULTS
# ============================================================

with open(
    SEVERITY_FILE,
    "r",
    encoding="utf-8"
) as f:

    results = json.load(f)


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Pepper__bell___Bacterial_spot":
        "Pepper bacterial spot",

    "Pepper__bell___healthy":
        "Healthy pepper leaf",

    "Potato___Early_blight":
        "Potato early blight",

    "Potato___Late_blight":
        "Potato late blight",

    "Potato___healthy":
        "Healthy potato leaf",

    "Tomato_Bacterial_spot":
        "Tomato bacterial spot",

    "Tomato_Early_blight":
        "Tomato early blight",

    "Tomato_Late_blight":
        "Tomato late blight",

    "Tomato_Leaf_Mold":
        "Tomato leaf mold",

    "Tomato_Septoria_leaf_spot":
        "Tomato Septoria leaf spot",

    "Tomato_Spider_mites_Two_spotted_spider_mite":
        "Tomato two-spotted spider mites",

    "Tomato__Target_Spot":
        "Tomato target spot",

    "Tomato__Tomato_YellowLeaf__Curl_Virus":
        "Tomato yellow leaf curl virus",

    "Tomato__Tomato_mosaic_virus":
        "Tomato mosaic virus",

    "Tomato_healthy":
        "Healthy tomato leaf"
}


# ============================================================
# IMAGE ENCODER
# ============================================================

def encode_image(image_path):

    with open(
        image_path,
        "rb"
    ) as f:

        return base64.b64encode(
            f.read()
        ).decode("utf-8")


# ============================================================
# QWEN REPORT FUNCTION
# ============================================================

def _fallback_farmer_report(
    result,
    reason="Ollama is not available."
):

    disease_name = result.get(
        "predicted_class",
        "Unknown condition"
    )

    confidence = result.get(
        "confidence_percent",
        0
    )

    affected = result.get(
        "affected_leaf_percent",
        0
    )

    severity = result.get(
        "severity",
        "Unknown"
    )

    disease_name = DISEASE_INFO.get(
        disease_name,
        disease_name
    )

    return f"""
## 🌿 AgriVision AI — Farmer Report

### 1. Crop / Plant

**{disease_name}**

### 2. Detected Condition

**{disease_name}**

The condition above was produced by the AgriVision AI computer-vision model.

### 3. Model Confidence

**{confidence:.2f}%**

### 4. Severity

**{severity}**

### 5. What the Result Means

The computer-vision model identified visual patterns in the uploaded leaf image that are associated with the detected condition.

The estimated affected-leaf percentage is a model-derived visual estimate.

**Estimated affected leaf area: {affected:.2f}%**

### 6. Visible / Relevant Symptoms

The uploaded image contains visual features associated with the predicted condition.

Image-based AI analysis cannot by itself provide laboratory confirmation.

### 7. General Management Guidance

- Inspect the affected plant and nearby plants regularly.
- Maintain good crop and field hygiene.
- Remove or manage visibly affected plant material according to local agricultural guidance.
- Monitor the plant for changes over time.
- Consult a local agricultural expert or plant pathologist for treatment decisions.

### 8. Prevention

- Regularly inspect leaves for new symptoms.
- Maintain suitable crop hygiene.
- Avoid unnecessary movement of potentially affected plant material.
- Follow locally recommended disease-prevention practices.

### 9. Important Note

This is an AI-based image analysis result.

It is **not a laboratory diagnosis**.

The affected-leaf percentage is a **model-derived visual estimate**.

Treatment decisions should be made with appropriate local agricultural guidance.

---

*Qwen2.5-VL was unavailable in the current environment, so AgriVision AI generated this built-in report from the computer-vision result.*
"""


def generate_report(
    result
):

    try:

        image_path = Path(
            result["image"]
        )

        image_base64 = encode_image(
            image_path
        )

        predicted_class = result[
            "predicted_class"
        ]

        disease_name = DISEASE_INFO.get(
            predicted_class,
            predicted_class
        )

        confidence = result[
            "confidence_percent"
        ]

        affected = result[
            "affected_leaf_percent"
        ]

        severity = result[
            "severity"
        ]

        structured_data = {

            "crop_condition":
                disease_name,

            "model_prediction":
                predicted_class,

            "confidence_percent":
                confidence,

            "affected_leaf_region_percent":
                affected,

            "severity":
                severity
        }

        prompt = f"""
You are the AgriVision AI agricultural explanation assistant.

The computer-vision model has already made the disease prediction.
DO NOT change, override, or invent another disease prediction.

Use the following model result as the primary analysis:

{json.dumps(structured_data, indent=2)}

Generate a clear farmer-friendly report.

Include exactly these sections:

1. Crop / Plant
2. Detected Condition
3. Model Confidence
4. Severity
5. What the Result Means
6. Visible / Relevant Symptoms
7. General Management Guidance
8. Prevention
9. Important Note

Rules:

- Respect the model prediction exactly.
- Do not invent another disease.
- Explain the result in simple language.
- Do not claim laboratory confirmation.
- Do not recommend a specific pesticide, chemical dose, or unsafe chemical treatment.
- For treatment decisions, recommend consulting a local agricultural expert.
- The affected-leaf percentage is a model-derived visual estimate.
- Keep the report practical and concise.
"""

        payload = {

            "model":
                MODEL,

            "messages": [

                {
                    "role": "user",

                    "content":
                        prompt,

                    "images": [
                        image_base64
                    ]
                }

            ],

            "stream":
                False,

            "options": {

                "temperature":
                    0.2
            }
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        return data[
            "message"
        ][
            "content"
        ]

    except requests.exceptions.RequestException:

        return _fallback_farmer_report(
            result,
            "Ollama is not reachable."
        )

    except (
        KeyError,
        TypeError,
        ValueError,
        OSError
    ):

        return _fallback_farmer_report(
            result,
            "Qwen2.5-VL could not be used."
        )


# ============================================================# TEST ONE REPORT
# ============================================================

print("=" * 70)
print("AgriVision AI - Qwen2.5-VL Report Generator")
print("=" * 70)

print(
    f"Model: {MODEL}"
)

print(
    f"Available reports: {len(results)}"
)


# Use the first test result
result = results[0]

print()
print(
    f"Image: {result['image']}"
)

print(
    f"Prediction: {result['predicted_class']}"
)

print(
    f"Confidence: "
    f"{result['confidence_percent']:.2f}%"
)

print(
    f"Severity: {result['severity']}"
)

print()
print(
    "Sending image + structured result to Qwen2.5-VL..."
)


# ============================================================
# GENERATE
# ============================================================

try:

    report_text = generate_report(
        result
    )

except Exception as e:

    print()
    print("ERROR:")
    print(e)
    raise


# ============================================================
# SAVE
# ============================================================

output_file = (
    OUTPUT_DIR /
    "sample_farmer_report.txt"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        report_text
    )


# Also save structured result
json_file = (
    OUTPUT_DIR /
    "sample_farmer_report.json"
)

output_data = {

    "model": MODEL,

    "computer_vision_result":
        result,

    "farmer_report":
        report_text
}

with open(
    json_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output_data,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 70)
print("QWEN2.5-VL FARMER REPORT")
print("=" * 70)

print()
print(report_text)

print()
print("=" * 70)

print(
    f"Saved: {output_file}"
)

print(
    f"Saved: {json_file}"
)

print("=" * 70)


