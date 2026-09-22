from pathlib import Path
import json
from datetime import datetime


# ============================================================
# AgriVision AI - Structured Disease Report Generator
# ============================================================

SEVERITY_FILE = Path(
    "data/reports/severity/severity_results.json"
)

OUTPUT_DIR = Path(
    "data/reports/farmer_reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD RESULTS
# ============================================================

with open(
    SEVERITY_FILE,
    "r",
    encoding="utf-8"
) as f:

    results = json.load(f)


print("=" * 70)
print("AgriVision AI - Farmer Report Generator")
print("=" * 70)

print(
    f"Available model results: {len(results)}"
)


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Pepper__bell___Bacterial_spot": {
        "crop": "Pepper",
        "disease": "Bacterial Spot"
    },

    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight"
    },

    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight"
    },

    "Tomato_Bacterial_spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot"
    },

    "Tomato_Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight"
    },

    "Tomato_Late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight"
    },

    "Tomato_Leaf_Mold": {
        "crop": "Tomato",
        "disease": "Leaf Mold"
    },

    "Tomato_Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot"
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "crop": "Tomato",
        "disease": "Two-Spotted Spider Mite"
    },

    "Tomato__Target_Spot": {
        "crop": "Tomato",
        "disease": "Target Spot"
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "crop": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus"
    },

    "Tomato__Tomato_mosaic_virus": {
        "crop": "Tomato",
        "disease": "Tomato Mosaic Virus"
    },

    "Pepper__bell___healthy": {
        "crop": "Pepper",
        "disease": "Healthy"
    },

    "Potato___healthy": {
        "crop": "Potato",
        "disease": "Healthy"
    },

    "Tomato_healthy": {
        "crop": "Tomato",
        "disease": "Healthy"
    }
}


# ============================================================
# GENERATE REPORTS
# ============================================================

report_count = 0

for index, result in enumerate(
    results,
    1
):

    predicted_class = result[
        "predicted_class"
    ]

    info = DISEASE_INFO.get(
        predicted_class,
        {
            "crop": predicted_class.split("_")[0],
            "disease": predicted_class
        }
    )

    report = {

        "report_id":
            f"AGRIVISION-{index:05d}",

        "generated_at":
            datetime.now().isoformat(),

        "image":
            result["image"],

        "crop":
            info["crop"],

        "detected_condition":
            info["disease"],

        "model_class":
            predicted_class,

        "prediction": {

            "confidence_percent":
                result[
                    "confidence_percent"
                ],

            "correct_against_dataset_label":
                result[
                    "correct_prediction"
                ]
        },

        "leaf_analysis": {

            "leaf_area_percent":
                result[
                    "leaf_area_percent"
                ],

            "attention_area_percent":
                result[
                    "attention_area_percent"
                ],

            "affected_leaf_percent":
                result[
                    "affected_leaf_percent"
                ]
        },

        "severity": {

            "level":
                result["severity"],

            "method":
                "AgriVision AI model-derived "
                "Grad-CAM and leaf-region analysis"
        },

        "explainability": {

            "method":
                "Grad-CAM",

            "description":
                "The highlighted region represents "
                "the visual region contributing to "
                "the model prediction."
        },

        "status":
            "Model analysis completed"
    }

    output_file = (
        OUTPUT_DIR /
        f"report_{index:05d}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    report_count += 1


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("REPORT GENERATION COMPLETE")
print("=" * 70)

print(
    f"Reports generated: {report_count}"
)

print(
    f"Output: {OUTPUT_DIR}"
)
