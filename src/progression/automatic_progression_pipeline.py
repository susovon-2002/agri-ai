"""
AgriVision AI
Automatic Plant Disease Progression Pipeline

Uses the EXISTING Streamlit inference functions from app.py.

Input:
    data/progression_images/
        plant_001/
            2026-09-01.jpg
            2026-09-04.jpg
            2026-09-07.jpg

Output:
    data/reports/progression/automatic_observations.json
    data/reports/progression/automatic_observations.csv
    data/reports/progression/automatic_progression_results.json
    data/reports/progression/automatic_progression_summary.csv

The pipeline:
    Image
      ↓
    Existing AgriVision analyze_image()
      ↓
    Disease
    Confidence
    Affected Leaf
    Severity
      ↓
    Observation JSON
      ↓
    Disease Progression Engine
"""

from __future__ import annotations

import sys
import json
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

APP_PATH = PROJECT_ROOT / "app"

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "progression_images"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "progression"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# IMPORT EXISTING AGRIVISION PIPELINE
# ============================================================

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

try:

    from src.inference.inference_engine import (
        load_model,
        analyze_image,
    )

except Exception as exc:

    print()
    print(
        "ERROR: Could not import the existing "
        "AgriVision inference pipeline from src.inference.inference_engine."
    )

    print()
    print(
        f"Details: {exc}"
    )

    raise


# ============================================================
# DATE EXTRACTION
# ============================================================

def extract_date(
    image_path: Path,
):

    """
    Try to extract observation date from filename.

    Supported examples:

        2026-09-01.jpg
        2026_09_01.jpg
        day_01.jpg

    If no date is found, file modification time is used.
    """

    name = image_path.stem

    formats = [
        "%Y-%m-%d",
        "%Y_%m_%d",
        "%Y%m%d",
    ]

    for fmt in formats:

        try:

            return datetime.strptime(
                name,
                fmt,
            )

        except ValueError:
            pass

    # Search filename for common date patterns
    import re

    match = re.search(
        r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})",
        name,
    )

    if match:

        year = int(
            match.group(1)
        )

        month = int(
            match.group(2)
        )

        day = int(
            match.group(3)
        )

        try:

            return datetime(
                year,
                month,
                day,
            )

        except ValueError:
            pass

    # Fallback: file modification date
    return datetime.fromtimestamp(
        image_path.stat().st_mtime
    )


# ============================================================
# CROP NAME
# ============================================================

def infer_crop(
    prediction: str,
):

    prediction_lower = (
        prediction.lower()
    )

    if "pepper" in prediction_lower:

        return "Pepper"

    if "potato" in prediction_lower:

        return "Potato"

    if "tomato" in prediction_lower:

        return "Tomato"

    return "Unknown"


# ============================================================
# SAFE VALUE
# ============================================================

def safe_float(
    value,
    default=0.0,
):

    try:

        value = float(value)

        if not np.isfinite(
            value
        ):

            return default

        return value

    except (
        TypeError,
        ValueError,
    ):

        return default


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def analyze_single_image(
    image_path,
    model,
    plant_id,
):

    print()
    print(
        f"Analyzing: {image_path.name}"
    )

    observation_date = extract_date(
        image_path
    )

    try:

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

    except Exception as exc:

        print(
            f"Could not open image: {exc}"
        )

        return None

    # --------------------------------------------------------
    # USE EXISTING AGRIVISION PIPELINE
    # --------------------------------------------------------

    result = analyze_image(
        image,
        model,
    )

    prediction = str(
        result.get(
            "prediction",
            "Unknown",
        )
    )

    friendly_prediction = str(
        result.get(
            "friendly_prediction",
            prediction,
        )
    )

    confidence = safe_float(
        result.get(
            "confidence",
            0.0,
        )
    )

    affected_leaf = safe_float(
        result.get(
            "affected_leaf",
            0.0,
        )
    )

    leaf_area = safe_float(
        result.get(
            "leaf_area",
            0.0,
        )
    )

    attention_area = safe_float(
        result.get(
            "attention_area",
            0.0,
        )
    )

    severity = str(
        result.get(
            "severity",
            "Unknown",
        )
    )

    crop = infer_crop(
        prediction
    )

    observation = {

        "plant_id":
            plant_id,

        "observation_date":
            observation_date.strftime(
                "%Y-%m-%d"
            ),

        "image_file":
            str(
                image_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "crop":
            crop,

        "disease":
            prediction,

        "friendly_prediction":
            friendly_prediction,

        "confidence":
            round(
                confidence / 100.0
                if confidence > 1
                else confidence,
                6,
            ),

        "confidence_percent":
            round(
                confidence,
                2,
            ),

        "leaf_area_percent":
            round(
                leaf_area,
                2,
            ),

        "attention_area_percent":
            round(
                attention_area,
                2,
            ),

        "affected_area_percent":
            round(
                affected_leaf,
                2,
            ),

        "severity":
            severity,

        "analysis_source":
            "AgriVision AI existing inference pipeline",
    }

    print(
        f"  Crop: {crop}"
    )

    print(
        f"  Disease: {friendly_prediction}"
    )

    print(
        f"  Confidence: {confidence:.2f}%"
    )

    print(
        f"  Affected leaf: "
        f"{affected_leaf:.2f}%"
    )

    print(
        f"  Severity: {severity}"
    )

    return observation


# ============================================================
# FIND PLANT FOLDERS
# ============================================================

def find_plant_folders():

    if not INPUT_DIR.exists():

        return []

    folders = [
        item
        for item in INPUT_DIR.iterdir()
        if item.is_dir() and item.name != "_generated_observations"
    ]

    return sorted(
        folders,
        key=lambda item: item.name,
    )


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(
    plant_folder,
):

    images = []

    for path in plant_folder.iterdir():

        if not path.is_file():

            continue

        if (
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            images.append(
                path
            )

    return sorted(
        images,
        key=lambda path: (
            extract_date(path),
            path.name,
        ),
    )


# ============================================================
# PROCESS ALL PLANTS
# ============================================================

def generate_observations(
    model,
):

    plant_folders = (
        find_plant_folders()
    )

    print()
    print(
        f"Plants found: "
        f"{len(plant_folders)}"
    )

    all_observations = []

    for plant_folder in plant_folders:

        plant_id = (
            plant_folder.name
        )

        images = find_images(
            plant_folder
        )

        print()
        print(
            "=" * 60
        )

        print(
            f"Plant: {plant_id}"
        )

        print(
            f"Images: {len(images)}"
        )

        print(
            "=" * 60
        )

        if not images:

            print(
                "No images found."
            )

            continue

        for image_path in images:

            observation = (
                analyze_single_image(
                    image_path,
                    model,
                    plant_id,
                )
            )

            if observation:

                all_observations.append(
                    observation
                )

    return all_observations


# ============================================================
# SAVE OBSERVATIONS
# ============================================================

def save_observations(
    observations,
):

    json_path = (
        OUTPUT_DIR
        / "automatic_observations.json"
    )

    csv_path = (
        OUTPUT_DIR
        / "automatic_observations.csv"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            observations,
            file,
            indent=2,
            ensure_ascii=False,
        )

    if observations:

        fieldnames = list(
            observations[0].keys()
        )

        with open(
            csv_path,
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                observations
            )

    print()
    print(
        "Observation files saved:"
    )

    print(
        json_path
    )

    print(
        csv_path
    )

    return json_path


# ============================================================
# CONVERT TO PROGRESSION ENGINE FORMAT
# ============================================================

def create_engine_input(
    observations,
):

    engine_dir = (
        INPUT_DIR
        / "_generated_observations"
    )

    engine_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove previous generated JSON
    for path in engine_dir.glob(
        "*.json"
    ):

        try:

            path.unlink()

        except Exception:

            pass

    grouped = {}

    for observation in observations:

        grouped.setdefault(
            observation[
                "plant_id"
            ],
            [],
        ).append(
            observation
        )

    created_files = []

    for plant_id, items in grouped.items():

        plant_dir = (
            engine_dir
            / plant_id
        )

        plant_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, item in enumerate(
            sorted(
                items,
                key=lambda x:
                x[
                    "observation_date"
                ],
            ),
            1,
        ):

            engine_record = {

                "plant_id":
                    item["plant_id"],

                "observation_date":
                    item[
                        "observation_date"
                    ],

                "disease":
                    item["disease"],

                "crop":
                    item["crop"],

                "confidence":
                    item["confidence"],

                "affected_area_percent":
                    item[
                        "affected_area_percent"
                    ],

                "severity":
                    item["severity"],
            }

            output_file = (
                plant_dir
                / (
                    f"observation_"
                    f"{index:03d}.json"
                )
            )

            with open(
                output_file,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    engine_record,
                    file,
                    indent=2,
                )

            created_files.append(
                output_file
            )

    return created_files


# ============================================================
# RUN EXISTING PROGRESSION ENGINE
# ============================================================

def run_progression_engine():

    """
    Import the existing progression engine.

    This avoids duplicating its calculations.
    """

    try:

        from src.progression.disease_progression import (
            load_json_observations,
            sort_observations,
            analyze_plant,
            save_reports,
        )

    except Exception as exc:

        print()
        print(
            "Could not import disease progression engine."
        )

        print(
            f"Details: {exc}"
        )

        return None

    observations = (
        load_json_observations()
    )

    if not observations:

        print(
            "No observations available "
            "for progression analysis."
        )

        return None

    observations = (
        sort_observations(
            observations
        )
    )

    grouped = {}

    for observation in observations:

        grouped.setdefault(
            observation.plant_id,
            [],
        ).append(
            observation
        )

    results = []

    print()
    print(
        "=" * 70
    )

    print(
        "RUNNING DISEASE PROGRESSION ENGINE"
    )

    print(
        "=" * 70
    )

    for plant_id, items in grouped.items():

        print()
        print(
            f"Progression analysis: "
            f"{plant_id}"
        )

        result = analyze_plant(
            items
        )

        results.append(
            result
        )

        if result.get(
            "status"
        ) == "ANALYZED":

            print(
                f"  State: "
                f"{result['progression_state']}"
            )

            area = result[
                "affected_area_analysis"
            ]

            print(
                f"  Initial affected area: "
                f"{area['initial_affected_area_percent']:.2f}%"
            )

            print(
                f"  Latest affected area: "
                f"{area['latest_affected_area_percent']:.2f}%"
            )

            print(
                f"  Change: "
                f"{area['absolute_change_percentage_points']:+.2f} "
                f"percentage points"
            )

            print(
                f"  Velocity: "
                f"{area['progression_velocity_percent_per_day']:+.3f}%/day"
            )

    save_reports(
        results
    )

    return results


# ============================================================
# CREATE AUTOMATIC SUMMARY
# ============================================================

def save_automatic_summary(
    observations,
    results,
):

    summary_path = (
        OUTPUT_DIR
        / "automatic_progression_summary.json"
    )

    summary = {

        "generated_at":
            datetime.now().isoformat(),

        "pipeline":
            "Automatic AgriVision AI Progression Pipeline",

        "total_observations":
            len(observations),

        "total_plants":
            len(
                set(
                    item["plant_id"]
                    for item in observations
                )
            ),

        "progression_results":
            len(results)
            if results
            else 0,

        "architecture": [
            "ResNet18 classification",
            "Existing Grad-CAM / attention pipeline",
            "Existing leaf-region analysis",
            "Existing affected-leaf estimation",
            "Existing severity estimation",
            "Disease Progression Engine",
        ],

        "important_note":
            (
                "Progression is an image-derived "
                "decision-support signal. It does not "
                "establish biological disease progression "
                "without appropriate field validation."
            ),
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print(
        f"Automatic summary saved:\n"
        f"{summary_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "AgriVision AI"
    )

    print(
        "AUTOMATIC DISEASE PROGRESSION PIPELINE"
    )

    print("=" * 70)

    print()

    print(
        f"Input directory:\n"
        f"{INPUT_DIR}"
    )

    if not INPUT_DIR.exists():

        INPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print(
            "Created progression image directory."
        )

        print()
        print(
            "Add images using:"
        )

        print(
            r"data\progression_images\plant_001\2026-09-01.jpg"
        )

        print(
            r"data\progression_images\plant_001\2026-09-04.jpg"
        )

        print(
            r"data\progression_images\plant_001\2026-09-07.jpg"
        )

        return

    print()
    print(
        "Loading existing ResNet18..."
    )

    model = load_model()

    print(
        "Existing AgriVision model loaded."
    )

    observations = (
        generate_observations(
            model
        )
    )

    if not observations:

        print()
        print(
            "NO IMAGES WERE PROCESSED."
        )

        print()
        print(
            "Put images inside:"
        )

        print(
    "data/progression_images/<plant_id>/"
    "    <image>.jpg"
        )

        return

    save_observations(
        observations
    )

    generated = (
        create_engine_input(
            observations
        )
    )

    print()
    print(
        f"Generated progression observations: "
        f"{len(generated)}"
    )

    results = (
        run_progression_engine()
    )

    save_automatic_summary(
        observations,
        results,
    )

    print()
    print("=" * 70)

    print(
        "AUTOMATIC PROGRESSION PIPELINE COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()

