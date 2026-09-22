from pathlib import Path
import json
import csv

import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models


# ============================================================
# AgriVision AI - Real Field Validation
# ============================================================

FIELD_DIR = Path("data/field_test")

CHECKPOINT = Path(
    "models/classification/resnet18_best.pth"
)

OUTPUT_DIR = Path(
    "data/reports/field_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASS NAMES
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
# HEADER
# ============================================================

print("=" * 70)
print("AgriVision AI - Real Field Validation")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Field directory: {FIELD_DIR}")
print(f"Model: {CHECKPOINT}")


# ============================================================
# CHECK MODEL
# ============================================================

if not CHECKPOINT.exists():

    print()
    print("ERROR: ResNet18 checkpoint not found.")
    print(CHECKPOINT)
    raise SystemExit(1)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD RESNET18
# ============================================================

model = models.resnet18(
    weights=None
)

model.fc = torch.nn.Sequential(

    torch.nn.Dropout(0.3),

    torch.nn.Linear(
        512,
        len(CLASS_NAMES)
    )
)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)


if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(DEVICE)

model.eval()

print("ResNet18 loaded successfully.")


# ============================================================
# FIND FIELD IMAGES
# ============================================================

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

images = sorted([

    path

    for path in FIELD_DIR.rglob("*")

    if path.is_file()
    and path.suffix.lower() in extensions

])


print()
print(f"Field images found: {len(images)}")


# ============================================================
# EMPTY DATASET CHECK
# ============================================================

if not images:

    print()
    print("=" * 70)
    print("NO FIELD IMAGES FOUND")
    print("=" * 70)

    print()
    print("Add real images using this structure:")
    print()
    print(
        "data\\field_test\\"
        "<CLASS_NAME>\\image.jpg"
    )

    print()
    print("Example:")
    print()
    print(
        "data\\field_test\\"
        "Tomato_Late_blight\\field_001.jpg"
    )

    print()

    raise SystemExit(0)


# ============================================================
# PREDICTION
# ============================================================

results = []

correct = 0
labeled = 0


for number, image_path in enumerate(
    images,
    start=1
):

    print()
    print(
        f"[{number}/{len(images)}] "
        f"{image_path.name}"
    )

    try:

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")


        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        tensor = transform(
            image
        )

        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            DEVICE
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]


        # ----------------------------------------------------
        # TOP 3
        # ----------------------------------------------------

        top_values, top_indices = torch.topk(
            probabilities,
            k=3
        )


        predicted_id = int(
            top_indices[0].item()
        )

        predicted_class = CLASS_NAMES[
            predicted_id
        ]

        confidence = float(
            top_values[0].item() * 100
        )


        # ----------------------------------------------------
        # ACTUAL CLASS FROM FOLDER
        # ----------------------------------------------------

        relative_path = image_path.relative_to(
            FIELD_DIR
        )

        parts = relative_path.parts


        if len(parts) >= 2:

            actual_class = parts[0]

        else:

            actual_class = ""


        is_labeled = (
            actual_class
            in CLASS_NAMES
        )


        correct_prediction = None


        if is_labeled:

            labeled += 1

            correct_prediction = (
                predicted_class
                == actual_class
            )

            if correct_prediction:

                correct += 1


        # ----------------------------------------------------
        # TOP 3 RESULTS
        # ----------------------------------------------------

        top3 = []

        for value, class_index in zip(
            top_values,
            top_indices
        ):

            top3.append({

                "class":
                    CLASS_NAMES[
                        int(
                            class_index.item()
                        )
                    ],

                "confidence_percent":
                    round(
                        float(
                            value.item() * 100
                        ),
                        4
                    )
            })


        # ----------------------------------------------------
        # TERMINAL OUTPUT
        # ----------------------------------------------------

        print(
            f"Prediction: "
            f"{predicted_class}"
        )

        print(
            f"Confidence: "
            f"{confidence:.2f}%"
        )


        if is_labeled:

            print(
                f"Actual: "
                f"{actual_class}"
            )

            print(
                "Result: "
                + (
                    "CORRECT"
                    if correct_prediction
                    else "INCORRECT"
                )
            )

        else:

            print(
                "Actual: Not labelled"
            )


        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append({

            "image":
                str(image_path),

            "actual_class":
                actual_class,

            "predicted_class":
                predicted_class,

            "confidence_percent":
                round(
                    confidence,
                    4
                ),

            "correct":
                correct_prediction,

            "top3":
                top3

        })


    except Exception as error:

        print(
            f"ERROR: {error}"
        )

        results.append({

            "image":
                str(image_path),

            "error":
                str(error)

        })


# ============================================================
# METRICS
# ============================================================

if labeled > 0:

    accuracy = (
        correct
        / labeled
        * 100
    )

else:

    accuracy = None


confidence_values = [

    item["confidence_percent"]

    for item in results

    if "confidence_percent" in item

]


if confidence_values:

    mean_confidence = float(
        np.mean(
            confidence_values
        )
    )

else:

    mean_confidence = 0.0


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "project":
        "AgriVision AI",

    "model":
        "ResNet18",

    "total_field_images":
        len(images),

    "labeled_images":
        labeled,

    "correct_predictions":
        correct,

    "incorrect_predictions":
        labeled - correct,

    "field_accuracy_percent":
        (
            round(
                accuracy,
                4
            )
            if accuracy is not None
            else None
        ),

    "mean_confidence_percent":
        round(
            mean_confidence,
            4
        )

}


# ============================================================
# JSON
# ============================================================

json_file = (
    OUTPUT_DIR
    / "field_validation_results.json"
)


with open(
    json_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "summary":
                summary,

            "results":
                results

        },
        file,
        indent=4
    )


# ============================================================
# CSV
# ============================================================

csv_file = (
    OUTPUT_DIR
    / "field_predictions.csv"
)


with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(

        file,

        fieldnames=[

            "image",
            "actual_class",
            "predicted_class",
            "confidence_percent",
            "correct"

        ]

    )

    writer.writeheader()


    for item in results:

        writer.writerow({

            "image":
                item.get(
                    "image",
                    ""
                ),

            "actual_class":
                item.get(
                    "actual_class",
                    ""
                ),

            "predicted_class":
                item.get(
                    "predicted_class",
                    ""
                ),

            "confidence_percent":
                item.get(
                    "confidence_percent",
                    ""
                ),

            "correct":
                item.get(
                    "correct",
                    ""
                )

        })


# ============================================================
# TEXT REPORT
# ============================================================

report_file = (
    OUTPUT_DIR
    / "FIELD_VALIDATION_REPORT.txt"
)


report = [

    "=" * 70,

    "AGRIVISION AI - REAL FIELD VALIDATION REPORT",

    "=" * 70,

    "",

    "MODEL",

    "ResNet18",

    "",

    "FIELD DATA",

    f"Total images: {len(images)}",

    f"Labeled images: {labeled}",

    "",

    "RESULTS",

    f"Correct predictions: {correct}",

    f"Incorrect predictions: "
    f"{labeled - correct}",

    (
        f"Field accuracy: "
        f"{accuracy:.4f}%"
        if accuracy is not None
        else
        "Field accuracy: Not available"
    ),

    f"Mean confidence: "
    f"{mean_confidence:.4f}%",

    "",

    "BENCHMARK REFERENCE",

    "PlantVillage test accuracy: 99.6130%",

    "",

    "OUTPUT FILES",

    "field_validation_results.json",

    "field_predictions.csv",

    "FIELD_VALIDATION_REPORT.txt",

    "",

    "=" * 70

]


with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report)
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("REAL-FIELD VALIDATION COMPLETE")
print("=" * 70)

print()
print(
    f"Images processed: {len(images)}"
)

print(
    f"Labeled images: {labeled}"
)

if accuracy is not None:

    print(
        f"Field Accuracy: "
        f"{accuracy:.4f}%"
    )

else:

    print(
        "Field Accuracy: "
        "Not calculated"
    )

print(
    f"Mean Confidence: "
    f"{mean_confidence:.4f}%"
)

print()
print(
    "Results:"
)

print(
    OUTPUT_DIR
)

print("=" * 70)
