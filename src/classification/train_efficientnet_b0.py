"""
AgriVision AI - EfficientNet-B0 Training

Train EfficientNet-B0 using the existing PlantVillage
classification split.

CPU-friendly transfer learning configuration.
"""

from pathlib import Path
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "classification" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "processed" / "classification" / "val"

MODEL_DIR = PROJECT_ROOT / "models" / "classification"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-4
PATIENCE = 3
NUM_WORKERS = 0


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def build_model(num_classes):

    print("Loading pretrained EfficientNet-B0...")

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    # Freeze backbone initially.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    input_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            input_features,
            num_classes,
        ),
    )

    return model


def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        total_loss / total,
        correct / total,
    )


def main():

    print("=" * 70)
    print("AgriVision AI - EfficientNet-B0 Training")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Validation directory: {VAL_DIR}")

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=val_transform,
    )

    num_classes = len(
        train_dataset.classes
    )

    print()
    print(
        f"Classes: {num_classes}"
    )

    print(
        f"Training images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    model = build_model(
        num_classes
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    history = []

    best_model_path = (
        MODEL_DIR
        / "efficientnet_b0_best.pth"
    )

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_index, (images, labels) in enumerate(
            train_loader,
            start=1,
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            if batch_index % 100 == 0:

                print(
                    f"Epoch {epoch}/{EPOCHS} "
                    f"Batch {batch_index}/"
                    f"{len(train_loader)}",
                    flush=True,
                )

        train_loss = (
            running_loss / total
        )

        train_accuracy = (
            correct / total
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(
            val_accuracy
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_seconds": epoch_time,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        history.append(record)

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )
        print(
            f"Train Loss: {train_loss:.4f}"
        )
        print(
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )
        print(
            f"Val Loss: {val_loss:.4f}"
        )
        print(
            f"Val Accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )
        print(
            f"Time: {epoch_time:.1f}s"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": train_dataset.classes,
                    "best_val_accuracy": best_val_accuracy,
                },
                best_model_path,
            )

            print(
                "✓ New best model saved."
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement: "
                f"{epochs_without_improvement}/"
                f"{PATIENCE}"
            )

        if epochs_without_improvement >= PATIENCE:

            print(
                "Early stopping."
            )

            break

    total_time = (
        time.time() - start_time
    )

    results = {
        "model": "EfficientNet-B0",
        "device": str(DEVICE),
        "classes": num_classes,
        "train_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history),
        "best_validation_accuracy": best_val_accuracy,
        "total_training_seconds": total_time,
        "best_model": str(best_model_path),
        "class_names": train_dataset.classes,
        "history": history,
    }

    results_path = (
        REPORT_DIR
        / "efficientnet_b0_training_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("EFFICIENTNET-B0 TRAINING FINISHED")
    print("=" * 70)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Model saved: "
        f"{best_model_path}"
    )

    print(
        f"Results saved: "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()