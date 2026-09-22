from pathlib import Path
import json
import copy
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ============================================================
# AGRIVISION AI
# RESNET18 HIGH-ACCURACY CLASSIFIER
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

TRAIN_DIR = Path("data/processed/classification/train")
VAL_DIR = Path("data/processed/classification/val")
TEST_DIR = Path("data/processed/classification/test")

MODEL_DIR = Path("models/classification")
REPORT_DIR = Path("data/reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = MODEL_DIR / "resnet18_best.pth"

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

IMAGE_SIZE = 224
BATCH_SIZE = 32

# CPU-friendly
NUM_WORKERS = 0

EPOCHS = 30

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-4

PATIENCE = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("             AGRIVISION AI - RESNET18 TRAINING")
print("=" * 70)

print(f"\nDevice: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")

# ------------------------------------------------------------
# ImageNet normalization
# ------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ------------------------------------------------------------
# Training augmentation
# ------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomVerticalFlip(p=0.15),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.20,
        hue=0.05
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.90, 1.10)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    ),

    transforms.RandomErasing(
        p=0.20,
        scale=(0.02, 0.12),
        ratio=(0.3, 3.3)
    )
])

# ------------------------------------------------------------
# Validation / test transformation
# ------------------------------------------------------------

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform
)

classes = train_dataset.classes

NUM_CLASSES = len(classes)

print(f"\nNumber of classes: {NUM_CLASSES}")

print("\nClasses:")

for index, class_name in enumerate(classes):
    print(f"{index:2d} -> {class_name}")

# ------------------------------------------------------------
# Verify class mapping
# ------------------------------------------------------------

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise RuntimeError(
        "Training and validation class mappings do not match."
    )

if train_dataset.class_to_idx != test_dataset.class_to_idx:
    raise RuntimeError(
        "Training and testing class mappings do not match."
    )

# ------------------------------------------------------------
# Dataset sizes
# ------------------------------------------------------------

print("\nDataset sizes:")
print(f"Train: {len(train_dataset):,}")
print(f"Val:   {len(val_dataset):,}")
print(f"Test:  {len(test_dataset):,}")

# ------------------------------------------------------------
# Weighted sampler
# ------------------------------------------------------------

train_targets = np.array(train_dataset.targets)

class_counts = np.bincount(
    train_targets,
    minlength=NUM_CLASSES
)

print("\nTraining class counts:")

for index, count in enumerate(class_counts):

    print(
        f"{classes[index]:<65} {count:>6}"
    )

# Inverse-frequency weighting
class_weights = 1.0 / np.maximum(class_counts, 1)

sample_weights = class_weights[train_targets]

sample_weights = torch.as_tensor(
    sample_weights,
    dtype=torch.double
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# ------------------------------------------------------------
# DataLoaders
# ------------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# ------------------------------------------------------------
# Load pretrained ResNet18
# ------------------------------------------------------------

print("\nLoading pretrained ResNet18...")

try:
    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

except Exception as error:

    print("\nWARNING: Could not load pretrained weights.")
    print(error)

    model = models.resnet18(
        weights=None
    )

# ------------------------------------------------------------
# Replace classifier
# ------------------------------------------------------------

in_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(p=0.30),
    nn.Linear(in_features, NUM_CLASSES)
)

model = model.to(DEVICE)

# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------

# Label smoothing improves generalization.
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)

# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# ------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.3,
    patience=2,
    min_lr=1e-6
)

# ------------------------------------------------------------
# Training function
# ------------------------------------------------------------

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    from tqdm import tqdm

    progress = tqdm(
        train_loader,
        desc="Training",
        unit="batch",
        dynamic_ncols=True
    )

    for batch_index, (images, labels) in enumerate(progress, start=1):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        current_loss = running_loss / total
        current_accuracy = correct / total

        progress.set_postfix(
            loss=f"{current_loss:.4f}",
            acc=f"{current_accuracy * 100:.2f}%"
        )

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ------------------------------------------------------------
# Validation function
# ------------------------------------------------------------

def evaluate(loader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    loss_value = running_loss / total

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        )
    )

    return (
        loss_value,
        accuracy,
        precision,
        recall,
        f1,
        all_labels,
        all_predictions
    )


# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------

best_val_accuracy = 0.0
best_state = None

epochs_without_improvement = 0

history = []

print("\n")
print("=" * 70)
print("                    STARTING TRAINING")
print("=" * 70)

training_start = time.time()

for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    train_loss, train_accuracy = train_one_epoch()

    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
        _,
        _
    ) = evaluate(val_loader)

    scheduler.step(val_accuracy)

    current_lr = optimizer.param_groups[0]["lr"]

    epoch_time = time.time() - epoch_start

    print(
        f"\nEpoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: {val_accuracy * 100:.2f}%"
    )

    print(
        f"Val Precision: {val_precision * 100:.2f}%"
    )

    print(
        f"Val Recall: {val_recall * 100:.2f}%"
    )

    print(
        f"Val F1: {val_f1 * 100:.2f}%"
    )

    print(
        f"Learning Rate: {current_lr:.7f}"
    )

    print(
        f"Time: {epoch_time:.1f}s"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "learning_rate": current_lr
    })

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": best_state,
                "classes": classes,
                "class_to_idx": train_dataset.class_to_idx,
                "image_size": IMAGE_SIZE,
                "best_val_accuracy": best_val_accuracy
            },
            BEST_MODEL
        )

        epochs_without_improvement = 0

        print(
            f"\n*** NEW BEST MODEL: "
            f"{best_val_accuracy * 100:.2f}% ***"
        )

    else:

        epochs_without_improvement += 1

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping after {epoch} epochs."
        )

        break


training_time = time.time() - training_start

# ------------------------------------------------------------
# Load best model
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("                    FINAL EVALUATION")
print("=" * 70)

if best_state is not None:

    model.load_state_dict(best_state)

else:

    checkpoint = torch.load(
        BEST_MODEL,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

# ------------------------------------------------------------
# Test evaluation
# ------------------------------------------------------------

(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1,
    test_labels,
    test_predictions
) = evaluate(test_loader)

print(
    f"\nTest Loss:       {test_loss:.4f}"
)

print(
    f"Test Accuracy:   {test_accuracy * 100:.2f}%"
)

print(
    f"Test Precision:  {test_precision * 100:.2f}%"
)

print(
    f"Test Recall:     {test_recall * 100:.2f}%"
)

print(
    f"Test F1 Score:   {test_f1 * 100:.2f}%"
)

print(
    f"\nBest Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Training Time: "
    f"{training_time / 60:.2f} minutes"
)

# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

report = classification_report(
    test_labels,
    test_predictions,
    target_names=classes,
    digits=4,
    zero_division=0
)

print("\n")
print("=" * 70)
print("                 CLASSIFICATION REPORT")
print("=" * 70)

print(report)

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    test_labels,
    test_predictions
)

# ------------------------------------------------------------
# Save complete report
# ------------------------------------------------------------

results = {
    "model": "ResNet18",
    "device": str(DEVICE),

    "dataset": {
        "train": len(train_dataset),
        "validation": len(val_dataset),
        "test": len(test_dataset),
        "classes": len(classes)
    },

    "best_validation_accuracy": best_val_accuracy,

    "test": {
        "loss": test_loss,
        "accuracy": test_accuracy,
        "precision_weighted": test_precision,
        "recall_weighted": test_recall,
        "f1_weighted": test_f1
    },

    "training": {
        "epochs_completed": len(history),
        "training_time_minutes": training_time / 60,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY
    },

    "classes": classes,

    "confusion_matrix": cm.tolist(),

    "history": history,

    "classification_report": report
}

report_path = REPORT_DIR / "resnet18_results.json"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )

# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("                  TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest model:"
)

print(
    BEST_MODEL
)

print(
    f"\nResults:"
)

print(
    report_path
)

print("\n" + "=" * 70)
