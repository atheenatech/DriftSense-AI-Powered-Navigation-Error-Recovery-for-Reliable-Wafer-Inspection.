import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# DEFECT8 STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "Dataset" / "Defect8"

LABELS_FILE = DATASET_DIR / "labels.npy"
CLASSES_FILE = DATASET_DIR / "classes.npy"

OUTPUT_DIR = DATASET_DIR / "splits"

TRAIN_FILE = OUTPUT_DIR / "train_indices.npy"
VAL_FILE = OUTPUT_DIR / "val_indices.npy"

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

print("=" * 60)
print("DEFECT8 STRATIFIED DATASET SPLIT")
print("=" * 60)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

labels = np.load(LABELS_FILE)
classes = np.load(CLASSES_FILE, allow_pickle=True)

print(f"\nTotal samples: {len(labels)}")
print(f"Number of classes: {len(classes)}")

# ------------------------------------------------------------
# Create index array
# ------------------------------------------------------------

indices = np.arange(len(labels))

# ------------------------------------------------------------
# Stratified split
# ------------------------------------------------------------

train_indices, val_indices = train_test_split(
    indices,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)

# ------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Save indices
# ------------------------------------------------------------

np.save(TRAIN_FILE, train_indices)
np.save(VAL_FILE, val_indices)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\nSplit completed successfully.")

print(f"Training samples:   {len(train_indices)}")
print(f"Validation samples: {len(val_indices)}")

print("\nClass distribution:")
print("-" * 60)

for class_id, class_name in enumerate(classes):

    train_count = np.sum(labels[train_indices] == class_id)
    val_count = np.sum(labels[val_indices] == class_id)

    print(
        f"{class_id}: {class_name:<12} "
        f"Train={train_count:>5}  "
        f"Val={val_count:>5}"
    )

# ------------------------------------------------------------
# Verify every class exists in both sets
# ------------------------------------------------------------

for class_id in range(len(classes)):

    train_count = np.sum(labels[train_indices] == class_id)
    val_count = np.sum(labels[val_indices] == class_id)

    if train_count == 0 or val_count == 0:
        raise RuntimeError(
            f"Class {class_id} does not exist in both splits!"
        )

print("\nVerification: PASSED")

print("\nFiles created:")
print(TRAIN_FILE)
print(VAL_FILE)

print("=" * 60)