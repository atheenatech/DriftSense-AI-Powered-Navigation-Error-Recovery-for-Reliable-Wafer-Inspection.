
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from driftsense.generator import generate_pair


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

TRAIN_COUNT = 800
VAL_COUNT = 200
TEST_COUNT = 200

TRAIN_START_SEED = 1000
VAL_START_SEED = 2000
TEST_START_SEED = 3000

DEFAULT_ARCHITECTURE = "DRAM"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

SPLITS = {
    "train": {
        "count": TRAIN_COUNT,
        "start_seed": TRAIN_START_SEED,
    },
    "val": {
        "count": VAL_COUNT,
        "start_seed": VAL_START_SEED,
    },
    "test": {
        "count": TEST_COUNT,
        "start_seed": TEST_START_SEED,
    },
}


# ============================================================
# DIRECTORY HELPERS
# ============================================================

def prepare_split_directory(
    split_dir: Path,
    clean: bool = False,
) -> None:
    """
    Prepare a dataset split directory.

    clean=True removes previously generated cases.
    """

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if clean:
        for item in split_dir.iterdir():

            if item.is_dir():
                shutil.rmtree(item)

            elif item.is_file():
                item.unlink()


# ============================================================
# GENERATE ONE CASE
# ============================================================

def generate_case(
    split_dir: Path,
    case_index: int,
    seed: int,
    architecture: str,
) -> dict:
    """
    Generate and save one synthetic pair.
    """

    case_name = f"case_{case_index:06d}"

    case_dir = split_dir / case_name

    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference, search, metadata = generate_pair(
        seed=seed,
        architecture=architecture,
    )

    # Use the existing generator's save mechanism by temporarily
    # creating the pair ourselves so the case directory remains
    # completely self-contained.
    #
    # Import here to keep the top-level imports simple.
    import cv2
    import numpy as np
    from dataclasses import asdict

    reference_u8 = np.clip(
        reference * 255,
        0,
        255,
    ).astype(np.uint8)

    search_u8 = np.clip(
        search * 255,
        0,
        255,
    ).astype(np.uint8)

    cv2.imwrite(
        str(case_dir / "reference.png"),
        reference_u8,
    )

    cv2.imwrite(
        str(case_dir / "search.png"),
        search_u8,
    )

    metadata_dict = asdict(metadata)

    metadata_dict["split"] = split_dir.name
    metadata_dict["case_index"] = case_index

    with open(
        case_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata_dict,
            f,
            indent=2,
        )

    return {
        "case_id": case_name,
        "split": split_dir.name,
        "seed": seed,
        "architecture": architecture,
        "target_x": metadata.target_x,
        "target_y": metadata.target_y,
        "reference_path": str(
            case_dir / "reference.png"
        ),
        "search_path": str(
            case_dir / "search.png"
        ),
        "metadata_path": str(
            case_dir / "metadata.json"
        ),
    }


# ============================================================
# GENERATE SPLIT
# ============================================================

def generate_split(
    output_root: Path,
    split_name: str,
    count: int,
    start_seed: int,
    architecture: str,
    clean: bool = False,
) -> list[dict]:

    split_dir = output_root / split_name

    prepare_split_directory(
        split_dir,
        clean=clean,
    )

    records = []

    print()
    print("=" * 60)
    print(f"Generating {split_name.upper()} split")
    print("=" * 60)
    print(f"Cases       : {count}")
    print(f"Seed range  : {start_seed} - {start_seed + count - 1}")
    print(f"Architecture: {architecture}")

    for index in range(count):

        seed = start_seed + index

        record = generate_case(
            split_dir=split_dir,
            case_index=index,
            seed=seed,
            architecture=architecture,
        )

        records.append(record)

        completed = index + 1

        if (
            completed == 1
            or completed % 50 == 0
            or completed == count
        ):
            print(
                f"[{completed:4d}/{count}] "
                f"seed={seed}"
            )

    return records


# ============================================================
# MANIFEST
# ============================================================

def create_manifest(
    output_root: Path,
    records: list[dict],
    architecture: str,
) -> dict:
    """
    Create one reproducibility manifest for the entire dataset.
    """

    split_counts = {}

    for record in records:
        split = record["split"]

        split_counts[split] = (
            split_counts.get(split, 0) + 1
        )

    manifest = {
        "dataset_name": "DriftSense Synthetic Localization Dataset",

        "version": "1.0",

        "synthetic_only": True,

        "architecture": architecture,

        "coordinate_convention": (
            "x=column, y=row; target coordinates are "
            "in original search-image pixels"
        ),

        "reference": {
            "width_px": 1000,
            "height_px": 1000,
            "pitch_nm_per_pixel": 1.0,
        },

        "search": {
            "width_px": 1000,
            "height_px": 1000,
            "pitch_nm_per_pixel": 10.0,
        },

        "physical_scale_ratio": 10.0,

        "reference_footprint_in_search_pixels": {
            "width": 100,
            "height": 100,
        },

        "split_counts": split_counts,

        "seed_ranges": {
            "train": [
                TRAIN_START_SEED,
                TRAIN_START_SEED + TRAIN_COUNT - 1,
            ],
            "val": [
                VAL_START_SEED,
                VAL_START_SEED + VAL_COUNT - 1,
            ],
            "test": [
                TEST_START_SEED,
                TEST_START_SEED + TEST_COUNT - 1,
            ],
        },

        "records": records,
    }

    manifest_path = (
        output_root / "dataset_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    return manifest


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate the DriftSense persistent "
            "synthetic dataset."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/generated/dataset",
        help="Dataset output directory.",
    )

    parser.add_argument(
        "--architecture",
        type=str,
        default=DEFAULT_ARCHITECTURE,
        choices=["DRAM", "FinFET"],
        help="Synthetic layout architecture.",
    )

    parser.add_argument(
        "--train",
        type=int,
        default=TRAIN_COUNT,
        help="Number of training cases.",
    )

    parser.add_argument(
        "--val",
        type=int,
        default=VAL_COUNT,
        help="Number of validation cases.",
    )

    parser.add_argument(
        "--test",
        type=int,
        default=TEST_COUNT,
        help="Number of test cases.",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete previously generated dataset files.",
    )

    args = parser.parse_args()

    output_root = Path(args.output)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_records = []

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_records = generate_split(
        output_root=output_root,
        split_name="train",
        count=args.train,
        start_seed=TRAIN_START_SEED,
        architecture=args.architecture,
        clean=args.clean,
    )

    all_records.extend(train_records)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_records = generate_split(
        output_root=output_root,
        split_name="val",
        count=args.val,
        start_seed=VAL_START_SEED,
        architecture=args.architecture,
        clean=args.clean,
    )

    all_records.extend(val_records)

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_records = generate_split(
        output_root=output_root,
        split_name="test",
        count=args.test,
        start_seed=TEST_START_SEED,
        architecture=args.architecture,
        clean=args.clean,
    )

    all_records.extend(test_records)

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = create_manifest(
        output_root=output_root,
        records=all_records,
        architecture=args.architecture,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)

    print(
        f"Train      : "
        f"{len(train_records)}"
    )

    print(
        f"Validation : "
        f"{len(val_records)}"
    )

    print(
        f"Test       : "
        f"{len(test_records)}"
    )

    print(
        f"Total      : "
        f"{len(all_records)}"
    )

    print(
        f"Architecture: "
        f"{manifest['architecture']}"
    )

    print(
        f"Output      : "
        f"{output_root.resolve()}"
    )

    print(
        f"Manifest    : "
        f"{output_root / 'dataset_manifest.json'}"
    )


if __name__ == "__main__":
    main()
