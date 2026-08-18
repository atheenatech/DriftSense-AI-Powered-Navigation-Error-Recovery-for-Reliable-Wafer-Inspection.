"""
create_defect8.py

Builds the Defect8 dataset from the raw LSWMD/WM-811K pickle.

- Reads Dataset/Raw/LSWMD.pkl (legacy Python 2 / old-pandas pickle)
- Keeps the 172,950 labeled wafers, maps their failure types to 9 classes:
      index  class
      0      none
      1      center
      2      donut
      3      edge-loc
      4      edge-ring
      5      loc
      6      near-full
      7      random
      8      scratch
- Resizes every wafer map (variable shapes) to a fixed size with
  nearest-neighbor scaling, preserving the exact 0/1/2 die values
- Splits by the dataset's own Training/Test partition
- Saves Dataset/Defect8/train.npz and Dataset/Defect8/test.npz

Usage:
    python create_defect8.py [--size 64] [--cap-none 0] [--shuffle-seed 0]
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

RAW_PKL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "Dataset", "Raw", "LSWMD.pkl"
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "Dataset", "Defect8"
)

CLASS_NAMES = [
    "none",
    "center",
    "donut",
    "edge-loc",
    "edge-ring",
    "loc",
    "near-full",
    "random",
    "scratch",
]


def register_legacy_pandas_aliases():
    """Old pickles reference pandas.indexes.* which no longer exists."""
    targets = [
        "pandas.core.indexes",
        "pandas.core.indexes.base",
        "pandas.core.indexes.range",
        "pandas.core.indexes.numeric",
        "pandas.core.indexes.datetimes",
        "pandas.core.indexes.period",
        "pandas.core.indexes.timedeltas",
        "pandas.core.indexes.multi",
        "pandas.core.indexes.category",
    ]
    for target in targets:
        try:
            real = __import__(target, fromlist=["*"])
            alias = "pandas.indexes" + target[len("pandas.core.indexes"):]
            sys.modules[alias] = real
        except ImportError:
            pass


def load_raw(path):
    register_legacy_pandas_aliases()
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def scalar_of(cell):
    """Label cells are 1x1 numpy string arrays; collapse to a plain str."""
    if isinstance(cell, np.ndarray):
        if cell.size == 0:
            return ""
        return str(cell.ravel()[0])
    if cell is None:
        return ""
    return str(cell)


def die_size_of(cell):
    try:
        arr = np.asarray(cell, dtype=np.float64)
        if arr.size == 0:
            return float("nan")
        return float(arr.ravel()[0])
    except (ValueError, TypeError):
        return float("nan")


def resize_nearest(map_2d, size):
    """Nearest-neighbor resize preserving exact die values (0/1/2)."""
    h, w = map_2d.shape
    if h == size and w == size:
        return map_2d
    rows = (np.arange(size, dtype=np.int64) * h) // size
    cols = (np.arange(size, dtype=np.int64) * w) // size
    return map_2d[np.ix_(rows, cols)]


def main():
    ap = argparse.ArgumentParser(description="Build the Defect8 dataset from LSWMD.pkl")
    ap.add_argument("--size", type=int, default=64, help="Resized wafer map size (default 64)")
    ap.add_argument(
        "--cap-none",
        type=int,
        default=0,
        help="Max 'none' samples per split, 0 keeps all (default 0)",
    )
    ap.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Shuffle samples within each split; 0 disables (default 0)",
    )
    args = ap.parse_args()

    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}

    t0 = time.time()
    print(f"[create_defect8] loading {RAW_PKL} ...")
    df = load_raw(RAW_PKL)
    n_total = len(df)
    print(f"[create_defect8] loaded {n_total} wafers in {time.time() - t0:.1f}s")

    ft_col = df["failureType"].values
    ttl_col = df["trianTestLabel"].values
    wm_col = df["waferMap"].values
    ds_col = df["dieSize"].values

    splits = {"Training": [], "Test": []}
    skipped_missing = 0
    skipped_unknown = 0

    for i in range(n_total):
        label = scalar_of(ft_col[i]).strip().lower()
        if not label:
            skipped_missing += 1
            continue
        if label not in class_to_id:
            skipped_unknown += 1
            continue
        split = scalar_of(ttl_col[i]).strip().lower()
        if split not in ("training", "test"):
            skipped_missing += 1
            continue
        key = "Training" if split == "training" else "Test"
        splits[key].append(
            (resize_nearest(wm_col[i], args.size), class_to_id[label], die_size_of(ds_col[i]))
        )

    print(
        f"[create_defect8] kept {len(splits['Training']) + len(splits['Test'])} wafers "
        f"(skipped {skipped_missing} unlabeled/missing, {skipped_unknown} unknown labels)"
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    for key in ("Training", "Test"):
        rows = splits[key]
        if args.shuffle_seed:
            rng = np.random.default_rng(args.shuffle_seed)
            rng.shuffle(rows)
        if args.cap_none > 0:
            none_ids = [j for j, r in enumerate(rows) if r[1] == 0]
            if len(none_ids) > args.cap_none:
                drop = set(
                    np.random.default_rng(args.shuffle_seed + 1).choice(
                        none_ids, len(none_ids) - args.cap_none, replace=False
                    )
                )
                rows = [r for j, r in enumerate(rows) if j not in drop]

        if not rows:
            print(f"[create_defect8] WARNING: split '{key}' is empty, skipping file")
            continue

        maps = np.stack([r[0] for r in rows]).astype(np.uint8)
        labels = np.array([r[1] for r in rows], dtype=np.int64)
        die_sizes = np.array([r[2] for r in rows], dtype=np.float32)

        out_path = os.path.join(OUT_DIR, key.lower() + ".npz")
        np.savez(
            out_path,
            wafer_maps=maps,
            labels=labels,
            die_sizes=die_sizes,
            class_names=np.array(CLASS_NAMES, dtype="U16"),
        )

        counts, _ = np.histogram(labels, bins=len(CLASS_NAMES), range=(0, len(CLASS_NAMES)))
        summary = ", ".join(f"{CLASS_NAMES[j]}={counts[j]}" for j in range(len(CLASS_NAMES)))
        print(f"[create_defect8] wrote {out_path} with {len(rows)} wafers ({summary})")

    print(f"[create_defect8] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()