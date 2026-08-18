"""Evaluate DriftSense plus ResNet18 integration over a test directory."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

try:
    from .driftsense_resnet_pipeline import run_pipeline
except ImportError:
    from driftsense_resnet_pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--ranker-weights", required=True)
    parser.add_argument("--resnet-weights", default="Results/Training/best_model.pth")
    parser.add_argument("--output-dir", default="Results/DriftSense/integration_all")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted(path for path in test_dir.iterdir() if path.is_dir() and path.name.startswith("case_"))
    if args.limit > 0:
        cases = cases[: args.limit]

    rows = []
    start = time.perf_counter()
    for index, case_dir in enumerate(cases, start=1):
        result = run_pipeline(case_dir, args.ranker_weights, args.resnet_weights, output_dir / case_dir.name)
        loc = result["localization"]
        selected = loc["selected"]
        defect = result.get("defect_classification") or {}
        rows.append({
            "case": result["case"],
            "status": loc["status"],
            "confidence": float(loc["confidence"]),
            "x": float(selected["x"]),
            "y": float(selected["y"]),
            "dx_px": float(result["correction"]["dx_px"]),
            "dy_px": float(result["correction"]["dy_px"]),
            "correction_magnitude_px": float(result["correction"]["magnitude_px"]),
            "defect_class": defect.get("class_name", "ABSTAINED"),
            "defect_confidence": float(defect.get("confidence", 0.0)),
            "classification_status": result["classification_status"],
        })
        if index % 25 == 0 or index == len(cases):
            print(f"processed {index}/{len(cases)}")

    csv_path = output_dir / "integrated_results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    actionable = sum(row["status"] == "actionable" for row in rows)
    abstained = len(rows) - actionable
    summary = {
        "count": len(rows),
        "actionable_count": actionable,
        "actionable_percent": 100.0 * actionable / len(rows) if rows else 0.0,
        "abstained_count": abstained,
        "abstained_percent": 100.0 * abstained / len(rows) if rows else 0.0,
        "classified_count": sum(row["classification_status"] == "actionable" for row in rows),
        "mean_confidence": sum(row["confidence"] for row in rows) / len(rows) if rows else 0.0,
        "mean_correction_magnitude_px": sum(row["correction_magnitude_px"] for row in rows) / len(rows) if rows else 0.0,
        "runtime_seconds": time.perf_counter() - start,
        "results_csv": str(csv_path),
    }
    (output_dir / "integrated_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
