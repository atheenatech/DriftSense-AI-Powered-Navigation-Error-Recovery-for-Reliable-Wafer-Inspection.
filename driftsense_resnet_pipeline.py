"""End-to-end DriftSense hybrid localization plus ResNet18 classification."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

INTEGRATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INTEGRATION_DIR.parents[1]
TRAINING_DIR = PROJECT_ROOT / "AI" / "Training"
CHECKPOINT_PATH = PROJECT_ROOT / "Results" / "Training" / "best_model.pth"
CLASSES_PATH = PROJECT_ROOT / "Dataset" / "Defect8" / "classes.npy"

sys.path.insert(0, str(INTEGRATION_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from driftsense_hybrid import crop_candidate, infer_case, load_grayscale  # noqa: E402
from model import create_model  # noqa: E402


def load_resnet18(checkpoint_path: str | Path = CHECKPOINT_PATH):
    model = create_model()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_resnet(crop: np.ndarray) -> torch.Tensor:
    image = crop.astype(np.float32)
    maximum = float(image.max())
    if maximum > 0:
        image = image / maximum
    tensor = torch.from_numpy(image)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor.unsqueeze(0), size=(224, 224), mode="nearest")
    return tensor.repeat(1, 3, 1, 1)


def classify_crop(model, crop: np.ndarray, classes: list[str]) -> dict:
    tensor = preprocess_resnet(crop)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        confidence, index = torch.max(probabilities, dim=0)
    return {
        "class_index": int(index.item()),
        "class_name": classes[int(index.item())],
        "confidence": float(confidence.item()),
        "probabilities": [float(value) for value in probabilities.tolist()],
    }


def run_pipeline(case_dir: str | Path, ranker_weights: str | Path, resnet_weights: str | Path = CHECKPOINT_PATH, output_dir: str | Path | None = None) -> dict:
    case_dir = Path(case_dir)
    output_dir = Path(output_dir) if output_dir else case_dir / "integration_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    hybrid = infer_case(case_dir, ranker_weights, top_k=8)
    selected = hybrid["selected"]
    reference = load_grayscale(case_dir / "reference.png")
    search = load_grayscale(case_dir / "search.png")
    crop = crop_candidate(search, selected, size=(100, 100))
    correction = {
        "dx_px": float(selected["x"] - search.shape[1] / 2.0),
        "dy_px": float(selected["y"] - search.shape[0] / 2.0),
        "magnitude_px": float(np.hypot(selected["x"] - search.shape[1] / 2.0, selected["y"] - search.shape[0] / 2.0)),
    }
    result = {
        "case": case_dir.name,
        "localization": hybrid,
        "correction": correction,
        "classification_status": "abstained_uncertain_localization",
    }
    if hybrid["status"] == "actionable":
        classes = np.load(CLASSES_PATH, allow_pickle=True).tolist()
        model = load_resnet18(resnet_weights)
        result["classification_status"] = "actionable"
        result["defect_classification"] = classify_crop(model, crop, classes)
    crop_path = output_dir / f"{case_dir.name}_localized_crop.npy"
    np.save(crop_path, crop)
    result["localized_crop_path"] = str(crop_path)
    result_path = output_dir / f"{case_dir.name}_integrated_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", help="Case containing reference.png and search.png")
    parser.add_argument("--ranker-weights", required=True)
    parser.add_argument("--resnet-weights", default=str(CHECKPOINT_PATH))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    result = run_pipeline(args.case_dir, Path(args.ranker_weights), args.resnet_weights, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
