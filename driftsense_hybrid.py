"""Hybrid DriftSense: scale-aware NCC candidates plus an AI candidate ranker."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_grayscale(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image.astype(np.float32) / 255.0


def scale_reference_to_search(reference: np.ndarray, reference_pitch_nm: float = 1.0, search_pitch_nm: float = 10.0) -> np.ndarray:
    scale = float(reference_pitch_nm / search_pitch_nm)
    height, width = reference.shape[:2]
    new_width = max(8, int(round(width * scale)))
    new_height = max(8, int(round(height * scale)))
    return cv2.resize(reference, (new_width, new_height), interpolation=cv2.INTER_AREA)


class RankerEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(64, embedding_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.net(image).flatten(1)), dim=1)


class CandidateRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RankerEncoder()
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, reference: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        reference_embedding = self.encoder(reference)
        candidate_embedding = self.encoder(candidates)
        pair = torch.cat((reference_embedding.expand_as(candidate_embedding), candidate_embedding), dim=1)
        return self.head(pair).squeeze(1)


def extract_candidates(reference: np.ndarray, search: np.ndarray, top_k: int = 8, min_distance: int = 40) -> list[dict]:
    template = scale_reference_to_search(reference)
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    h, w = template.shape
    work = response.copy()
    candidates = []
    for _ in range(top_k):
        _, score, _, location = cv2.minMaxLoc(work)
        x, y = location
        candidates.append({"x": float(x + w / 2), "y": float(y + h / 2), "ncc_score": float(score), "top_left_x": int(x), "top_left_y": int(y)})
        yy, xx = np.ogrid[:work.shape[0], :work.shape[1]]
        work[(xx - x) ** 2 + (yy - y) ** 2 <= min_distance ** 2] = -1.0
    return candidates


def crop_candidate(search: np.ndarray, candidate: dict, size: tuple[int, int] = (100, 100)) -> np.ndarray:
    h, w = size[1], size[0]
    cx, cy = candidate["x"], candidate["y"]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    canvas = np.full((h, w), float(search.mean()), dtype=np.float32)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(search.shape[1], x0 + w), min(search.shape[0], y0 + h)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = search[sy0:sy1, sx0:sx1]
    return canvas


def infer_case(case_dir: str | Path, weights: str | Path | None = None, top_k: int = 8) -> dict:
    case_dir = Path(case_dir)
    reference = load_grayscale(case_dir / "reference.png")
    search = load_grayscale(case_dir / "search.png")
    candidates = extract_candidates(reference, search, top_k=top_k)
    result = {"case": case_dir.name, "method": "hybrid_ncc_candidate_ranker", "candidates": candidates}
    if weights is None or not Path(weights).exists():
        best = candidates[0]
        result.update({"selected": best, "status": "ncc_fallback", "confidence": best["ncc_score"]})
        return result
    model = CandidateRanker()
    model.load_state_dict(torch.load(weights, map_location="cpu")["model_state_dict"])
    model.eval()
    reference_scaled = scale_reference_to_search(reference)
    ref_tensor = torch.from_numpy(cv2.resize(reference_scaled, (128, 128), interpolation=cv2.INTER_AREA)[None, None]).float()
    candidate_images = np.stack([cv2.resize(crop_candidate(search, candidate), (128, 128), interpolation=cv2.INTER_AREA) for candidate in candidates])
    candidate_tensor = torch.from_numpy(candidate_images[:, None]).float()
    with torch.no_grad():
        scores = torch.sigmoid(model(ref_tensor, candidate_tensor)).numpy()
    for candidate, score in zip(candidates, scores):
        candidate["ai_score"] = float(score)
        candidate["combined_score"] = float(0.35 * candidate["ncc_score"] + 0.65 * score)
    ranked = sorted(candidates, key=lambda item: item["combined_score"], reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    margin = best["combined_score"] - second["combined_score"] if second else best["combined_score"]
    result.update({"selected": best, "status": "actionable" if margin >= 0.08 else "uncertain", "confidence": float(margin), "ranked_candidates": ranked})
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    print(json.dumps(infer_case(args.case_dir, args.weights), indent=2))


if __name__ == "__main__":
    main()
