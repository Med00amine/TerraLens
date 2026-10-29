"""High-level inference: load a checkpoint once, then classify and explain images."""

from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from terralens import __version__
from terralens.checkpoint import load_checkpoint, model_from_checkpoint
from terralens.constants import color_for, label_for
from terralens.data import build_transforms
from terralens.explain import GradCAM, overlay_heatmap
from terralens.models import count_parameters

DEFAULT_MODEL_PATH = Path("models/terralens.pt")


def default_model_path() -> Path:
    """Model location: the TERRALENS_MODEL environment variable, else models/terralens.pt."""
    return Path(os.environ.get("TERRALENS_MODEL", DEFAULT_MODEL_PATH))


@dataclass(frozen=True)
class Prediction:
    class_name: str
    label: str
    probability: float
    color: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Explanation:
    overlay: Image.Image
    heatmap: np.ndarray
    explained: Prediction
    predictions: list[Prediction]


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class Predictor:
    """Wraps a trained checkpoint for single-tile and batched inference."""

    def __init__(self, checkpoint: dict, device: str = "auto"):
        self.checkpoint = checkpoint
        self.device = resolve_device(device)
        self.model = model_from_checkpoint(checkpoint, self.device)
        self.class_names: list[str] = list(checkpoint["class_names"])
        self.img_size = int(checkpoint["img_size"])
        self._mean = torch.tensor(checkpoint["mean"], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(checkpoint["std"], device=self.device).view(1, 3, 1, 1)
        self._transform = build_transforms(
            self.img_size, train=False, mean=checkpoint["mean"], std=checkpoint["std"]
        )
        # Grad-CAM keeps per-call state in hooks, so explanations are serialised.
        self._explain_lock = threading.Lock()

    @classmethod
    def from_path(cls, path: str | Path, device: str = "auto") -> Predictor:
        return cls(load_checkpoint(path), device)

    def warmup(self) -> None:
        """One dummy forward pass, so the first request doesn't pay for CUDA/cuDNN set-up."""
        self.predict_proba(torch.zeros(1, 3, self.img_size, self.img_size))

    # ------------------------------------------------------------------ preprocessing
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """A single image as a normalised (1, 3, S, S) tensor, as seen in training."""
        return self._transform(image.convert("RGB")).unsqueeze(0)

    def preprocess_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        """Batched equivalent of `preprocess` for uint8 tiles of shape (N, 3, t, t)."""
        x = tiles.to(self.device, non_blocking=True).float().div_(255)
        if x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
                antialias=x.shape[-1] > self.img_size,
            )
        return (x - self._mean) / self._std

    # ---------------------------------------------------------------------- inference
    @torch.inference_mode()
    def predict_proba(self, batch: torch.Tensor) -> np.ndarray:
        """Class probabilities (N, C) for a normalised batch."""
        logits = self.model(batch.to(self.device))
        return torch.softmax(logits.float(), dim=1).cpu().numpy()

    def _ranked(self, probs: np.ndarray, top_k: int | None) -> list[Prediction]:
        order = np.argsort(-probs)[: top_k or len(probs)]
        return [self._prediction(i, probs[i]) for i in order]

    def _prediction(self, idx: int, prob: float) -> Prediction:
        name = self.class_names[idx]
        return Prediction(name, label_for(name), float(prob), color_for(name))

    def predict(self, image: Image.Image, top_k: int | None = 5) -> list[Prediction]:
        """Classes sorted by probability for one tile."""
        return self._ranked(self.predict_proba(self.preprocess(image))[0], top_k)

    def explain(
        self,
        image: Image.Image,
        class_name: str | None = None,
        alpha: float = 0.5,
        size: int | None = 256,
        top_k: int | None = 5,
    ) -> Explanation:
        """Grad-CAM for the top class (or `class_name`) with a heatmap overlay."""
        class_idx = None
        if class_name is not None:
            if class_name not in self.class_names:
                raise ValueError(f"Unknown class {class_name!r}")
            class_idx = self.class_names.index(class_name)
        x = self.preprocess(image).to(self.device)
        with self._explain_lock, GradCAM(self.model) as cam_fn:
            heatmap, idx, probs = cam_fn(x, class_idx)
        probs = probs.numpy()
        return Explanation(
            overlay=overlay_heatmap(image, heatmap, alpha=alpha, size=size),
            heatmap=heatmap,
            explained=self._prediction(idx, probs[idx]),
            predictions=self._ranked(probs, top_k),
        )

    def info(self) -> dict:
        """Model card data for the API and UI."""
        return {
            "architecture": self.checkpoint["architecture"],
            "img_size": self.img_size,
            "parameters": count_parameters(self.model),
            "classes": [
                {"name": n, "label": label_for(n), "color": color_for(n)} for n in self.class_names
            ],
            "metrics": self.checkpoint.get("metrics", {}),
            "trained_with": self.checkpoint.get("terralens_version"),
            "created_at": self.checkpoint.get("created_at"),
            "device": str(self.device),
            "terralens_version": __version__,
        }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Classify satellite tiles.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, default=None, help="checkpoint path")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--explain-dir", type=Path, default=None, help="save Grad-CAM overlays")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)

    predictor = Predictor.from_path(args.model or default_model_path())
    results = {}
    for path in args.images:
        image = Image.open(path)
        preds = predictor.predict(image, top_k=args.top_k)
        results[str(path)] = [p.to_dict() for p in preds]
        if args.explain_dir is not None:
            args.explain_dir.mkdir(parents=True, exist_ok=True)
            predictor.explain(image).overlay.save(args.explain_dir / f"{path.stem}_gradcam.png")
        if not args.json:
            ranked = ", ".join(f"{p.label} {p.probability:.1%}" for p in preds)
            print(f"{path}: {ranked}")
    if args.json:
        print(json.dumps(results, indent=2))
