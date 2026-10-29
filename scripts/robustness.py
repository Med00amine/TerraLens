"""Test accuracy under JPEG re-compression, a common corruption in user uploads.

python scripts/robustness.py runs/resnet18-128/model.pt runs/resnet18-128-jpeg/model.pt
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from terralens.checkpoint import load_checkpoint, model_from_checkpoint
from terralens.data import TransformedSubset, build_transforms, load_dataset, stratified_split
from terralens.evaluate import predict_loader

QUALITIES = (None, 95, 75, 50)


class FixedJPEG:
    def __init__(self, quality: int):
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/results/jpeg_robustness.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for path in args.checkpoints:
        ckpt = load_checkpoint(path)
        cfg = ckpt["config"]
        dataset = load_dataset(cfg["dataset"], cfg["data_root"])
        split = stratified_split(
            dataset.targets, cfg["val_fraction"], cfg["test_fraction"], cfg["seed"]
        )
        model = model_from_checkpoint(ckpt, device)
        base = build_transforms(ckpt["img_size"], train=False, mean=ckpt["mean"], std=ckpt["std"])
        row = {}
        for q in QUALITIES:
            tf = base if q is None else transforms.Compose([FixedJPEG(q), base])
            loader = DataLoader(
                TransformedSubset(dataset, split.test, tf), batch_size=256, num_workers=4
            )
            y, probs = predict_loader(model, loader, device, ckpt.get("temperature", 1.0))
            row["original" if q is None else f"jpeg_q{q}"] = float((probs.argmax(1) == y).mean())
        results[path.parent.name] = row
        print(path.parent.name, " ".join(f"{k}={v:.2%}" for k, v in row.items()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
