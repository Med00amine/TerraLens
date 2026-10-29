"""Figures for the README that are not produced by `terralens evaluate`.

- eurosat_classes.png: example tiles for each class
- gradcam_gallery.png: Grad-CAM for one random test tile per class
- mosaic_mapping.png: the mapping pipeline scored on a mosaic of test tiles
  with a known layout (see make_samples.py)

    python scripts/make_figures.py --model models/terralens.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

from terralens import viz  # noqa: E402
from terralens.constants import color_for, label_for  # noqa: E402
from terralens.data import load_dataset, stratified_split  # noqa: E402
from terralens.mapping import map_scene  # noqa: E402
from terralens.predictor import Predictor  # noqa: E402

SCENES = Path("src/terralens/api/static/samples/scenes")


def gradcam_gallery(predictor: Predictor, dataset, test_idx, out: Path) -> None:
    targets = np.asarray(dataset.targets)
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, 10, figsize=(20, 5.0))
    for c, name in enumerate(dataset.classes):
        i = int(rng.choice([j for j in test_idx if targets[j] == c]))
        tile = dataset[i][0]
        exp = predictor.explain(tile, alpha=0.5, size=256)
        axes[0, c].imshow(tile.resize((256, 256), Image.Resampling.NEAREST))
        axes[1, c].imshow(exp.overlay)
        mark = "✓" if exp.explained.class_name == name else "✗"
        axes[0, c].set_title(label_for(name), fontsize=10, fontweight="bold")
        axes[1, c].text(
            0.5,
            -0.04,
            f"{mark} {exp.explained.label}\n{exp.explained.probability:.0%}",
            transform=axes[1, c].transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Grad-CAM: where the model looks (one random test tile per class)", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mosaic_mapping(predictor: Predictor, out: Path) -> dict:
    truth = json.loads(Path("docs/mosaic_ground_truth.json").read_text())
    layout = np.array(truth["layout"])
    image = Image.open(SCENES / "test_tile_mosaic.png").convert("RGB")
    result = map_scene(predictor, image, tile_px=truth["tile_px"])
    accuracy = float((result.classes == layout).mean())

    palette = np.array(
        [[int(color_for(n)[i : i + 2], 16) for i in (1, 3, 5)] for n in truth["classes"]]
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.9))
    panels = [
        (np.asarray(image), "Mosaic of 216 held-out test tiles"),
        (palette[layout].astype(np.uint8), "Ground truth"),
        (
            palette[result.classes].astype(np.uint8),
            f"TerraLens map: {accuracy:.1%} of tiles correct",
        ),
    ]
    for ax, (img, title) in zip(axes, panels, strict=True):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    handles = [Patch(color=color_for(n), label=label_for(n)) for n in truth["classes"]]
    fig.legend(handles=handles, loc="lower center", ncol=10, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"tiles": int(layout.size), "accuracy": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/terralens.pt"))
    parser.add_argument("--out", type=Path, default=Path("docs/images"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    predictor = Predictor.from_path(args.model)
    dataset = load_dataset("eurosat", "data", download=True)
    cfg = predictor.checkpoint["config"]
    split = stratified_split(
        dataset.targets, cfg["val_fraction"], cfg["test_fraction"], cfg["seed"]
    )

    viz.plot_class_examples(dataset, dataset.classes, args.out / "eurosat_classes.png")
    gradcam_gallery(predictor, dataset, split.test, args.out / "gradcam_gallery.png")
    mosaic = mosaic_mapping(predictor, args.out / "mosaic_mapping.png")
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    Path("docs/results/mosaic_mapping.json").write_text(json.dumps(mosaic, indent=2))
    print(f"Mosaic mapping accuracy: {mosaic['accuracy']:.1%} of {mosaic['tiles']} tiles")
    print(f"Figures saved to {args.out}/")


if __name__ == "__main__":
    main()
