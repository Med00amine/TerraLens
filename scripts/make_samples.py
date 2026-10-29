"""Build the web UI samples from the EuroSAT *test* split (never seen in training).

- one tile per class, picked at random (no cherry-picking)
- a synthetic "landscape" mosaic of test tiles with a known ground-truth map,
  used to measure mapping accuracy (see `scripts/evaluate_mosaic.py`)

    python scripts/make_samples.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from terralens.data import load_dataset, stratified_split

STATIC = Path("src/terralens/api/static/samples")
TILE = 64
ROWS, COLS = 12, 18  # 768 x 1152 px = 7.7 km x 11.5 km at 10 m/px


def landscape_layout(rng: np.random.Generator, classes: list[str]) -> np.ndarray:
    """A plausible class layout: Voronoi regions plus a coastline, a river and a highway."""
    idx = {name: i for i, name in enumerate(classes)}
    land = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Pasture", "PermanentCrop"]
    seeds = rng.uniform([0, 0], [ROWS, COLS], size=(16, 2))
    seed_classes = [idx[c] for c in rng.choice(land, size=len(seeds))]
    # A town: two residential seeds and one industrial seed near the centre.
    seeds[:3] = [(ROWS * 0.45, COLS * 0.55), (ROWS * 0.6, COLS * 0.62), (ROWS * 0.4, COLS * 0.72)]
    seed_classes[:3] = [idx["Residential"], idx["Residential"], idx["Industrial"]]

    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    dist = (yy[..., None] - seeds[:, 0]) ** 2 + (xx[..., None] - seeds[:, 1]) ** 2
    layout = np.array(seed_classes)[dist.argmin(-1)]

    coast = (2 + np.round(np.sin(np.arange(ROWS) / 1.7))).astype(int)  # 1-3 cells wide
    for r in range(ROWS):
        layout[r, : coast[r]] = idx["SeaLake"]
    river = (np.sin(np.arange(COLS) / 2.5) * 1.5 + ROWS * 0.25).round().astype(int)
    for col in range(3, COLS):
        layout[river[col], col] = idx["River"]
    highway_col = int(COLS * 0.4)
    layout[:, highway_col] = np.where(
        layout[:, highway_col] == idx["River"], idx["River"], idx["Highway"]
    )
    return layout


def main() -> None:
    dataset = load_dataset("eurosat", "data", download=True)
    split = stratified_split(dataset.targets, 0.15, 0.15, seed=42)  # same split as training
    targets = np.asarray(dataset.targets)
    test = np.asarray(split.test)
    classes = list(dataset.classes)
    rng = np.random.default_rng(2026)

    tiles_dir = STATIC / "tiles"
    shutil.rmtree(tiles_dir, ignore_errors=True)
    tiles_dir.mkdir(parents=True)
    pools = {c: rng.permutation(test[targets[test] == c]).tolist() for c in range(len(classes))}
    for c, name in enumerate(classes):
        src = Path(dataset.samples[pools[c].pop()][0])
        shutil.copy(src, tiles_dir / f"{name}.jpg")

    layout = landscape_layout(rng, classes)
    mosaic = Image.new("RGB", (COLS * TILE, ROWS * TILE))
    for r in range(ROWS):
        for col in range(COLS):
            path = dataset.samples[pools[layout[r, col]].pop()][0]
            mosaic.paste(Image.open(path).convert("RGB"), (col * TILE, r * TILE))

    scenes_dir = STATIC / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    mosaic.save(scenes_dir / "test_tile_mosaic.png")  # lossless: JPEG costs ~2 points
    truth = {"tile_px": TILE, "classes": classes, "layout": layout.tolist()}
    Path("docs").mkdir(exist_ok=True)
    Path("docs/mosaic_ground_truth.json").write_text(json.dumps(truth))
    print(f"Saved {len(classes)} sample tiles and a {mosaic.size[0]}x{mosaic.size[1]} mosaic")


if __name__ == "__main__":
    main()
