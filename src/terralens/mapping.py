"""Land-cover maps for scenes larger than a single tile.

The scene is cut into tiles of the size the model was trained on (64 px, i.e.
640 m at Sentinel-2's 10 m resolution), every tile is classified in batches, and
the probabilities are assembled into a map. With a stride smaller than the tile,
overlapping predictions are averaged onto a finer grid for smoother boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from terralens.constants import EUROSAT_TILE_PX, SENTINEL2_GSD_M, color_for, label_for
from terralens.explain import apply_colormap


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))


@dataclass
class LandCoverMap:
    probs: np.ndarray  # (rows, cols, C) class probabilities per map cell
    class_names: list[str]
    cell_px: int  # side of one map cell in source pixels (the stride)
    tile_px: int
    image_size: tuple[int, int]  # (width, height) of the source scene
    n_tiles: int
    seconds: float

    @property
    def classes(self) -> np.ndarray:
        return self.probs.argmax(axis=-1)

    @property
    def confidence(self) -> np.ndarray:
        return self.probs.max(axis=-1)

    def _cell_pixel_areas(self) -> np.ndarray:
        """Number of source pixels inside each cell (edge cells may be partial)."""
        width, height = self.image_size
        rows, cols = self.classes.shape
        h = np.minimum(self.cell_px, height - self.cell_px * np.arange(rows))
        w = np.minimum(self.cell_px, width - self.cell_px * np.arange(cols))
        return np.outer(h, w).astype(float)

    def stats(self, gsd_m: float = SENTINEL2_GSD_M) -> list[dict]:
        """Area share of each class present in the scene, largest first."""
        areas = self._cell_pixel_areas()
        total = areas.sum()
        classes, confidence = self.classes, self.confidence
        rows = []
        for idx, name in enumerate(self.class_names):
            mask = classes == idx
            if not mask.any():
                continue
            pixels = areas[mask].sum()
            rows.append(
                {
                    "class_name": name,
                    "label": label_for(name),
                    "color": color_for(name),
                    "fraction": float(pixels / total),
                    "area_km2": float(pixels * gsd_m**2 / 1e6),
                    "mean_confidence": float(np.average(confidence[mask], weights=areas[mask])),
                }
            )
        return sorted(rows, key=lambda r: -r["fraction"])

    def summary(self, gsd_m: float = SENTINEL2_GSD_M) -> dict:
        areas = self._cell_pixel_areas()
        confidence = self.confidence
        width, height = self.image_size
        return {
            "image_size": [width, height],
            "area_km2": float(width * height * gsd_m**2 / 1e6),
            "tiles": self.n_tiles,
            "grid": list(self.classes.shape),
            "mean_confidence": float(np.average(confidence, weights=areas)),
            "uncertain_fraction": float(areas[confidence < 0.5].sum() / areas.sum()),
            "seconds": self.seconds,
        }

    def _upscale(self, cells: np.ndarray) -> Image.Image:
        rows, cols = cells.shape[:2]
        image = Image.fromarray(cells).resize(
            (cols * self.cell_px, rows * self.cell_px), Image.Resampling.NEAREST
        )
        return image.crop((0, 0, *self.image_size))

    def render(self) -> Image.Image:
        """Colour-coded land-cover map at the source resolution."""
        palette = np.array([_hex_to_rgb(color_for(n)) for n in self.class_names], dtype=np.uint8)
        return self._upscale(palette[self.classes])

    def overlay(self, image: Image.Image, alpha: float = 0.45) -> Image.Image:
        return Image.blend(image.convert("RGB"), self.render(), alpha)

    def confidence_image(self) -> Image.Image:
        """Model confidence per cell: dark blue = uncertain, red = confident."""
        floor = 1 / len(self.class_names)
        scaled = (self.confidence - floor) / (1 - floor)
        return self._upscale(apply_colormap(scaled))


def map_scene(
    predictor,
    image: Image.Image,
    tile_px: int = EUROSAT_TILE_PX,
    stride_px: int | None = None,
    batch_size: int = 256,
    max_tiles: int = 20_000,
) -> LandCoverMap:
    """Classify every tile of `image` and assemble a `LandCoverMap`.

    `tile_px` should cover the same ground area as a training tile (640 m), so
    keep 64 for 10 m imagery and scale it for other resolutions (e.g. 128 at 5 m).
    """
    stride = stride_px or tile_px
    if tile_px % stride:
        raise ValueError("tile_px must be a multiple of stride_px")

    rgb = np.array(image.convert("RGB"))  # writable copy for torch.from_numpy
    height, width = rgb.shape[:2]
    if width < tile_px or height < tile_px:
        raise ValueError(f"Image must be at least {tile_px}x{tile_px} pixels")

    rows = math.ceil((height - tile_px) / stride) + 1
    cols = math.ceil((width - tile_px) / stride) + 1
    n_tiles = rows * cols
    if n_tiles > max_tiles:
        raise ValueError(f"Scene needs {n_tiles} tiles (limit {max_tiles}); increase the stride")

    # Mirror-pad the bottom/right edge so the last row/column of tiles is complete.
    pad_h = (rows - 1) * stride + tile_px - height
    pad_w = (cols - 1) * stride + tile_px - width
    if pad_h or pad_w:
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    started = time.perf_counter()
    grid = (
        torch.from_numpy(rgb).permute(2, 0, 1).unfold(1, tile_px, stride).unfold(2, tile_px, stride)
    )
    flat = torch.arange(n_tiles)
    probs = []
    for start in range(0, n_tiles, batch_size):
        idx = flat[start : start + batch_size]
        tiles = grid[:, idx // cols, idx % cols].permute(1, 0, 2, 3)  # (B, 3, t, t) uint8
        probs.append(predictor.predict_proba(predictor.preprocess_tiles(tiles)))
    tile_probs = np.concatenate(probs).reshape(rows, cols, -1)

    # Average overlapping tile predictions onto stride-sized cells.
    k = tile_px // stride
    cells = np.zeros((rows + k - 1, cols + k - 1, tile_probs.shape[-1]))
    counts = np.zeros((rows + k - 1, cols + k - 1, 1))
    for dy in range(k):
        for dx in range(k):
            cells[dy : dy + rows, dx : dx + cols] += tile_probs
            counts[dy : dy + rows, dx : dx + cols] += 1
    cells /= counts
    cells = cells[: math.ceil(height / stride), : math.ceil(width / stride)]

    return LandCoverMap(
        probs=cells,
        class_names=list(predictor.class_names),
        cell_px=stride,
        tile_px=tile_px,
        image_size=(width, height),
        n_tiles=n_tiles,
        seconds=time.perf_counter() - started,
    )


def main(argv: list[str] | None = None) -> None:
    from terralens.predictor import Predictor, default_model_path

    parser = argparse.ArgumentParser(description="Create a land-cover map of a satellite scene.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=None, help="checkpoint path")
    parser.add_argument("--out-dir", type=Path, default=Path("maps"))
    parser.add_argument("--tile", type=int, default=EUROSAT_TILE_PX, help="tile size in pixels")
    parser.add_argument("--stride", type=int, default=None, help="e.g. tile/2 for smoother maps")
    parser.add_argument("--gsd", type=float, default=SENTINEL2_GSD_M, help="metres per pixel")
    parser.add_argument("--alpha", type=float, default=0.45, help="overlay opacity")
    args = parser.parse_args(argv)

    predictor = Predictor.from_path(args.model or default_model_path())
    image = Image.open(args.image)
    result = map_scene(predictor, image, tile_px=args.tile, stride_px=args.stride)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    result.render().save(args.out_dir / f"{stem}_map.png")
    result.overlay(image, args.alpha).save(args.out_dir / f"{stem}_overlay.png")
    result.confidence_image().save(args.out_dir / f"{stem}_confidence.png")
    stats = {"summary": result.summary(args.gsd), "classes": result.stats(args.gsd)}
    (args.out_dir / f"{stem}_stats.json").write_text(json.dumps(stats, indent=2))

    s = stats["summary"]
    print(f"{s['tiles']} tiles, {s['area_km2']:.1f} km2, {s['seconds']:.2f}s")
    for row in stats["classes"]:
        print(f"  {row['label']:<24} {row['fraction']:6.1%}  {row['area_km2']:8.2f} km2")
    print(f"Saved map, overlay, confidence and stats to {args.out_dir}/")
