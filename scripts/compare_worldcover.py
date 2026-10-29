"""External validation: compare a TerraLens map of a real scene with ESA WorldCover.

EuroSAT accuracy only measures performance on EuroSAT-style tiles. To check how
the model generalises to real, atmospherically corrected Sentinel-2 scenes, this
script maps a scene fetched with `fetch_sentinel2_scene.py` and compares it, tile
by tile, with ESA WorldCover 2021 (10 m, CC BY 4.0) after mapping both legends to
five shared classes. WorldCover is itself a model output (~77% overall accuracy
reported by ESA), so this is an agreement score, not ground truth.

    pip install rasterio
    python scripts/compare_worldcover.py --scene lake_constance --model models/terralens.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
COARSE = ["Water", "Built-up", "Tree cover", "Cropland", "Grassland"]
COARSE_COLORS = ["#1565c0", "#e53935", "#1b5e20", "#e6c229", "#9ccc65"]
EUROSAT_TO_COARSE = {
    "SeaLake": "Water",
    "River": "Water",
    "Residential": "Built-up",
    "Industrial": "Built-up",
    "Highway": "Built-up",
    "Forest": "Tree cover",
    "AnnualCrop": "Cropland",
    # WorldCover assigns perennial woody crops (orchards, vineyards) to tree cover.
    "PermanentCrop": "Tree cover",
    "Pasture": "Grassland",
    "HerbaceousVegetation": "Grassland",
}
WORLDCOVER_TO_COARSE = {
    80: "Water",
    50: "Built-up",
    10: "Tree cover",
    40: "Cropland",
    30: "Grassland",
    20: "Grassland",
    90: "Grassland",
}


def worldcover_tile(lon: float, lat: float) -> str:
    """WorldCover is split into 3x3 degree tiles named after their south-west corner."""
    lat0, lon0 = 3 * math.floor(lat / 3), 3 * math.floor(lon / 3)
    return f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"


def read_worldcover(meta: dict) -> np.ndarray:
    """WorldCover classes resampled (nearest) onto the scene's 10 m UTM grid."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    width, height = meta["size_px"]
    dst = np.zeros((height, width), dtype=np.uint8)
    dst_transform = from_bounds(*meta["bounds"], width, height)
    url = WORLDCOVER_URL.format(tile=worldcover_tile(*meta["center_lon_lat"]))
    with rasterio.open(url) as src:
        reproject(
            rasterio.band(src, 1),
            dst,
            dst_transform=dst_transform,
            dst_crs=meta["crs"],
            resampling=Resampling.nearest,
        )
    return dst


def majority_per_tile(codes: np.ndarray, tile: int) -> tuple[np.ndarray, np.ndarray]:
    """Most frequent coarse class per tile (-1 if none dominates) and its share of the tile."""
    lut = np.full(256, -1, dtype=int)
    for code, name in WORLDCOVER_TO_COARSE.items():
        lut[code] = COARSE.index(name)
    coarse = lut[codes]
    rows, cols = codes.shape[0] // tile, codes.shape[1] // tile
    out = np.full((rows, cols), -1)
    purity = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            block = coarse[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile]
            counts = np.bincount(block[block >= 0], minlength=len(COARSE))
            if counts.sum() > 0.5 * block.size:
                out[r, c] = counts.argmax()
                purity[r, c] = counts.max() / block.size
    return out, purity


def agreement(reference: np.ndarray, predicted: np.ndarray, mask: np.ndarray | None = None) -> dict:
    valid = reference >= 0 if mask is None else (reference >= 0) & mask
    ref, pred = reference[valid], predicted[valid]
    cm = np.zeros((len(COARSE), len(COARSE)), dtype=int)
    np.add.at(cm, (ref, pred), 1)
    po = np.trace(cm) / cm.sum()
    pe = (cm.sum(0) * cm.sum(1)).sum() / cm.sum() ** 2
    recall = {COARSE[i]: float(cm[i, i] / cm[i].sum()) for i in range(len(COARSE)) if cm[i].sum()}
    return {
        "tiles_compared": int(valid.sum()),
        "agreement": float(po),
        "cohen_kappa": float((po - pe) / (1 - pe)),
        "recall_vs_worldcover": recall,
        "confusion_matrix": cm.tolist(),
        "labels": COARSE,
    }


def coarse_rgb(grid: np.ndarray) -> np.ndarray:
    palette = np.array([[int(c[i : i + 2], 16) for i in (1, 3, 5)] for c in COARSE_COLORS])
    rgb = np.full((*grid.shape, 3), 255, dtype=np.uint8)
    rgb[grid >= 0] = palette[grid[grid >= 0]]
    return rgb


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image

    from terralens.mapping import map_scene
    from terralens.predictor import Predictor

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", default="lake_constance")
    parser.add_argument("--model", type=Path, default=Path("models/terralens.pt"))
    parser.add_argument(
        "--scene-dir", type=Path, default=Path("src/terralens/api/static/samples/scenes")
    )
    parser.add_argument("--out", type=Path, default=Path("docs/results"))
    parser.add_argument(
        "--min-purity",
        type=float,
        default=0.8,
        help="also score tiles where one WorldCover class covers at least this share",
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("docs/images/worldcover_comparison.png")
    )
    args = parser.parse_args()

    meta = json.loads(Path(f"docs/{args.scene}.json").read_text())
    path = next(
        p for ext in ("png", "jpg") if (p := args.scene_dir / f"{args.scene}.{ext}").is_file()
    )
    image = Image.open(path).convert("RGB")
    predictor = Predictor.from_path(args.model)
    result = map_scene(predictor, image)  # stride = tile, so cells align with tiles

    to_coarse = np.array([COARSE.index(EUROSAT_TO_COARSE[c]) for c in predictor.class_names])
    predicted = to_coarse[result.classes]
    reference, purity = majority_per_tile(read_worldcover(meta), result.tile_px)
    predicted = predicted[: reference.shape[0], : reference.shape[1]]

    report = agreement(reference, predicted)
    report["homogeneous"] = agreement(reference, predicted, purity >= args.min_purity)
    report["homogeneous"]["min_purity"] = args.min_purity
    report.update({"scene": meta["scene"], "model": predictor.info()["architecture"]})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"worldcover_{args.scene}.json").write_text(json.dumps(report, indent=2))
    print(
        f"{report['tiles_compared']} tiles: agreement {report['agreement']:.1%}, "
        f"kappa {report['cohen_kappa']:.3f}"
    )
    for name, value in report["recall_vs_worldcover"].items():
        print(f"  {name:<11} {value:.1%}")
    h = report["homogeneous"]
    print(
        f"Homogeneous tiles (>= {args.min_purity:.0%} one class): {h['tiles_compared']} tiles, "
        f"agreement {h['agreement']:.1%}, kappa {h['cohen_kappa']:.3f}"
    )
    for name, value in h["recall_vs_worldcover"].items():
        print(f"  {name:<11} {value:.1%}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    panels = [
        (np.asarray(image), f"Sentinel-2 L2A, {meta['date'][:10]}"),
        (coarse_rgb(reference), "ESA WorldCover 2021 (reference)"),
        (coarse_rgb(predicted), f"TerraLens: {report['agreement']:.0%} agreement"),
    ]
    for ax, (img, title) in zip(axes, panels, strict=True):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    handles = [Patch(color=c, label=n) for n, c in zip(COARSE, COARSE_COLORS, strict=True)]
    handles.append(Patch(facecolor="white", edgecolor="#999", label="No dominant class"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.figure}")


if __name__ == "__main__":
    main()
