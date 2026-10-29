"""Download a true-colour Sentinel-2 crop to test TerraLens on real imagery.

Searches the public Earth Search STAC catalogue (Element 84, AWS Open Data) for
the clearest Sentinel-2 L2A scene over a point, then reads only the requested
window of the 10 m true-colour COG over HTTP. No account or API key is needed.

    pip install rasterio
    python scripts/fetch_sentinel2_scene.py --lon 9.48 --lat 47.68 --name lake_constance

Contains modified Copernicus Sentinel data (free and open, see the Sentinel data
legal notice).
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"


def find_scene(lon: float, lat: float, start: str, end: str, max_cloud: float) -> dict:
    query = {
        "collections": ["sentinel-2-l2a"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": max_cloud}, "s2:nodata_pixel_percentage": {"lt": 1}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "limit": 1,
    }
    request = urllib.request.Request(
        STAC_SEARCH, json.dumps(query).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as resp:
        features = json.load(resp)["features"]
    if not features:
        raise SystemExit("No matching scene; widen the date range or cloud limit")
    return features[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--width", type=int, default=1536, help="pixels (10 m each)")
    parser.add_argument("--height", type=int, default=1024, help="pixels (10 m each)")
    parser.add_argument("--start", default="2025-06-01")
    parser.add_argument("--end", default="2025-08-31")
    parser.add_argument("--max-cloud", type=float, default=3.0)
    parser.add_argument("--name", default="sentinel2_scene")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("src/terralens/api/static/samples/scenes")
    )
    args = parser.parse_args()

    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.warp import transform
    from rasterio.windows import Window

    item = find_scene(args.lon, args.lat, args.start, args.end, args.max_cloud)
    href = item["assets"]["visual"]["href"]
    print(
        f"Scene {item['id']} ({item['properties']['datetime'][:10]}), cloud cover "
        f"{item['properties']['eo:cloud_cover']:.3f}%"
    )

    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    with rasterio.open(href) as src:
        (x,), (y,) = transform("EPSG:4326", src.crs, [args.lon], [args.lat])
        row, col = src.index(x, y)
        window = Window(col - args.width // 2, row - args.height // 2, args.width, args.height)
        rgb = src.read([1, 2, 3], window=window)
        bounds = rasterio.windows.bounds(window, src.transform)
        crs = src.crs.to_string()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.jpg"
    Image.fromarray(np.moveaxis(rgb, 0, -1)).save(out, quality=92)
    meta = {
        "scene": item["id"],
        "date": item["properties"]["datetime"],
        "cloud_cover": item["properties"]["eo:cloud_cover"],
        "center_lon_lat": [args.lon, args.lat],
        "bounds": bounds,
        "crs": crs,
        "size_px": [args.width, args.height],
        "gsd_m": 10,
        "source": href,
        "attribution": "Contains modified Copernicus Sentinel data "
        + item["properties"]["datetime"][:4],
    }
    Path("docs").mkdir(exist_ok=True)
    Path(f"docs/{args.name}.json").write_text(json.dumps(meta, indent=2))
    print(
        f"Saved {out} ({args.width}x{args.height} px = "
        f"{args.width / 100:.1f} x {args.height / 100:.1f} km)"
    )


if __name__ == "__main__":
    main()
