"""Regenerate the README screenshots by driving the web UI in a real browser.

pip install playwright            # uses an installed Edge/Chrome, or run
                                  # `playwright install chromium` first
terralens serve --port 8899 &
python scripts/take_screenshots.py --url http://127.0.0.1:8899
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright


def wait_for_server(url: str, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise SystemExit(f"Server at {url} did not become healthy")


def launch(p):
    for channel in ("msedge", "chrome", None):
        try:
            return p.chromium.launch(channel=channel) if channel else p.chromium.launch()
        except Exception:  # browser channel not installed; try the next one
            continue
    raise SystemExit("No Chromium-based browser found; run `playwright install chromium`")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8899")
    parser.add_argument("--out", type=Path, default=Path("docs/images"))
    parser.add_argument("--tile", default="River", help="sample tile to classify")
    parser.add_argument("--scene", default="lake_constance", help="sample scene to map")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wait_for_server(args.url)

    with sync_playwright() as p:
        browser = launch(p)
        for scheme in ("light", "dark"):
            page = browser.new_page(
                viewport={"width": 1440, "height": 960}, device_scale_factor=1, color_scheme=scheme
            )
            suffix = "" if scheme == "light" else "_dark"

            page.goto(args.url)
            page.locator("#tile-samples .sample").first.wait_for()
            # Warm-up request so the timing chip shows steady-state latency.
            page.locator("#tile-samples .sample").first.click()
            page.locator("#tile-result .verdict").wait_for()
            page.locator(f"#tile-samples .sample[title$='{args.tile}']").first.click()
            page.wait_for_timeout(300)
            page.locator("#tile-result .verdict").wait_for()
            page.wait_for_timeout(900)  # let the probability bars animate
            page.screenshot(path=args.out / f"ui_classify{suffix}.png")

            if scheme == "light":
                page.locator("#tab-btn-map").click()
                page.locator("#scene-samples .sample").first.wait_for()
                page.locator(f"#scene-samples .sample img[src*='{args.scene}']").click()
                page.locator("#map-result table").wait_for(timeout=120_000)
                page.wait_for_timeout(500)
                page.screenshot(path=args.out / "ui_map.png", full_page=True)
                page.locator("#map-result [data-view='map']").click()
                page.wait_for_timeout(300)
                page.locator("#map-result .map-viewer").screenshot(
                    path=args.out / "ui_map_layer.png"
                )

                page.goto(f"{args.url}/docs")
                page.locator(".opblock").first.wait_for()
                page.screenshot(path=args.out / "api_docs.png")
            page.close()
        browser.close()
    print(f"Screenshots saved to {args.out}/")


if __name__ == "__main__":
    main()
