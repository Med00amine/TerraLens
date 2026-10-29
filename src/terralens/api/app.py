"""TerraLens REST API.

uvicorn terralens.api.app:app          # or: terralens serve
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from terralens import __version__
from terralens.api.schemas import (
    ExplainOut,
    HealthOut,
    MapOut,
    ModelInfoOut,
    PredictOut,
    SampleOut,
)
from terralens.mapping import map_scene
from terralens.predictor import Predictor, default_model_path

logger = logging.getLogger("terralens.api")

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = int(float(os.environ.get("TERRALENS_MAX_UPLOAD_MB", "20")) * 2**20)
MAX_SCENE_PIXELS = int(os.environ.get("TERRALENS_MAX_SCENE_PIXELS", str(4096 * 4096)))
MAX_RETURNED_SIDE = 1600  # images in responses are downscaled to keep payloads small

DESCRIPTION = """
Land-cover classification and mapping for Sentinel-2 RGB imagery (10 m/pixel).

* **/api/predict**: classify a single tile (about 640 m × 640 m)
* **/api/explain**: Grad-CAM heatmap showing what drove the prediction
* **/api/map**: land-cover map and area statistics for a large scene
"""


def _data_url(image: Image.Image, fmt: str = "PNG") -> str:
    image = image.convert("RGB")
    if max(image.size) > MAX_RETURNED_SIDE:
        image.thumbnail((MAX_RETURNED_SIDE, MAX_RETURNED_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format=fmt, **({"quality": 90} if fmt == "JPEG" else {}))
    mime = "jpeg" if fmt == "JPEG" else "png"
    return f"data:image/{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


async def _read_image(file: UploadFile) -> Image.Image:
    max_pixels = MAX_SCENE_PIXELS
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(415, f"Expected an image upload, got {file.content_type}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_BYTES // 2**20} MB")
    try:
        image = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError as exc:
        raise HTTPException(413, "Image dimensions are too large") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(415, "Could not decode the image") from exc
    if image.width * image.height > max_pixels:
        raise HTTPException(
            413, f"Image has {image.width}x{image.height} px; limit is {max_pixels:,} pixels"
        )
    try:
        image.load()  # header parsed fine, now decode (catches truncated files)
    except OSError as exc:
        raise HTTPException(415, "Could not decode the image") from exc
    return image.convert("RGB")


def create_app(predictor: Predictor | None = None, model_path: str | Path | None = None) -> FastAPI:
    """Build the app; pass a `predictor` directly (tests) or load one at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.predictor is None:
            path = Path(model_path) if model_path else default_model_path()
            try:
                app.state.predictor = Predictor.from_path(path)
                logger.info("Loaded %s from %s", app.state.predictor.info()["architecture"], path)
            except FileNotFoundError:
                logger.error(
                    "No model at %s: run `terralens download-model` or set TERRALENS_MODEL", path
                )
        if app.state.predictor is not None:
            app.state.predictor.warmup()
        yield

    app = FastAPI(
        title="TerraLens API", version=__version__, description=DESCRIPTION, lifespan=lifespan
    )
    app.state.predictor = predictor

    @app.middleware("http")
    async def timing(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
        if request.url.path.startswith("/api/"):
            logger.info(
                "%s %s %d %.1fms", request.method, request.url.path, response.status_code, elapsed
            )
        return response

    def loaded(request: Request) -> Predictor:
        p = request.app.state.predictor
        if p is None:
            raise HTTPException(503, "Model not loaded; run `terralens download-model`")
        return p

    @app.get("/api/health", response_model=HealthOut, tags=["service"])
    def health(request: Request):
        p = request.app.state.predictor
        return HealthOut(
            status="ok" if p else "degraded",
            model_loaded=p is not None,
            version=__version__,
            device=str(p.device) if p else None,
        )

    @app.get("/api/model", response_model=ModelInfoOut, tags=["service"])
    def model_info(request: Request):
        return loaded(request).info()

    @app.post("/api/predict", response_model=PredictOut, tags=["inference"])
    async def predict(
        request: Request,
        file: Annotated[UploadFile, File(description="A satellite tile (64x64 px at 10 m/px)")],
        top_k: Annotated[int, Query(ge=1, le=50)] = 5,
    ):
        p = loaded(request)
        image = await _read_image(file)
        started = time.perf_counter()
        preds = await run_in_threadpool(p.predict, image, top_k)
        return {
            "top": preds[0].to_dict(),
            "predictions": [x.to_dict() for x in preds],
            "image_size": image.size,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }

    @app.post("/api/explain", response_model=ExplainOut, tags=["inference"])
    async def explain(
        request: Request,
        file: Annotated[UploadFile, File(description="A satellite tile")],
        class_name: Annotated[str | None, Query(description="Default: the top class")] = None,
        alpha: Annotated[float, Query(ge=0, le=1, description="Heatmap opacity")] = 0.5,
    ):
        p = loaded(request)
        image = await _read_image(file)
        if class_name is not None and class_name not in p.class_names:
            raise HTTPException(422, f"Unknown class {class_name!r}")
        started = time.perf_counter()
        exp = await run_in_threadpool(p.explain, image, class_name, alpha)
        return {
            "explained": exp.explained.to_dict(),
            "predictions": [x.to_dict() for x in exp.predictions],
            "overlay": _data_url(exp.overlay),
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }

    @app.post("/api/map", response_model=MapOut, tags=["inference"])
    async def map_endpoint(
        request: Request,
        file: Annotated[UploadFile, File(description="A Sentinel-2 RGB scene at ~10 m/px")],
        tile_px: Annotated[int, Query(ge=16, le=512, description="Tile covering ~640 m")] = 64,
        stride_px: Annotated[
            int | None, Query(ge=8, le=512, description="tile/2: smoother")
        ] = None,
        alpha: Annotated[float, Query(ge=0, le=1, description="Overlay opacity")] = 0.45,
    ):
        p = loaded(request)
        image = await _read_image(file)
        started = time.perf_counter()
        try:
            result = await run_in_threadpool(map_scene, p, image, tile_px, stride_px)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "summary": result.summary(),
            "classes": result.stats(),
            "map": _data_url(result.render()),
            "overlay": _data_url(result.overlay(image, alpha), "JPEG"),
            "confidence": _data_url(result.confidence_image()),
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }

    @app.get("/api/samples", response_model=list[SampleOut], tags=["service"])
    def samples():
        found = []
        for kind in ("tiles", "scenes"):
            folder = STATIC_DIR / "samples" / kind
            for path in sorted(folder.glob("*")) if folder.is_dir() else []:
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    url = f"/static/samples/{kind}/{path.name}"
                    found.append(SampleOut(name=path.stem, kind=kind[:-1], url=url))
        return found

    if (STATIC_DIR / "index.html").is_file():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(STATIC_DIR / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        def index():
            return RedirectResponse("/docs")

    return app


app = create_app()


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the TerraLens API and web interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=Path, default=None, help="checkpoint path")
    args = parser.parse_args(argv)
    if args.model is not None:
        os.environ["TERRALENS_MODEL"] = str(args.model)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    uvicorn.run("terralens.api.app:app", host=args.host, port=args.port)
