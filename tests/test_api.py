import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import terralens.api.app as api
from terralens import __version__


def png_bytes(size=(64, 64), colour=(40, 110, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def upload(data: bytes, name="tile.png", mime="image/png"):
    return {"file": (name, data, mime)}


def decode_data_url(url: str) -> Image.Image:
    header, payload = url.split(",", 1)
    assert header.startswith("data:image/")
    return Image.open(io.BytesIO(base64.b64decode(payload)))


@pytest.fixture
def client(predictor):
    with TestClient(api.create_app(predictor=predictor)) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "model_loaded": True, "version": __version__, "device": "cpu"}


def test_model_info(client):
    body = client.get("/api/model").json()
    assert body["architecture"] == "resnet18"
    assert body["temperature"] > 0
    assert [c["name"] for c in body["classes"]] == ["Forest", "River", "SeaLake"]


def test_predict(client):
    resp = client.post("/api/predict", files=upload(png_bytes()))
    assert resp.status_code == 200
    body = resp.json()
    assert body["top"] == body["predictions"][0]
    assert sum(p["probability"] for p in body["predictions"]) == pytest.approx(1, abs=1e-4)
    assert body["image_size"] == [64, 64]
    assert "X-Process-Time-Ms" in resp.headers


def test_predict_top_k(client):
    body = client.post("/api/predict?top_k=1", files=upload(png_bytes())).json()
    assert len(body["predictions"]) == 1


def test_explain(client):
    body = client.post("/api/explain", files=upload(png_bytes())).json()
    assert body["explained"] == body["predictions"][0]
    assert decode_data_url(body["overlay"]).size == (256, 256)


def test_explain_requested_and_unknown_class(client):
    body = client.post("/api/explain?class_name=River", files=upload(png_bytes())).json()
    assert body["explained"]["class_name"] == "River"
    resp = client.post("/api/explain?class_name=Glacier", files=upload(png_bytes()))
    assert resp.status_code == 422


def test_map(client):
    resp = client.post("/api/map", files=upload(png_bytes((192, 128))))
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["tiles"] == 6
    assert body["summary"]["grid"] == [2, 3]
    assert sum(c["fraction"] for c in body["classes"]) == pytest.approx(1)
    assert decode_data_url(body["map"]).size == (192, 128)
    assert decode_data_url(body["overlay"]).format == "JPEG"


def test_map_with_overlap(client):
    body = client.post("/api/map?stride_px=32", files=upload(png_bytes((128, 128)))).json()
    assert body["summary"]["grid"] == [4, 4]


@pytest.mark.parametrize(
    "query, size",
    [("", (40, 40)), ("?stride_px=48", (128, 128))],
)
def test_map_invalid_requests(client, query, size):
    assert client.post(f"/api/map{query}", files=upload(png_bytes(size))).status_code == 422


def test_rejects_non_images(client):
    resp = client.post("/api/predict", files=upload(b"hello", "notes.txt", "text/plain"))
    assert resp.status_code == 415
    resp = client.post("/api/predict", files=upload(b"not really a png"))
    assert resp.status_code == 415


def test_rejects_truncated_image(client):
    assert client.post("/api/predict", files=upload(png_bytes()[:60])).status_code == 415


def test_rejects_oversized_uploads(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 100)
    assert client.post("/api/predict", files=upload(png_bytes())).status_code == 413


def test_rejects_too_many_pixels(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_SCENE_PIXELS", 1000)
    resp = client.post("/api/map", files=upload(png_bytes((64, 64))))
    assert resp.status_code == 413
    assert "limit" in resp.json()["detail"]


def test_missing_model_degrades_gracefully(tmp_path):
    app = api.create_app(model_path=tmp_path / "missing.pt")
    with TestClient(app) as client:
        assert client.get("/api/health").json()["model_loaded"] is False
        assert client.post("/api/predict", files=upload(png_bytes())).status_code == 503
        assert client.get("/api/model").status_code == 503


def test_app_loads_model_from_path(trained_checkpoint):
    with TestClient(api.create_app(model_path=trained_checkpoint)) as client:
        assert client.get("/api/health").json()["model_loaded"] is True


def test_openapi_documents_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/predict", "/api/explain", "/api/map", "/api/health"} <= set(paths)
