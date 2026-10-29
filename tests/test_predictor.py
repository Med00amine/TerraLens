import numpy as np
import pytest
import torch
from PIL import Image

from terralens.predictor import Prediction


@pytest.fixture
def tile():
    rng = np.random.default_rng(3)
    return Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))


def test_predict_ranks_all_classes(predictor, tile):
    preds = predictor.predict(tile, top_k=None)
    assert {p.class_name for p in preds} == {"Forest", "River", "SeaLake"}
    probs = [p.probability for p in preds]
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) == pytest.approx(1.0, abs=1e-5)
    assert isinstance(preds[0], Prediction)
    assert preds[0].color.startswith("#")


def test_predict_top_k(predictor, tile):
    assert len(predictor.predict(tile, top_k=2)) == 2


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_predict_accepts_any_image_mode(predictor, tile, mode):
    assert len(predictor.predict(tile.convert(mode))) == 3


def test_batched_preprocessing_matches_single_image(predictor, tile):
    single = predictor.predict_proba(predictor.preprocess(tile))
    tiles = torch.from_numpy(np.array(tile)).permute(2, 0, 1)[None]
    batched = predictor.predict_proba(predictor.preprocess_tiles(tiles))
    assert np.allclose(single, batched, atol=0.02)


def test_explain_top_class(predictor, tile):
    exp = predictor.explain(tile, size=128)
    assert exp.overlay.size == (128, 128)
    assert exp.heatmap.shape == (predictor.img_size, predictor.img_size)
    assert exp.explained == exp.predictions[0]


def test_explain_requested_class(predictor, tile):
    assert predictor.explain(tile, class_name="River").explained.class_name == "River"
    with pytest.raises(ValueError, match="Unknown class"):
        predictor.explain(tile, class_name="Glacier")


def test_info(predictor):
    info = predictor.info()
    assert info["architecture"] == "resnet18"
    assert info["img_size"] == 32
    assert [c["name"] for c in info["classes"]] == ["Forest", "River", "SeaLake"]
    assert "test_acc" in info["metrics"]
    assert info["device"] == "cpu"
