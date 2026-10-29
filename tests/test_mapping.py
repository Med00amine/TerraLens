import numpy as np
import pytest
import torch
from PIL import Image

from terralens.mapping import map_scene

RED, GREEN, BLUE = (220, 20, 20), (20, 220, 20), (20, 20, 220)


class ColourPredictor:
    """Stand-in model: the dominant colour channel decides the class."""

    class_names = ["Forest", "River", "SeaLake"]

    def preprocess_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        return tiles.float() / 255

    def predict_proba(self, batch: torch.Tensor) -> np.ndarray:
        means = batch.mean(dim=(2, 3))
        return torch.softmax(20 * means, dim=1).numpy()


def stripes(width_per_stripe=64, height=128, colours=(RED, GREEN, BLUE)):
    img = Image.new("RGB", (width_per_stripe * len(colours), height))
    for i, colour in enumerate(colours):
        img.paste(colour, (i * width_per_stripe, 0, (i + 1) * width_per_stripe, height))
    return img


def test_map_matches_known_layout():
    result = map_scene(ColourPredictor(), stripes())
    assert result.classes.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert result.n_tiles == 6
    assert result.render().size == (192, 128)


def test_stats_areas_and_fractions():
    result = map_scene(ColourPredictor(), stripes())
    stats = result.stats(gsd_m=10)
    assert {s["class_name"] for s in stats} == {"Forest", "River", "SeaLake"}
    assert sum(s["fraction"] for s in stats) == pytest.approx(1.0)
    # 192 x 128 px at 10 m/px = 2.4576 km2, split evenly between three stripes.
    assert sum(s["area_km2"] for s in stats) == pytest.approx(2.4576)
    assert all(s["area_km2"] == pytest.approx(0.8192) for s in stats)
    summary = result.summary()
    assert summary["area_km2"] == pytest.approx(2.4576)
    assert summary["uncertain_fraction"] == 0.0


def test_partial_edge_tiles_are_weighted_by_area():
    # 100 x 70 px: the second row/column of tiles is mostly mirror padding.
    img = Image.new("RGB", (100, 70), RED)
    result = map_scene(ColourPredictor(), img)
    assert result.classes.shape == (2, 2)
    assert result.render().size == (100, 70)
    assert result.stats(gsd_m=10)[0]["area_km2"] == pytest.approx(0.7)


def test_overlapping_tiles_give_a_finer_grid():
    result = map_scene(ColourPredictor(), stripes(), stride_px=32)
    assert result.classes.shape == (4, 6)
    assert result.cell_px == 32
    assert result.n_tiles == 3 * 5
    assert result.classes[:, 0].tolist() == [0] * 4
    assert result.classes[:, -1].tolist() == [2] * 4


def test_overlay_and_confidence_images():
    img = stripes()
    result = map_scene(ColourPredictor(), img)
    assert result.overlay(img).size == img.size
    assert result.confidence_image().size == img.size


@pytest.mark.parametrize(
    "kwargs, size, message",
    [
        ({}, (40, 40), "at least 64x64"),
        ({"stride_px": 48}, (128, 128), "multiple"),
        ({"max_tiles": 3}, (192, 128), "tiles"),
    ],
)
def test_invalid_requests(kwargs, size, message):
    with pytest.raises(ValueError, match=message):
        map_scene(ColourPredictor(), Image.new("RGB", size), **kwargs)


def test_single_tile_scene_matches_predictor(predictor):
    rng = np.random.default_rng(5)
    tile = Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
    result = map_scene(predictor, tile)
    expected = {p.class_name: p.probability for p in predictor.predict(tile, top_k=None)}
    got = dict(zip(predictor.class_names, result.probs[0, 0], strict=True))
    for name in expected:
        assert got[name] == pytest.approx(expected[name], abs=0.02)
