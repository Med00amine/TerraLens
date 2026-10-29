import numpy as np
import pytest
from PIL import Image

CLASSES = ("Forest", "River", "SeaLake")


@pytest.fixture
def image_folder(tmp_path):
    """A tiny ImageFolder dataset: 3 classes x 20 random 64x64 RGB tiles."""
    rng = np.random.default_rng(0)
    for cls in CLASSES:
        (tmp_path / cls).mkdir()
        for i in range(20):
            pixels = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(tmp_path / cls / f"{i}.png")
    return tmp_path
