import numpy as np
import pytest
from PIL import Image

from terralens.predictor import Predictor
from terralens.train import TrainConfig, train

CLASSES = ("Forest", "River", "SeaLake")


@pytest.fixture(scope="session")
def image_folder(tmp_path_factory):
    """A tiny ImageFolder dataset: 3 classes x 20 random 64x64 RGB tiles."""
    root = tmp_path_factory.mktemp("dataset")
    rng = np.random.default_rng(0)
    for cls in CLASSES:
        (root / cls).mkdir()
        for i in range(20):
            pixels = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / cls / f"{i}.png")
    return root


@pytest.fixture(scope="session")
def trained_checkpoint(image_folder, tmp_path_factory):
    """A real (if useless) checkpoint: 1 epoch of ResNet18 on the tiny dataset."""
    out = tmp_path_factory.mktemp("run")
    train(
        TrainConfig(
            dataset="folder",
            data_root=str(image_folder),
            download=False,
            pretrained=False,
            img_size=32,
            epochs=1,
            batch_size=8,
            num_workers=0,
            amp=False,
            output_dir=str(out),
        )
    )
    return out / "model.pt"


@pytest.fixture(scope="session")
def predictor(trained_checkpoint):
    return Predictor.from_path(trained_checkpoint, device="cpu")
