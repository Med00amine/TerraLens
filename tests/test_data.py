from collections import Counter

import numpy as np
import pytest
import torch
from PIL import Image

from terralens.data import (
    RandomRotate90,
    build_transforms,
    load_dataset,
    make_dataloaders,
    stratified_split,
)


def test_stratified_split_keeps_class_proportions():
    targets = [0] * 100 + [1] * 60 + [2] * 40
    split = stratified_split(targets, val_fraction=0.15, test_fraction=0.15, seed=0)

    for part, frac in ((split.val, 0.15), (split.test, 0.15)):
        counts = Counter(targets[i] for i in part)
        assert counts == {0: round(100 * frac), 1: round(60 * frac), 2: round(40 * frac)}


def test_stratified_split_is_disjoint_and_complete():
    targets = np.repeat(np.arange(5), 37)
    split = stratified_split(targets, seed=1)
    train, val, test = set(split.train), set(split.val), set(split.test)
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(range(len(targets)))


def test_stratified_split_is_deterministic():
    targets = np.repeat(np.arange(3), 50)
    assert stratified_split(targets, seed=7) == stratified_split(targets, seed=7)
    assert stratified_split(targets, seed=7) != stratified_split(targets, seed=8)


@pytest.mark.parametrize("val, test", [(-0.1, 0.1), (0.5, 0.5), (0.7, 0.4)])
def test_stratified_split_rejects_bad_fractions(val, test):
    with pytest.raises(ValueError):
        stratified_split([0, 1, 0, 1], val_fraction=val, test_fraction=test)


def test_random_rotate90_preserves_size_and_pixels():
    img = Image.fromarray(np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3))
    torch.manual_seed(0)
    for _ in range(8):
        out = RandomRotate90()(img)
        assert out.size == img.size
        assert sorted(np.asarray(out).ravel()) == sorted(np.asarray(img).ravel())


@pytest.mark.parametrize("train", [True, False])
def test_transforms_output_normalised_tensor(train):
    img = Image.new("RGB", (64, 64), (120, 140, 90))
    x = build_transforms(96, train=train)(img)
    assert x.shape == (3, 96, 96)
    assert x.dtype == torch.float32


def test_load_folder_dataset(image_folder):
    ds = load_dataset("folder", image_folder)
    assert ds.classes == ["Forest", "River", "SeaLake"]
    assert len(ds) == 60


def test_load_dataset_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset("folder", tmp_path / "missing")
    with pytest.raises(ValueError):
        load_dataset("imagenet", tmp_path)


def test_make_dataloaders_batches(image_folder):
    ds = load_dataset("folder", image_folder)
    bundle = make_dataloaders(ds, img_size=32, batch_size=8, num_workers=0)

    assert bundle.class_names == ["Forest", "River", "SeaLake"]
    assert len(bundle.split.train) + len(bundle.split.val) + len(bundle.split.test) == 60
    images, labels = next(iter(bundle.train))
    assert images.shape == (8, 3, 32, 32)
    assert labels.shape == (8,)
    assert sum(len(b[1]) for b in bundle.test) == len(bundle.split.test)


def test_radiometric_jitter_shifts_colours_but_keeps_shape():
    from terralens.data import RadiometricJitter

    img = Image.fromarray(np.full((64, 64, 3), (90, 100, 110), dtype=np.uint8))
    jitter = RadiometricJitter(p=1.0)
    torch.manual_seed(0)
    outputs = [np.asarray(jitter(img), dtype=float) for _ in range(20)]
    assert all(o.shape == (64, 64, 3) for o in outputs)
    means = np.array([o.mean(axis=(0, 1)) for o in outputs])
    assert means.std(axis=0).min() > 3  # colours actually vary between draws
    assert means[:, 2].min() < 80  # haze removal darkens the blue channel


def test_radiometric_jitter_probability_zero_is_identity():
    from terralens.data import RadiometricJitter

    img = Image.new("RGB", (8, 8), (10, 20, 30))
    assert RadiometricJitter(p=0.0)(img) is img


def test_train_transforms_include_radiometric_jitter_only_when_asked():
    names = lambda tf: [type(t).__name__ for t in tf.transforms]  # noqa: E731
    assert "RadiometricJitter" in names(build_transforms(64, train=True, radiometric=True))
    assert "RadiometricJitter" not in names(build_transforms(64, train=True))
    assert "RadiometricJitter" not in names(build_transforms(64, train=False, radiometric=True))


def test_random_jpeg_changes_pixels_but_not_size():
    from terralens.data import RandomJPEG

    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
    out = RandomJPEG(quality=(30, 30), p=1.0)(img)
    assert out.size == img.size and out.mode == "RGB"
    assert np.abs(np.asarray(out, float) - np.asarray(img, float)).mean() > 5
    assert RandomJPEG(p=0.0)(img) is img


def test_jpeg_augmentation_runs_before_resize():
    names = [type(t).__name__ for t in build_transforms(96, train=True, jpeg=True).transforms]
    assert names.index("RandomJPEG") < names.index("Resize")
    assert "RandomJPEG" not in [
        type(t).__name__ for t in build_transforms(96, train=False).transforms
    ]
