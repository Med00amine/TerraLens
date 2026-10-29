"""Datasets, reproducible splits and transforms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from terralens.constants import IMAGENET_MEAN, IMAGENET_STD

_ROTATIONS = {
    1: Image.Transpose.ROTATE_90,
    2: Image.Transpose.ROTATE_180,
    3: Image.Transpose.ROTATE_270,
}


class RandomRotate90:
    """Rotate by a random multiple of 90 degrees.

    Satellite tiles have no canonical "up", so these rotations are label-preserving
    and, unlike arbitrary-angle rotations, introduce no interpolation or empty corners.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        k = int(torch.randint(0, 4, ()).item())
        return img.transpose(_ROTATIONS[k]) if k else img

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def build_transforms(
    img_size: int,
    train: bool,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Preprocessing for training (with augmentation) or evaluation/inference."""
    steps: list = [transforms.Resize((img_size, img_size))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRotate90(),
            # No hue jitter: colour is a key cue for separating vegetation classes.
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    val: list[int]
    test: list[int]


def stratified_split(
    targets: Sequence[int],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    """Split indices so every class keeps the same proportions in train/val/test.

    Deterministic for a given seed, so evaluation can rebuild the exact test set
    a checkpoint was never trained on.
    """
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be >= 0 and sum to less than 1")

    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    train, val, test = [], [], []
    for cls in np.unique(targets):
        idx = np.flatnonzero(targets == cls)
        rng.shuffle(idx)
        n_test = round(len(idx) * test_fraction)
        n_val = round(len(idx) * val_fraction)
        test += idx[:n_test].tolist()
        val += idx[n_test : n_test + n_val].tolist()
        train += idx[n_test + n_val :].tolist()
    return SplitIndices(sorted(train), sorted(val), sorted(test))


class TransformedSubset(Dataset):
    """A view of `base` restricted to `indices`, with its own transform."""

    def __init__(self, base: Dataset, indices: Sequence[int], transform=None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        image, label = self.base[self.indices[i]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_dataset(name: str, root: str | Path, download: bool = False) -> datasets.ImageFolder:
    """Load EuroSAT (downloaded on demand) or any ImageFolder-style directory.

    `name="folder"` expects `root/<class_name>/<image files>`, so TerraLens can be
    retrained on custom land-cover classes.
    """
    root = Path(root)
    if name == "eurosat":
        return datasets.EuroSAT(root=str(root), download=download)
    if name == "folder":
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {root}")
        return datasets.ImageFolder(str(root))
    raise ValueError(f"Unknown dataset {name!r}; expected 'eurosat' or 'folder'")


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]
    split: SplitIndices


def make_dataloaders(
    dataset: datasets.ImageFolder,
    img_size: int,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> DataBundle:
    """Stratified train/val/test loaders over a dataset that yields PIL images."""
    split = stratified_split(dataset.targets, val_fraction, test_fraction, seed)
    train_tf = build_transforms(img_size, train=True)
    eval_tf = build_transforms(img_size, train=False)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    return DataBundle(
        train=DataLoader(
            TransformedSubset(dataset, split.train, train_tf),
            shuffle=True,
            drop_last=True,
            generator=generator,
            **loader_kwargs,
        ),
        val=DataLoader(TransformedSubset(dataset, split.val, eval_tf), **loader_kwargs),
        test=DataLoader(TransformedSubset(dataset, split.test, eval_tf), **loader_kwargs),
        class_names=list(dataset.classes),
        split=split,
    )
