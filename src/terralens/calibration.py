"""Probability calibration with temperature scaling (Guo et al., ICML 2017).

Label smoothing improves accuracy but caps softmax confidence (about 91% for 10
classes with epsilon = 0.1), so raw probabilities understate how often the model
is right. A single temperature T, fitted on the validation split, rescales the
logits so that predicted confidence matches observed accuracy. Accuracy is
unchanged because dividing by T never changes the arg-max.

    terralens calibrate runs/resnet18-128/model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from terralens.checkpoint import load_checkpoint, model_from_checkpoint
from terralens.data import TransformedSubset, build_transforms, load_dataset, stratified_split


@torch.no_grad()
def collect_logits(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, labels = [], []
    for images, y in loader:
        logits.append(model(images.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Temperature minimising the negative log-likelihood of `labels`."""
    log_t = torch.zeros(1, requires_grad=True)  # optimise log T so that T stays positive
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Average gap between confidence and accuracy, weighted by bin size."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def calibrate(
    model: nn.Module, val_loader: DataLoader, test_loader: DataLoader, device: torch.device
) -> dict:
    """Fit T on validation logits and report test ECE before and after scaling."""
    val_logits, val_labels = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    test_logits, test_labels = collect_logits(model, test_loader, device)
    labels = test_labels.numpy()
    return {
        "temperature": temperature,
        "test_ece_before": expected_calibration_error(test_logits.softmax(1).numpy(), labels),
        "test_ece_after": expected_calibration_error(
            (test_logits / temperature).softmax(1).numpy(), labels
        ),
    }


def calibrate_checkpoint(
    path: str | Path, data_root: str | None = None, out: str | Path | None = None
) -> dict:
    """Add a fitted temperature to an existing checkpoint (in place by default)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(path)
    cfg = ckpt["config"]
    dataset = load_dataset(cfg["dataset"], data_root or cfg["data_root"], download=False)
    split = stratified_split(
        dataset.targets, cfg["val_fraction"], cfg["test_fraction"], cfg["seed"]
    )
    tf = build_transforms(ckpt["img_size"], train=False, mean=ckpt["mean"], std=ckpt["std"])
    loaders = [
        DataLoader(TransformedSubset(dataset, idx, tf), batch_size=256, num_workers=0)
        for idx in (split.val, split.test)
    ]
    result = calibrate(model_from_checkpoint(ckpt, device), *loaders, device)
    ckpt["temperature"] = result["temperature"]
    ckpt.setdefault("metrics", {}).update(result)
    torch.save(ckpt, out or path)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit temperature scaling for a checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", type=Path, default=None, help="default: overwrite the input")
    args = parser.parse_args(argv)
    r = calibrate_checkpoint(args.checkpoint, args.data_root, args.out)
    print(
        f"T = {r['temperature']:.3f}; test ECE {r['test_ece_before']:.4f} -> "
        f"{r['test_ece_after']:.4f}"
    )


if __name__ == "__main__":
    main()
