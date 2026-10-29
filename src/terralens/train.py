"""Train a land-cover classifier.

python -m terralens.train --model resnet18 --img-size 128 --epochs 12
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from terralens import __version__
from terralens.calibration import calibrate
from terralens.constants import IMAGENET_MEAN, IMAGENET_STD
from terralens.data import load_dataset, make_dataloaders
from terralens.models import ARCHITECTURES, build_model, count_parameters

CHECKPOINT_FORMAT = 1


@dataclass
class TrainConfig:
    dataset: str = "eurosat"
    data_root: str = "data"
    download: bool = True
    model: str = "resnet18"
    pretrained: bool = True
    img_size: int = 128
    epochs: int = 12
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    radiometric_aug: bool = False
    jpeg_aug: bool = True
    jpeg_min_quality: int = 50
    patience: int = 4
    num_workers: int = 4
    seed: int = 42
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    amp: bool = True
    output_dir: str = "runs/resnet18"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
) -> dict[str, float]:
    """One pass over `loader`; trains when an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    channels_last = device.type == "cuda"

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if channels_last:
                images = images.to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

    return {"loss": total_loss / max(seen, 1), "acc": correct / max(seen, 1)}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainConfig,
    class_names: list[str],
    metrics: dict,
    temperature: float = 1.0,
) -> None:
    """Save weights together with everything needed to use them for inference."""
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT,
            "terralens_version": __version__,
            "architecture": config.model,
            "num_classes": len(class_names),
            "class_names": list(class_names),
            "img_size": config.img_size,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "temperature": temperature,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "metrics": metrics,
            "config": asdict(config),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        path,
    )


def train(config: TrainConfig) -> dict:
    """Train with early stopping; writes model.pt, history.json and metrics.json."""
    seed_everything(config.seed)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = config.amp and device.type == "cuda"

    dataset = load_dataset(config.dataset, config.data_root, download=config.download)
    data = make_dataloaders(
        dataset,
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        radiometric_aug=config.radiometric_aug,
        jpeg_aug=config.jpeg_aug,
        jpeg_min_quality=config.jpeg_min_quality,
    )
    print(
        f"Data: {len(data.split.train)} train / {len(data.split.val)} val / "
        f"{len(data.split.test)} test images, {len(data.class_names)} classes"
    )

    model = build_model(config.model, len(data.class_names), pretrained=config.pretrained)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    print(f"Model: {config.model} ({count_parameters(model) / 1e6:.1f}M params) on {device}")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr, total_steps=config.epochs * len(data.train), pct_start=0.15
    )
    scaler = torch.amp.GradScaler("cuda") if amp else None

    history: list[dict] = []
    best = {"val_acc": -1.0, "val_loss": float("inf"), "epoch": 0}
    checkpoint_path = out / "model.pt"
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        t0 = time.perf_counter()
        tr = run_epoch(model, data.train, criterion, device, optimizer, scheduler, scaler, amp)
        va = run_epoch(model, data.val, criterion, device, amp=amp)
        record = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - t0,
        }
        history.append(record)
        print(
            f"epoch {epoch:>2}/{config.epochs} | train loss {tr['loss']:.4f} acc {tr['acc']:.2%}"
            f" | val loss {va['loss']:.4f} acc {va['acc']:.2%} | {record['seconds']:.1f}s"
        )

        improved = va["acc"] > best["val_acc"] or (
            va["acc"] == best["val_acc"] and va["loss"] < best["val_loss"]
        )
        if improved:
            best = {"val_acc": va["acc"], "val_loss": va["loss"], "epoch": epoch}
            save_checkpoint(checkpoint_path, model, config, data.class_names, dict(best))
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"Early stopping: no val improvement for {config.patience} epochs")
                break

    # Final, unbiased score: best checkpoint on the held-out test split.
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    te = run_epoch(model, data.test, criterion, device, amp=amp)
    calibration = calibrate(model, data.val, data.test, device)

    summary = {
        "model": config.model,
        "img_size": config.img_size,
        "params": count_parameters(model),
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "val_acc": best["val_acc"],
        "val_loss": best["val_loss"],
        "test_acc": te["acc"],
        "test_loss": te["loss"],
        "train_minutes": (time.perf_counter() - started) / 60,
        **calibration,
    }
    save_checkpoint(
        checkpoint_path, model, config, data.class_names, summary, calibration["temperature"]
    )
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(
        f"Test accuracy {te['acc']:.2%} (best epoch {best['epoch']}), temperature "
        f"{calibration['temperature']:.2f}, test ECE {calibration['test_ece_before']:.4f} -> "
        f"{calibration['test_ece_after']:.4f} -> {checkpoint_path}"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train a TerraLens land-cover classifier.")
    parser.add_argument("--dataset", choices=["eurosat", "folder"], default=defaults.dataset)
    parser.add_argument("--data-root", default=defaults.data_root)
    parser.add_argument("--model", choices=ARCHITECTURES, default=defaults.model)
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument(
        "--radiometric-aug",
        action="store_true",
        help="experimental L1C->L2A colour augmentation (see data.RadiometricJitter)",
    )
    parser.add_argument(
        "--no-jpeg-aug",
        dest="jpeg_aug",
        action="store_false",
        help="disable random JPEG re-compression (see data.RandomJPEG)",
    )
    numeric = {"int": int, "float": float}  # annotations are strings (postponed evaluation)
    for f in fields(TrainConfig):
        if f.type in numeric:
            flag = "--" + f.name.replace("_", "-")
            parser.add_argument(flag, type=numeric[f.type], default=getattr(defaults, f.name))
    parser.add_argument("--output-dir", default=None, help="default: runs/<model>-<img_size>")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = f"runs/{args.model}-{args.img_size}"
    return TrainConfig(**vars(args))


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
