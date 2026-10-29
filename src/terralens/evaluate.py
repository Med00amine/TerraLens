"""Evaluate checkpoints on their held-out test split and compare models.

python -m terralens.evaluate runs/resnet18-128/model.pt
python -m terralens.evaluate --compare runs/*/model.pt --out reports/comparison
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from terralens.calibration import expected_calibration_error
from terralens.checkpoint import load_checkpoint, model_from_checkpoint
from terralens.data import TransformedSubset, build_transforms, load_dataset, stratified_split
from terralens.models import count_parameters


@torch.no_grad()
def predict_loader(
    model: nn.Module, loader: DataLoader, device: torch.device, temperature: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Labels and (temperature-scaled) softmax probabilities for every sample."""
    labels, probs = [], []
    for images, y in loader:
        logits = model(images.to(device)).float() / temperature
        probs.append(torch.softmax(logits, dim=1).cpu())
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Accuracy, macro/weighted F1 and per-class precision, recall, F1 and support."""
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / max(support.sum(), 1)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
    }


def benchmark_latency(
    model: nn.Module, img_size: int, batch_size: int = 1, runs: int = 30, warmup: int = 5
) -> dict:
    """Median CPU inference latency, the number that matters for a CPU-only deployment."""
    model = model.to("cpu").eval()
    x = torch.randn(batch_size, 3, img_size, img_size)
    times = []
    with torch.inference_mode():
        for i in range(warmup + runs):
            t0 = time.perf_counter()
            model(x)
            if i >= warmup:
                times.append(time.perf_counter() - t0)
    ms = statistics.median(times) * 1000 / batch_size
    return {
        "ms_per_image": ms,
        "images_per_second": 1000 / ms,
        "batch_size": batch_size,
        "threads": torch.get_num_threads(),
    }


@dataclass
class EvaluationResult:
    report: dict
    checkpoint: dict
    dataset: object
    test_indices: list[int]
    y_true: np.ndarray
    probs: np.ndarray

    @property
    def y_pred(self) -> np.ndarray:
        return self.probs.argmax(axis=1)


def evaluate_checkpoint(
    checkpoint: str | Path,
    data_root: str | None = None,
    batch_size: int = 256,
    num_workers: int = 4,
    device: str | None = None,
) -> EvaluationResult:
    """Rebuild the checkpoint's exact test split from its saved config and score it."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(checkpoint)
    cfg = ckpt["config"]
    dataset = load_dataset(cfg["dataset"], data_root or cfg["data_root"], download=False)
    split = stratified_split(
        dataset.targets, cfg["val_fraction"], cfg["test_fraction"], cfg["seed"]
    )
    transform = build_transforms(ckpt["img_size"], train=False, mean=ckpt["mean"], std=ckpt["std"])
    loader = DataLoader(
        TransformedSubset(dataset, split.test, transform),
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=False,
    )
    model = model_from_checkpoint(ckpt, device)
    y_true, probs = predict_loader(model, loader, device, ckpt.get("temperature", 1.0))
    report = classification_report(y_true, probs.argmax(axis=1), ckpt["class_names"])
    report.update(
        {
            "ece": expected_calibration_error(probs, y_true),
            "temperature": ckpt.get("temperature", 1.0),
            "architecture": ckpt["architecture"],
            "img_size": ckpt["img_size"],
            "params": count_parameters(model),
            "test_samples": int(len(y_true)),
        }
    )
    return EvaluationResult(report, ckpt, dataset, split.test, y_true, probs)


def write_report(result: EvaluationResult, out: Path, history_path: Path | None = None) -> None:
    """metrics.json plus confusion matrix, misclassified examples and training curves."""
    from terralens import viz

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result.report, indent=2))
    class_names = result.checkpoint["class_names"]
    cm = np.array(result.report["confusion_matrix"])
    viz.plot_confusion_matrix(cm, class_names, out / "confusion_matrix.png")
    viz.plot_per_class_f1(result.report["per_class"], out / "per_class_f1.png")

    wrong = np.flatnonzero(result.y_pred != result.y_true)
    # Most confident mistakes first: the most informative failure cases.
    wrong = wrong[np.argsort(-result.probs[wrong].max(axis=1))][:12]
    samples = [
        (
            result.dataset[result.test_indices[i]][0],
            class_names[result.y_true[i]],
            class_names[result.y_pred[i]],
            float(result.probs[i].max()),
        )
        for i in wrong
    ]
    if samples:
        viz.plot_misclassified(samples, out / "misclassified.png")
    if history_path is not None and history_path.is_file():
        viz.plot_training_curves(json.loads(history_path.read_text()), out / "training_curves.png")


def compare_checkpoints(checkpoints: list[Path], data_root: str | None = None) -> list[dict]:
    """Accuracy, macro-F1, size and CPU latency for each checkpoint."""
    rows = []
    for path in checkpoints:
        result = evaluate_checkpoint(path, data_root)
        model = model_from_checkpoint(result.checkpoint, "cpu")
        latency = benchmark_latency(model, result.checkpoint["img_size"])
        rows.append(
            {
                "run": path.parent.name,
                "architecture": result.report["architecture"],
                "img_size": result.report["img_size"],
                "params_m": result.report["params"] / 1e6,
                "size_mb": path.stat().st_size / 2**20,
                "accuracy": result.report["accuracy"],
                "macro_f1": result.report["macro_f1"],
                "cpu_ms": latency["ms_per_image"],
                "train_minutes": result.checkpoint["metrics"].get("train_minutes"),
            }
        )
        print(
            f"{rows[-1]['run']:<24} acc {rows[-1]['accuracy']:.2%}  "
            f"F1 {rows[-1]['macro_f1']:.4f}  CPU {rows[-1]['cpu_ms']:.1f} ms"
        )
    return sorted(rows, key=lambda r: -r["accuracy"])


def comparison_markdown(rows: list[dict]) -> str:
    lines = [
        "| Model | Input | Params | Size | Test accuracy | Macro F1 | CPU latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['architecture']} | {r['img_size']} px | {r['params_m']:.1f} M "
            f"| {r['size_mb']:.0f} MB | {r['accuracy']:.2%} | {r['macro_f1']:.4f} "
            f"| {r['cpu_ms']:.1f} ms |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate or compare TerraLens checkpoints.")
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--compare", action="store_true", help="benchmark several checkpoints")
    parser.add_argument("--data-root", default=None, help="override the dataset location")
    parser.add_argument("--out", type=Path, default=None, help="output directory for the report")
    args = parser.parse_args(argv)

    if args.compare:
        out = args.out or Path("reports/comparison")
        out.mkdir(parents=True, exist_ok=True)
        rows = compare_checkpoints(args.checkpoints, args.data_root)
        (out / "comparison.json").write_text(json.dumps(rows, indent=2))
        (out / "comparison.md").write_text(comparison_markdown(rows))
        from terralens import viz

        viz.plot_model_comparison(rows, out / "model_comparison.png")
        print(comparison_markdown(rows))
        return

    for path in args.checkpoints:
        result = evaluate_checkpoint(path, args.data_root)
        out = args.out or path.parent / "report"
        write_report(result, out, history_path=path.parent / "history.json")
        print(
            f"{path}: accuracy {result.report['accuracy']:.2%}, "
            f"macro F1 {result.report['macro_f1']:.4f} -> {out}"
        )


if __name__ == "__main__":
    main()
