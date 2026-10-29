"""Report figures (requires the `train` extra for matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from terralens.constants import color_for, label_for  # noqa: E402

DPI = 150


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_training_curves(history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], "--", color="#1f77b4", label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], "-o", color="#1f77b4", label="val")
    ax_acc.plot(
        epochs, [100 * h["train_acc"] for h in history], "--", color="#2ca02c", label="train"
    )
    ax_acc.plot(epochs, [100 * h["val_acc"] for h in history], "-o", color="#2ca02c", label="val")
    ax_loss.set(title="Loss (label-smoothed cross-entropy)", xlabel="Epoch")
    ax_acc.set(title="Accuracy (%)", xlabel="Epoch")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_xticks(epochs)
    _save(fig, path)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    """Row-normalised confusion matrix; the diagonal is per-class recall."""
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    labels = [label_for(c) for c in class_names]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Confusion matrix (test set)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] == 0:
                continue
            text = f"{100 * norm[i, j]:.1f}%" if i == j else str(cm[i, j])
            color = "white" if norm[i, j] > 0.5 else "#333333"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")
    _save(fig, path)


def plot_per_class_f1(per_class: dict, path: Path) -> None:
    items = sorted(per_class.items(), key=lambda kv: kv[1]["f1"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(
        [label_for(k) for k, _ in items],
        [100 * v["f1"] for _, v in items],
        color=[color_for(k) for k, _ in items],
    )
    low = min(100 * v["f1"] for _, v in items)
    ax.set_xlim(max(0, low - 5), 100)
    ax.set(xlabel="F1 score (%)", title="Per-class F1 (test set)")
    ax.grid(axis="x", alpha=0.3)
    for i, (_, v) in enumerate(items):
        ax.text(100 * v["f1"] + 0.1, i, f"{100 * v['f1']:.1f}", va="center", fontsize=8)
    _save(fig, path)


def plot_misclassified(samples: list[tuple], path: Path, cols: int = 6) -> None:
    """Grid of (image, true, predicted, confidence) tuples."""
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (image, true, pred, conf) in zip(axes.ravel(), samples, strict=False):
        ax.imshow(image, interpolation="nearest")
        ax.set_title(f"true: {label_for(true)}\npred: {label_for(pred)} ({conf:.0%})", fontsize=8)
    fig.suptitle("Most confident mistakes on the test set", fontweight="bold")
    _save(fig, path)


def plot_class_examples(dataset, class_names: list[str], path: Path, per_class: int = 5) -> None:
    """One row per class with example tiles, for documenting the dataset."""
    targets = np.asarray(dataset.targets)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(
        len(class_names), per_class, figsize=(1.6 * per_class, 1.6 * len(class_names))
    )
    for r, name in enumerate(class_names):
        idx = rng.choice(np.flatnonzero(targets == r), per_class, replace=False)
        for c, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(dataset[int(i)][0], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label_for(name), rotation=0, ha="right", va="center", fontsize=9)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    _save(fig, path)


def plot_model_comparison(rows: list[dict], path: Path) -> None:
    """Accuracy vs CPU latency; marker area is proportional to parameter count."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows:
        ax.scatter(
            r["cpu_ms"],
            100 * r["accuracy"],
            s=40 + 12 * r["params_m"],
            alpha=0.7,
            edgecolor="black",
        )
        ax.annotate(
            f"{r['architecture']} @{r['img_size']}",
            (r["cpu_ms"], 100 * r["accuracy"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set(
        xlabel="CPU latency per image (ms, log scale)",
        ylabel="Test accuracy (%)",
        title="Accuracy vs. inference cost (marker size = parameters)",
    )
    ax.grid(alpha=0.3, which="both")
    _save(fig, path)
