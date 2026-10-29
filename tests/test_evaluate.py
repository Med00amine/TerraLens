import json

import numpy as np
import pytest
import torch

from terralens.checkpoint import load_checkpoint, model_from_checkpoint
from terralens.evaluate import (
    benchmark_latency,
    classification_report,
    compare_checkpoints,
    comparison_markdown,
    confusion_matrix,
    evaluate_checkpoint,
    write_report,
)
from terralens.models import build_model
from terralens.train import TrainConfig, train


@pytest.fixture
def trained_checkpoint(image_folder, tmp_path):
    out = tmp_path / "run"
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


def test_confusion_matrix_counts():
    cm = confusion_matrix(np.array([0, 0, 1, 2, 2, 2]), np.array([0, 1, 1, 2, 2, 0]), 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 2]]


def test_classification_report_values():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    report = classification_report(y_true, y_pred, ["a", "b"])
    assert report["accuracy"] == 0.75
    assert report["per_class"]["a"] == {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2}
    assert report["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert report["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_classification_report_handles_never_predicted_class():
    report = classification_report(np.array([0, 1]), np.array([0, 0]), ["a", "b"])
    assert report["per_class"]["b"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}


def test_benchmark_latency_reports_positive_numbers():
    model = build_model("mobilenet_v3_large", 3, pretrained=False)
    result = benchmark_latency(model, img_size=32, runs=3, warmup=1)
    assert result["ms_per_image"] > 0
    assert result["images_per_second"] == pytest.approx(1000 / result["ms_per_image"])


def test_load_checkpoint_rejects_foreign_files(tmp_path):
    torch.save({"state_dict": {}}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not a TerraLens checkpoint"):
        load_checkpoint(tmp_path / "other.pt")
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")


def test_evaluate_and_report(trained_checkpoint, tmp_path):
    result = evaluate_checkpoint(trained_checkpoint, num_workers=0, device="cpu")
    assert result.report["test_samples"] == 9  # 15% of 20 images x 3 classes
    assert set(result.report["per_class"]) == {"Forest", "River", "SeaLake"}
    assert np.allclose(result.probs.sum(axis=1), 1, atol=1e-5)

    # The score must match what training reported for the same split.
    ckpt = load_checkpoint(trained_checkpoint)
    assert result.report["accuracy"] == pytest.approx(ckpt["metrics"]["test_acc"])

    out = tmp_path / "report"
    write_report(result, out, history_path=trained_checkpoint.parent / "history.json")
    assert json.loads((out / "metrics.json").read_text())["accuracy"] == result.report["accuracy"]
    assert (out / "confusion_matrix.png").is_file()
    assert (out / "training_curves.png").is_file()


def test_compare_checkpoints(trained_checkpoint):
    rows = compare_checkpoints([trained_checkpoint])
    assert rows[0]["architecture"] == "resnet18"
    assert rows[0]["cpu_ms"] > 0
    assert "| resnet18 | 32 px |" in comparison_markdown(rows)


def test_model_from_checkpoint_is_in_eval_mode(trained_checkpoint):
    model = model_from_checkpoint(load_checkpoint(trained_checkpoint))
    assert not model.training
