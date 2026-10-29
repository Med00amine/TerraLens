import numpy as np
import pytest
import torch

from terralens.calibration import (
    calibrate_checkpoint,
    expected_calibration_error,
    fit_temperature,
)
from terralens.checkpoint import load_checkpoint
from terralens.predictor import Predictor


def synthetic_logits(n=20_000, classes=10, seed=0):
    """Labels drawn from softmax(z): z is calibrated, so 3*z is over-confident."""
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, classes, generator=gen) * 2
    labels = torch.multinomial(z.softmax(1), 1, generator=gen).squeeze(1)
    return z, labels


def test_fit_temperature_undoes_overconfidence():
    z, labels = synthetic_logits()
    assert fit_temperature(3 * z, labels) == pytest.approx(3.0, rel=0.08)
    assert fit_temperature(z, labels) == pytest.approx(1.0, rel=0.08)


def test_ece_drops_after_temperature_scaling():
    z, labels = synthetic_logits()
    over = (3 * z).softmax(1).numpy()
    t = fit_temperature(3 * z, labels)
    scaled = (3 * z / t).softmax(1).numpy()
    y = labels.numpy()
    assert expected_calibration_error(over, y) > 0.1
    assert expected_calibration_error(scaled, y) < 0.02


def test_ece_of_perfect_confident_predictions_is_zero():
    probs = np.eye(4)[[0, 1, 2, 3]]
    assert expected_calibration_error(probs, np.array([0, 1, 2, 3])) == 0.0


def test_training_stores_a_temperature(trained_checkpoint):
    ckpt = load_checkpoint(trained_checkpoint)
    assert ckpt["temperature"] > 0
    assert {"test_ece_before", "test_ece_after"} <= set(ckpt["metrics"])


def test_calibrate_checkpoint_writes_copy(trained_checkpoint, tmp_path):
    out = tmp_path / "calibrated.pt"
    result = calibrate_checkpoint(trained_checkpoint, out=out)
    assert load_checkpoint(out)["temperature"] == pytest.approx(result["temperature"])


def test_predictor_applies_temperature(trained_checkpoint):
    ckpt = load_checkpoint(trained_checkpoint)
    ckpt["temperature"] = 1e4  # an absurdly soft temperature flattens the distribution
    soft = Predictor(ckpt, device="cpu")
    tile = torch.zeros(1, 3, 32, 32)
    probs = soft.predict_proba(tile)[0]
    assert probs.max() - probs.min() < 1e-3
