import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from terralens.explain import GradCAM, apply_colormap, overlay_heatmap
from terralens.models import build_model


def red_detector() -> nn.Module:
    """A transparent 2-class model whose class-0 evidence is the red channel."""
    conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
    head = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1))
        head.weight.copy_(torch.tensor([[1.0], [-1.0]]))
    return nn.Sequential(conv, nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), head)


def test_gradcam_localises_the_evidence():
    x = torch.zeros(1, 3, 32, 32)
    x[:, 0, 4:12, 20:28] = 1.0  # red square in the top-right corner
    with GradCAM(red_detector()) as cam_fn:
        cam, cls, probs = cam_fn(x, class_idx=0)

    assert cls == 0
    square = np.zeros_like(cam, dtype=bool)
    square[4:12, 20:28] = True
    assert cam[square].mean() > 0.9
    assert cam[~square].mean() < 0.2


@pytest.mark.parametrize("name", ["resnet18", "efficientnet_b0", "mobilenet_v3_large"])
def test_gradcam_output_contract(name):
    model = build_model(name, num_classes=4, pretrained=False)
    with GradCAM(model) as cam_fn:
        cam, cls, probs = cam_fn(torch.randn(1, 3, 96, 96))
    assert cam.shape == (96, 96)
    assert cam.min() >= 0 and cam.max() <= 1
    assert 0 <= cls < 4
    assert probs.shape == (4,)
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)


def test_gradcam_respects_requested_class_and_frozen_weights():
    model = build_model("resnet18", num_classes=3, pretrained=False)
    for p in model.parameters():
        p.requires_grad = False
    with GradCAM(model) as cam_fn:
        _, cls, _ = cam_fn(torch.randn(1, 3, 64, 64), class_idx=2)
    assert cls == 2


def test_gradcam_removes_hook_and_leaves_no_gradients():
    model = build_model("resnet18", num_classes=3, pretrained=False)
    with GradCAM(model) as cam_fn:
        cam_fn(torch.randn(1, 3, 64, 64))
    assert not model.layer4._forward_hooks
    assert all(p.grad is None for p in model.parameters())


def test_gradcam_rejects_batches():
    with GradCAM(red_detector()) as cam_fn, pytest.raises(ValueError):
        cam_fn(torch.zeros(2, 3, 16, 16))


def test_colormap_endpoints():
    rgb = apply_colormap(np.array([0.0, 1.0, 2.0, -1.0]))
    assert rgb.dtype == np.uint8
    assert rgb[0].tolist() == [48, 18, 59]
    assert rgb[1].tolist() == [122, 4, 3]
    assert rgb[2].tolist() == rgb[1].tolist()  # clipped
    assert rgb[3].tolist() == rgb[0].tolist()


def test_overlay_heatmap_size():
    tile = Image.new("RGB", (64, 64), (10, 120, 30))
    cam = np.linspace(0, 1, 64 * 64).reshape(64, 64)
    assert overlay_heatmap(tile, cam, size=256).size == (256, 256)
    assert overlay_heatmap(tile, cam, size=None).size == (64, 64)
