import pytest
import torch

from terralens.models import ARCHITECTURES, build_model, count_parameters, gradcam_target_layer


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_build_model_resizes_head(name):
    model = build_model(name, num_classes=3, pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 3)


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_gradcam_target_layer_is_spatial(name):
    model = build_model(name, num_classes=3, pretrained=False).eval()
    captured = {}
    layer = gradcam_target_layer(model)
    handle = layer.register_forward_hook(lambda m, i, o: captured.update(out=o))
    with torch.no_grad():
        model(torch.randn(1, 3, 128, 128))
    handle.remove()
    assert captured["out"].ndim == 4
    assert captured["out"].shape[-1] >= 2


def test_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model("vgg11", num_classes=3, pretrained=False)


def test_count_parameters_trainable_only():
    model = build_model("resnet18", num_classes=10, pretrained=False)
    total = count_parameters(model)
    for p in model.parameters():
        p.requires_grad = False
    model.fc.weight.requires_grad = True
    assert count_parameters(model, trainable_only=True) == model.fc.weight.numel()
    assert count_parameters(model) == total
