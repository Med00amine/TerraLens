"""Transfer-learning backbones from torchvision."""

from __future__ import annotations

from torch import nn
from torchvision import models

ARCHITECTURES = (
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "mobilenet_v3_large",
    "convnext_tiny",
)


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """An ImageNet backbone with its classification head resized to `num_classes`."""
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {name!r}; choose from {', '.join(ARCHITECTURES)}")

    model = models.get_model(name, weights="DEFAULT" if pretrained else None)
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # EfficientNet, MobileNetV3 and ConvNeXt all end their classifier with a Linear layer.
        head = model.classifier[-1]
        model.classifier[-1] = nn.Linear(head.in_features, num_classes)
    return model


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """The last convolutional stage, the usual Grad-CAM target.

    It is the deepest layer that still has a spatial layout, so it balances
    class-specific semantics with localisation.
    """
    if hasattr(model, "layer4"):  # ResNet
        return model.layer4
    if hasattr(model, "features"):  # EfficientNet, MobileNetV3, ConvNeXt
        return model.features[-1]
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("Model has no convolutional layer to explain")
    return convs[-1]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
