"""Grad-CAM explanations: which part of a tile drove the prediction?"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from terralens.models import gradcam_target_layer

# Anchor points of a turbo-like colour map (dark blue -> green -> yellow -> dark red).
_COLORMAP_STOPS = np.array([0.0, 0.13, 0.25, 0.38, 0.5, 0.63, 0.75, 0.87, 1.0])
_COLORMAP_RGB = np.array(
    [
        (48, 18, 59),
        (70, 107, 227),
        (40, 170, 245),
        (26, 228, 182),
        (106, 253, 106),
        (191, 238, 53),
        (250, 186, 57),
        (238, 98, 23),
        (122, 4, 3),
    ],
    dtype=float,
)


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Selvaraju et al., ICCV 2017).

    The activations of `target_layer` are weighted by the spatially averaged
    gradient of the class score, summed over channels and rectified, giving a
    coarse map of the evidence for that class.

    Use as a context manager (or call `remove()`) to detach the hook afterwards.
    """

    def __init__(
        self, model: nn.Module, target_layer: nn.Module | None = None, temperature: float = 1.0
    ):
        self.model = model
        self.temperature = temperature
        self.target_layer = (
            target_layer if target_layer is not None else gradcam_target_layer(model)
        )
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handle = self.target_layer.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output: torch.Tensor) -> None:
        self._activations = output
        if output.requires_grad:  # plain no-grad forward passes are left untouched
            output.register_hook(self._store_gradient)

    def _store_gradient(self, grad: torch.Tensor) -> None:
        self._gradients = grad

    def __call__(
        self, x: torch.Tensor, class_idx: int | None = None
    ) -> tuple[np.ndarray, int, torch.Tensor]:
        """Explain a single preprocessed image `x` of shape (1, 3, H, W).

        Returns the heatmap (H, W) scaled to [0, 1], the explained class index
        (the top prediction unless `class_idx` is given) and the class probabilities.
        """
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(f"Expected a single image of shape (1, 3, H, W), got {tuple(x.shape)}")
        self.model.eval()
        with torch.enable_grad():
            # Requiring input gradients keeps the hook working even with frozen weights.
            x = x.detach().requires_grad_(True)
            logits = self.model(x)
            probs = (logits / self.temperature).softmax(dim=1)[0].detach()
            if class_idx is None:
                class_idx = int(probs.argmax())
            self.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()

        acts = self._activations[0].detach()
        grads = self._gradients[0].detach()
        self.model.zero_grad(set_to_none=True)

        weights = grads.mean(dim=(1, 2))
        cam = F.relu((weights[:, None, None] * acts).sum(dim=0))
        cam = F.interpolate(
            cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0]
        cam -= cam.min()
        peak = cam.max()
        if peak > 0:
            cam /= peak
        return cam.cpu().numpy(), class_idx, probs.cpu()

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def apply_colormap(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB (uint8) with a turbo-like colour map."""
    v = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    rgb = np.stack([np.interp(v, _COLORMAP_STOPS, _COLORMAP_RGB[:, c]) for c in range(3)], -1)
    return rgb.round().astype(np.uint8)


def overlay_heatmap(
    image: Image.Image, cam: np.ndarray, alpha: float = 0.5, size: int | None = 256
) -> Image.Image:
    """Blend a Grad-CAM heatmap over `image`, optionally upscaled to `size` x `size`.

    Small tiles are enlarged with nearest-neighbour sampling so individual pixels
    stay visible, while the heatmap is smoothly interpolated.
    """
    base = image.convert("RGB")
    if size is not None:
        base = base.resize((size, size), Image.Resampling.NEAREST)
    heat = Image.fromarray(apply_colormap(cam)).resize(base.size, Image.Resampling.BILINEAR)
    return Image.blend(base, heat, alpha)
