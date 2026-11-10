"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for CV model explainability.
Generates heatmaps showing which image regions influenced the model's prediction.
"""

import uuid
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import settings
from app.utils.logger import cv_logger


def classify_features(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Class logits from the output of `model.features`.

    The rest of torchvision's DenseNet.forward (ReLU, global average pooling,
    classifier), written out so that the feature map of one forward pass can
    serve both the prediction and its Grad-CAM map.
    """
    activations = F.relu(features)
    pooled = torch.flatten(F.adaptive_avg_pool2d(activations, 1), 1)
    return model.classifier(pooled)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for a torchvision DenseNet.

    The map is taken on the feature map the classifier actually pools: the
    output of `features` (after norm5) passed through the final ReLU. The
    forward pass is recomposed here instead of hooking a layer, because
    DenseNet applies that ReLU in place, which a backward hook does not
    allow; hooking denseblock4 instead, before norm5 and the ReLU, gave a
    map that on some images agreed poorly with what the classifier used.

    Only the gradient of the class score with respect to that feature map is
    needed, so nothing is propagated back through the network and no gradient
    is stored on the model: concurrent calls do not interfere.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Generate a Grad-CAM heatmap normalised to [0, 1]."""
        self.model.eval()
        with torch.no_grad():
            features = self.model.features(input_tensor)
        return self.generate_from_features(features, class_idx)

    def generate_from_features(
        self, features: torch.Tensor, class_idx: int = None
    ) -> np.ndarray:
        """The heatmap from an already computed output of `model.features`."""
        features = features.detach().requires_grad_(True)
        with torch.enable_grad():
            activations = F.relu(features)
            pooled = torch.flatten(F.adaptive_avg_pool2d(activations, 1), 1)
            output = self.model.classifier(pooled)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            (gradients,) = torch.autograd.grad(output[0, class_idx], features)

        # Channel weights: gradients averaged over the spatial grid
        weights = gradients[0].mean(dim=(1, 2))
        heatmap = torch.relu((weights[:, None, None] * activations[0]).sum(dim=0))
        heatmap = heatmap.detach().cpu().numpy()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap


# --- Drawing ---------------------------------------------------------------
# The three panels are composed with Pillow. They used to be a matplotlib
# figure, which took 0.6 to 1.8 s per image (several times the model itself)
# and had to be serialised across requests because pyplot is not thread-safe.

# Longest side of one panel. A chest film can be 3000 px wide; the page shows
# three panels side by side, so nothing is gained by drawing them larger.
PANEL_MAX_SIDE = 640
_MARGIN = 16
_TITLE_HEIGHT = 36
_OVERLAY_HEATMAP_WEIGHT = 0.4


def _jet_lookup_table() -> np.ndarray:
    """The 256 RGB colours of the "jet" colour map (blue, cyan, yellow, red)."""
    x = np.linspace(0.0, 1.0, 256)
    red = np.interp(x, [0.0, 0.35, 0.66, 0.89, 1.0], [0.0, 0.0, 1.0, 1.0, 0.5])
    green = np.interp(x, [0.0, 0.125, 0.375, 0.64, 0.91, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    blue = np.interp(x, [0.0, 0.11, 0.34, 0.65, 1.0], [0.5, 1.0, 1.0, 0.0, 0.0])
    return np.round(np.stack([red, green, blue], axis=1) * 255).astype(np.uint8)


_JET = _jet_lookup_table()


def _title_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=20)
    except (TypeError, OSError):
        # Pillow without FreeType: the small built-in bitmap font
        return ImageFont.load_default()


def _panel_size(image_size: tuple[int, int]) -> tuple[int, int]:
    """The image size scaled so that its longest side is PANEL_MAX_SIDE."""
    width, height = image_size
    scale = PANEL_MAX_SIDE / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def compose_panels(
    original_image: Image.Image, heatmap: Image.Image, overlay_title: str
) -> Image.Image:
    """Original, heatmap and overlay side by side, each under its title.

    Args:
        original_image: The uploaded image, any mode and size.
        heatmap: Single-channel ("L") heatmap covering the same area.
        overlay_title: Title of the third panel.
    """
    size = _panel_size(original_image.size)
    # A grayscale upload stays gray: it is converted, not colour-mapped
    original = original_image.convert("RGB").resize(size, Image.LANCZOS)
    heat = np.asarray(heatmap.resize(size, Image.BICUBIC))
    coloured = _JET[heat]
    blended = (
        (1.0 - _OVERLAY_HEATMAP_WEIGHT) * np.asarray(original, dtype=np.float32)
        + _OVERLAY_HEATMAP_WEIGHT * coloured
    )
    panels = (
        ("Original Image", original),
        ("Grad-CAM Heatmap", Image.fromarray(coloured)),
        (overlay_title, Image.fromarray(np.uint8(np.clip(blended, 0, 255)))),
    )

    width, height = size
    canvas = Image.new(
        "RGB",
        (3 * width + 4 * _MARGIN, _TITLE_HEIGHT + height + _MARGIN),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = _title_font()
    for index, (title, panel) in enumerate(panels):
        left = _MARGIN + index * (width + _MARGIN)
        canvas.paste(panel, (left, _TITLE_HEIGHT))
        draw.text(
            (left + width / 2, _TITLE_HEIGHT / 2), title,
            fill="black", font=font, anchor="mm",
        )
    return canvas


class GradCAMVisualizer:
    """Generates and saves Grad-CAM overlay visualizations."""

    def __init__(self, grad_cam: GradCAM, class_names: list[str]):
        self.grad_cam = grad_cam
        self.class_names = class_names

    def generate_and_save(
        self,
        tensor: torch.Tensor,
        original_image: Image.Image,
        class_idx: int,
        content_box: tuple[int, int, int, int] | None = None,
        features: torch.Tensor | None = None,
    ) -> str:
        """Generate Grad-CAM heatmap overlay and save as PNG.

        Args:
            tensor: Preprocessed input tensor [1, C, H, W]
            original_image: Original PIL Image for overlay
            class_idx: Target class index for Grad-CAM
            content_box: (left, top, right, bottom) of the image content
                inside the padded model input. None means no padding.
            features: Output of `model.features` for this tensor, when the
                caller already has it. Saves a second pass through the model.

        Returns:
            File path to saved visualization, or empty string on failure.
        """
        try:
            if features is None:
                heatmap = self.grad_cam.generate(tensor, class_idx)
            else:
                heatmap = self.grad_cam.generate_from_features(features, class_idx)

            # The heatmap covers the padded square the model saw. Scale it to
            # the model input and cut the padding off before stretching it
            # over the image, otherwise it is shifted on any non-square image.
            heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(
                (tensor.shape[-1], tensor.shape[-2]), Image.LANCZOS
            )
            if content_box is not None:
                heatmap_image = heatmap_image.crop(content_box)

            figure = compose_panels(
                original_image, heatmap_image,
                f"Overlay: {self.class_names[class_idx]}",
            )

            save_dir = Path(settings.image_storage_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
            save_path = str(save_dir / filename)
            # Light compression: the default level spends more time on the
            # file than the model spends on the image, for a few percent
            figure.save(save_path, format="PNG", compress_level=3)

            cv_logger.info(f"Grad-CAM saved to {save_path}")
            return save_path

        except Exception as e:
            cv_logger.error(f"Grad-CAM generation failed: {e}")
            return ""
