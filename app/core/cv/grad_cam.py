"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for CV model explainability.
Generates heatmaps showing which image regions influenced the model's prediction.
"""

import uuid
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from app.config import settings
from app.utils.logger import cv_logger


class GradCAM:
    """Gradient-weighted Class Activation Mapping for a torchvision DenseNet.

    The map is taken on the feature map the classifier actually pools: the
    output of `features` (after norm5) passed through the final ReLU. The
    forward pass is recomposed here instead of hooking a layer, because
    DenseNet applies that ReLU in place, which a backward hook does not
    allow; hooking denseblock4 instead, before norm5 and the ReLU, gave a
    map that on some images agreed poorly with what the classifier used.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Generate a Grad-CAM heatmap normalised to [0, 1]."""
        self.model.eval()
        features = self.model.features(input_tensor)
        features.retain_grad()
        activations = F.relu(features)
        pooled = torch.flatten(F.adaptive_avg_pool2d(activations, 1), 1)
        output = self.model.classifier(pooled)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Channel weights: gradients averaged over the spatial grid
        weights = features.grad[0].mean(dim=(1, 2))
        heatmap = torch.relu((weights[:, None, None] * activations[0]).sum(dim=0))
        heatmap = heatmap.detach().cpu().numpy()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap


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
    ) -> str:
        """Generate Grad-CAM heatmap overlay and save as PNG.

        Args:
            tensor: Preprocessed input tensor [1, C, H, W]
            original_image: Original PIL Image for overlay
            class_idx: Target class index for Grad-CAM
            content_box: (left, top, right, bottom) of the image content
                inside the padded model input. None means no padding.

        Returns:
            File path to saved visualization, or empty string on failure.
        """
        fig = None
        try:
            # Clone and enable gradients for Grad-CAM
            tensor_with_grad = tensor.clone().detach().requires_grad_(True)
            heatmap = self.grad_cam.generate(tensor_with_grad, class_idx)

            # The heatmap covers the padded square the model saw. Scale it to
            # the model input and cut the padding off before stretching it
            # over the original image, otherwise it is shifted on any
            # non-square image.
            heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(
                (tensor.shape[-1], tensor.shape[-2]), Image.LANCZOS
            )
            if content_box is not None:
                heatmap_image = heatmap_image.crop(content_box)
            heatmap_resized = np.array(
                heatmap_image.resize(original_image.size, Image.LANCZOS)
            ) / 255.0

            # Create overlay
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image (a grayscale upload is shown in gray, not in
            # matplotlib's default colour map)
            original_rgb = original_image.convert("RGB")
            axes[0].imshow(original_rgb)
            axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
            axes[0].axis("off")

            # Grad-CAM heatmap
            axes[1].imshow(heatmap_resized, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold")
            axes[1].axis("off")

            # Overlay
            original_array = np.array(original_rgb) / 255.0
            colored_heatmap = cm.jet(heatmap_resized)[:, :, :3]
            overlay = 0.6 * original_array + 0.4 * colored_heatmap
            overlay = np.clip(overlay, 0, 1)
            axes[2].imshow(overlay)
            axes[2].set_title(
                f"Overlay — {self.class_names[class_idx]}",
                fontsize=12, fontweight="bold",
            )
            axes[2].axis("off")

            plt.tight_layout()

            # Save
            save_dir = Path(settings.image_storage_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
            save_path = str(save_dir / filename)
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

            cv_logger.info(f"Grad-CAM saved to {save_path}")
            return save_path

        except Exception as e:
            cv_logger.error(f"Grad-CAM generation failed: {e}")
            return ""
        finally:
            if fig is not None:
                plt.close(fig)
