"""
Medical Image Preprocessor.
Handles image loading, resizing, normalization, and conversion to PyTorch tensors.
"""

import io
import numpy as np
from PIL import Image
from app.config import settings
from app.utils.logger import cv_logger

# ImageNet normalization values (standard for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImagePreprocessor:
    """Preprocesses medical images for CV model inference."""

    def __init__(self, image_size: int = None):
        self.image_size = image_size or settings.cv_image_size
        cv_logger.info(f"ImagePreprocessor initialized (size={self.image_size})")

    def preprocess(self, image_input) -> dict:
        """
        Full preprocessing pipeline for a medical image.

        Args:
            image_input: PIL Image, file path (str), or bytes

        Returns:
            Dict with 'tensor', 'original_image', 'metadata'
        """
        # Load image
        image = self._load_image(image_input)
        original = image.copy()

        # Get metadata
        metadata = {
            "original_size": image.size,
            "mode": image.mode,
            "format": getattr(image, "format", "unknown"),
        }

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
            cv_logger.info(f"Converted image from {metadata['mode']} to RGB")

        # Resize
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Convert to numpy and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Apply ImageNet normalization
        for c in range(3):
            img_array[:, :, c] = (img_array[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        # Convert to CHW format for PyTorch (channels, height, width)
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension
        tensor = np.expand_dims(img_array, axis=0)

        cv_logger.info(f"Image preprocessed: {metadata['original_size']} -> {self.image_size}x{self.image_size}")

        return {
            "tensor": tensor,
            "original_image": original,
            "preprocessed_image": image,
            "metadata": metadata,
        }

    def _load_image(self, image_input) -> Image.Image:
        """Load image from various input types."""
        if isinstance(image_input, Image.Image):
            return image_input
        elif isinstance(image_input, str):
            return Image.open(image_input)
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))
        elif hasattr(image_input, "read"):
            return Image.open(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
