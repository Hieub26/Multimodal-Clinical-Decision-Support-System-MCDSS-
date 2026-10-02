"""
Medical Image Preprocessor.
Handles image validation, aspect ratio preserving resize with padding,
vectorized ImageNet normalization, and PyTorch tensor conversion.
"""

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import settings
from app.utils.logger import cv_logger

# Workaround for Anaconda OpenMP library duplicate issue on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ImageNet normalization values (standard for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class ImagePreprocessResult:
    """Dataclass holding preprocessing results and image metadata."""
    tensor: torch.Tensor
    original_image: Image.Image
    preprocessed_image: Image.Image
    metadata: dict[str, Any]
    # (left, top, right, bottom) of the image content inside the padded square
    content_box: tuple[int, int, int, int]

    def __getitem__(self, item: str) -> Any:
        """Subscripting fallback for backward compatibility with dictionary access."""
        return getattr(self, item)


class ImagePreprocessor:
    """Preprocesses medical images for CV model inference with validation & aspect ratio preservation."""

    def __init__(self, image_size: int | None = None):
        self.image_size = image_size or settings.cv_image_size
        # Precompute mean and std tensors for vectorized normalization
        self.mean_tensor = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std_tensor = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        cv_logger.info(f"ImagePreprocessor initialized (size={self.image_size}x{self.image_size})")

    def preprocess(
        self,
        image_input: Image.Image | str | Path | bytes | BinaryIO
    ) -> ImagePreprocessResult:
        """
        Full preprocessing pipeline for a medical image.

        Args:
            image_input: PIL Image, file path (str/Path), bytes, or file-like object

        Returns:
            ImagePreprocessResult with torch.Tensor (B, C, H, W), images, and metadata.
        """
        # Step 1: Validate and Load Image
        image = self._load_and_validate_image(image_input)
        original = image.copy()

        # Step 2: Extract Metadata
        metadata: dict[str, Any] = {
            "original_size": image.size,
            "mode": image.mode,
            "format": getattr(image, "format", "unknown"),
        }

        # Step 3: Convert to RGB mode if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
            cv_logger.info(f"Converted image mode from {metadata['mode']} to RGB")

        # Step 4: Aspect Ratio Preserving Resize + Square Padding
        resized_image, content_box = self._resize_with_aspect_ratio(image)

        # Step 5: Convert PIL Image to PyTorch Tensor [C, H, W]
        img_array = np.array(resized_image, dtype=np.float32) / 255.0  # Range [0.0, 1.0]
        tensor_chw = torch.from_numpy(img_array).permute(2, 0, 1)  # [3, H, W]

        # Step 6: Vectorized Broadcast Normalization (no channel loops)
        normalized_tensor = (tensor_chw - self.mean_tensor) / self.std_tensor

        # Step 7: Add Batch Dimension -> [1, C, H, W]
        batch_tensor = normalized_tensor.unsqueeze(0)

        cv_logger.info(
            f"Image preprocessed: {metadata['original_size']} -> "
            f"{self.image_size}x{self.image_size} tensor shape {tuple(batch_tensor.shape)}"
        )

        return ImagePreprocessResult(
            tensor=batch_tensor,
            original_image=original,
            preprocessed_image=resized_image,
            metadata=metadata,
            content_box=content_box,
        )

    def _resize_with_aspect_ratio(
        self, image: Image.Image
    ) -> tuple[Image.Image, tuple[int, int, int, int]]:
        """Resize image preserving aspect ratio and pad to square target dimensions.

        Returns the padded image and the box the original content occupies in it.
        """
        image_copy = image.copy()
        image_copy.thumbnail((self.image_size, self.image_size), Image.LANCZOS)

        delta_w = self.image_size - image_copy.width
        delta_h = self.image_size - image_copy.height
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2),
        )
        content_box = (
            padding[0],
            padding[1],
            padding[0] + image_copy.width,
            padding[1] + image_copy.height,
        )
        return ImageOps.expand(image_copy, padding, fill=(0, 0, 0)), content_box

    def _load_and_validate_image(
        self,
        image_input: Image.Image | str | Path | bytes | BinaryIO
    ) -> Image.Image:
        """Load and strictly validate image input against corruption, zero-bytes, and invalid formats."""
        if image_input is None:
            raise ValueError("Image input cannot be None.")

        try:
            if isinstance(image_input, Image.Image):
                image = image_input
            elif isinstance(image_input, (str, Path)):
                path = Path(image_input)
                if not path.exists():
                    raise ValueError(f"Image file not found: {path}")
                if path.stat().st_size == 0:
                    raise ValueError(f"Image file is empty (0 bytes): {path}")
                image = Image.open(path)
            elif isinstance(image_input, bytes):
                if len(image_input) == 0:
                    raise ValueError("Image bytes input is empty (0 bytes).")
                image = Image.open(io.BytesIO(image_input))
            elif hasattr(image_input, "read"):
                buffer = image_input.read()
                if not buffer or len(buffer) == 0:
                    raise ValueError("Image stream buffer is empty (0 bytes).")
                image = Image.open(io.BytesIO(buffer))
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            # Verify image integrity (catches corrupted image files)
            image.verify()

            # Re-open after verify() because PIL verify() alters file pointer
            if isinstance(image_input, Image.Image):
                return image_input.copy()
            elif isinstance(image_input, (str, Path)):
                return Image.open(image_input)
            elif isinstance(image_input, bytes):
                return Image.open(io.BytesIO(image_input))
            elif hasattr(image_input, "read"):
                return Image.open(io.BytesIO(buffer))

            return image
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            raise ValueError(f"Invalid or corrupted image format: {e}") from e
