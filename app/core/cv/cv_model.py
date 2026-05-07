"""
CV Model: DenseNet-121 based medical image classifier with Grad-CAM explainability.
"""

import os
import uuid
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from app.config import settings
from app.core.cv.image_preprocessor import ImagePreprocessor
from app.utils.logger import cv_logger


class GradCAM:
    """Gradient-weighted Class Activation Mapping for model explainability."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach().clone()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        # Pool gradients across spatial dimensions
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight activations by gradients
        activations = self.activations[0]
        weighted_activations = activations * pooled_gradients[:, None, None]

        # Generate heatmap
        heatmap = torch.mean(weighted_activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap


class MedicalCVModel:
    """Medical image classification model with Grad-CAM support."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = settings.cv_num_classes
        self.class_names = settings.cv_class_names
        self.class_thresholds = [settings.confidence_threshold_cv] * self.num_classes
        self._load_model_metadata()
        self.image_preprocessor = ImagePreprocessor()
        self._model = None
        self._grad_cam = None
        cv_logger.info(f"MedicalCVModel initialized (device={self.device})")

    def _load_model_metadata(self):
        """Load optional model metadata such as class labels and thresholds."""
        meta_path = getattr(settings, "cv_model_meta_path", "")
        if not meta_path or not os.path.exists(meta_path):
            cv_logger.warning("No CV model metadata found. Using default class thresholds.")
            return

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            labels = metadata.get("labels")
            thresholds = metadata.get("thresholds")

            if labels:
                if labels != self.class_names:
                    cv_logger.warning(
                        "CV metadata labels differ from app config. Using metadata labels."
                    )
                self.class_names = labels
                self.num_classes = len(labels)

            if thresholds and len(thresholds) == self.num_classes:
                relaxation = float(getattr(settings, "cv_threshold_relaxation", 1.0))
                relaxation = min(max(relaxation, 0.0), 1.0)
                self.class_thresholds = [float(t) * relaxation for t in thresholds]
            else:
                cv_logger.warning(
                    "CV metadata thresholds missing or invalid. Using default threshold."
                )

            cv_logger.info(
                "Loaded CV metadata from %s (best_auc=%s)",
                meta_path,
                metadata.get("best_auc", "unknown"),
            )
        except Exception as e:
            cv_logger.error("Failed to load CV metadata from %s: %s", meta_path, e)

    @property
    def model(self) -> nn.Module:
        """Lazy-load the CV model."""
        if self._model is None:
            self._model = self._build_model()
            self._model.to(self.device)
            self._model.eval()
        return self._model

    @property
    def grad_cam(self) -> GradCAM:
        """Lazy-load Grad-CAM."""
        if self._grad_cam is None:
            # Target the last convolutional layer in DenseNet-121
            target_layer = self.model.features.denseblock4
            self._grad_cam = GradCAM(self.model, target_layer)
        return self._grad_cam

    def _build_model(self) -> nn.Module:
        """Build DenseNet-121 with custom classification head."""
        # Use pretrained weights from ImageNet as a starting point
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=weights)

        # Replace final FC layer for our classes
        # Modified to match the single Linear layer in the provided weights file
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, self.num_classes)

        # Load fine-tuned weights if available
        model_path = settings.cv_model_path
        if os.path.exists(model_path):
            cv_logger.info(f"Loading model weights from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            cv_logger.info("Fine-tuned weights loaded successfully")
        else:
            cv_logger.warning(
                "No fine-tuned weights found at %s. "
                "Using ImageNet pretrained features with untrained classifier. "
                "Predictions will be unreliable — please train the model first.",
                model_path,
            )

        return model

    def predict(self, image_input) -> dict:
        """
        Run inference on a medical image.

        Args:
            image_input: PIL Image, file path, or bytes

        Returns:
            CVDiagnosis dictionary
        """
        cv_logger.info("Starting CV prediction")

        # Preprocess
        preprocessed = self.image_preprocessor.preprocess(image_input)
        tensor = torch.from_numpy(preprocessed["tensor"]).to(self.device)

        # Inference. ChestX-ray14 is a multi-label dataset, so use sigmoid
        # probabilities rather than softmax over mutually-exclusive classes.
        with torch.no_grad():
            output = self.model(tensor)
            temperature = max(float(settings.cv_logit_temperature), 1e-6)
            scaled_logits = output[0] / temperature
            probabilities = torch.sigmoid(scaled_logits)

        # Get predictions
        probs = probabilities.cpu().numpy()
        top_indices = np.argsort(probs)[::-1][: settings.cv_top_k]
        top_predictions = [
            {
                "class": self.class_names[i],
                "probability": float(probs[i]),
                "threshold": float(self.class_thresholds[i]),
            }
            for i in top_indices
        ]

        predictions = [
            {
                "class": self.class_names[i],
                "probability": float(probs[i]),
                "threshold": float(self.class_thresholds[i]),
            }
            for i in range(len(probs))
            if probs[i] >= self.class_thresholds[i]
        ]

        max_probability = float(np.max(probs))
        top_class_idx = int(np.argmax(probs))
        top_class_threshold = float(self.class_thresholds[top_class_idx])

        # Multi-label formatting. This model has disease labels only, so a low
        # score across all classes is not the same thing as a calibrated
        # "normal" diagnosis.
        if len(predictions) == 0:
            predicted_class_name = "No confident CV finding"
            confidence = max_probability
            detected_predictions = []
            finding_detected = False
            severity = "unknown"
            urgency = "routine"
            explanation = (
                "The CV model did not produce any class probability above the "
                "configured decision threshold. This should be treated as an "
                "uncertain image-only result, not as a confirmed normal finding."
            )
        else:
            predictions.sort(key=lambda x: x["probability"], reverse=True)
            predicted_class_name = ", ".join([p["class"] for p in predictions])
            confidence = predictions[0]["probability"]
            detected_predictions = predictions
            finding_detected = True
            severity = "moderate"
            urgency = "routine"
            explanation = "Image analysis detected one or more modeled findings above the decision threshold."

        # Generate Grad-CAM only for a positive finding. Showing Grad-CAM for a
        # below-threshold class makes the heatmap look like a false diagnosis.
        if finding_detected:
            predicted_class_idx = int(np.argmax(probs))
            gradcam_path = self._generate_gradcam(
                tensor, preprocessed["original_image"], predicted_class_idx
            )
        else:
            gradcam_path = ""

        result = {
            "predicted_class": predicted_class_name,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "detected_predictions": detected_predictions,
            "finding_detected": finding_detected,
            "abnormality_score": max_probability,
            "decision_threshold": top_class_threshold,
            "class_thresholds": {
                self.class_names[i]: float(self.class_thresholds[i])
                for i in range(self.num_classes)
            },
            "explanation": explanation,
            "severity": severity,
            "urgency": urgency,
            "gradcam_path": gradcam_path,
            "image_metadata": preprocessed["metadata"],
            "input_type": "image",
        }

        cv_logger.info(
            f"CV prediction: {result['predicted_class']} "
            f"(confidence={confidence:.3f})"
        )
        return result

    def _generate_gradcam(self, tensor: torch.Tensor,
                          original_image: Image.Image,
                          class_idx: int) -> str:
        """Generate and save Grad-CAM visualization."""
        try:
            # Clone and enable gradients for Grad-CAM
            tensor_with_grad = tensor.clone().detach().requires_grad_(True)
            heatmap = self.grad_cam.generate(tensor_with_grad, class_idx)

            # Resize heatmap to original image size
            heatmap_resized = np.array(
                Image.fromarray(np.uint8(heatmap * 255)).resize(
                    original_image.size, Image.LANCZOS
                )
            ) / 255.0

            # Create overlay
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
            axes[0].axis("off")

            # Grad-CAM heatmap
            axes[1].imshow(heatmap_resized, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold")
            axes[1].axis("off")

            # Overlay
            original_array = np.array(original_image.convert("RGB")) / 255.0
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
            plt.close(fig)

            cv_logger.info(f"Grad-CAM saved to {save_path}")
            return save_path

        except Exception as e:
            cv_logger.error(f"Grad-CAM generation failed: {e}")
            return ""
