"""
CV Model: DenseNet-121 based medical image classifier.
Thin orchestrator that delegates preprocessing, postprocessing,
uncertainty analysis, and Grad-CAM to dedicated modules via Dependency Injection.
"""

import os
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models

from app.config import settings
from app.core.cv.image_preprocessor import ImagePreprocessor
from app.core.cv.cv_postprocessor import CVPostprocessor
from app.core.cv.uncertainty_analyzer import UncertaintyAnalyzer
from app.core.cv.grad_cam import GradCAM, GradCAMVisualizer
from app.utils.logger import cv_logger


class MedicalCVModel:
    """Medical image classification model with modular architecture.

    Delegates each concern to a dedicated module:
    - ImagePreprocessor: image loading, validation, resize+pad, normalization
    - CVPostprocessor: threshold logic, severity/urgency, result formatting
    - UncertaintyAnalyzer: entropy + margin uncertainty metrics
    - GradCAMVisualizer: explainability heatmap generation
    """

    def __init__(
        self,
        image_preprocessor: ImagePreprocessor | None = None,
        postprocessor: CVPostprocessor | None = None,
        uncertainty_analyzer: UncertaintyAnalyzer | None = None,
    ):
        """Initialize MedicalCVModel with Dependency Injection.

        Args:
            image_preprocessor: Optional injected ImagePreprocessor instance.
            postprocessor: Optional injected CVPostprocessor instance.
            uncertainty_analyzer: Optional injected UncertaintyAnalyzer instance.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = settings.cv_num_classes
        self.class_names = settings.cv_class_names
        self.class_thresholds = [settings.confidence_threshold_cv] * self.num_classes
        self._load_model_metadata()

        # Dependency Injection
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()
        self.postprocessor = postprocessor or CVPostprocessor()
        self.uncertainty_analyzer = uncertainty_analyzer or UncertaintyAnalyzer()

        self._model = None
        self._grad_cam = None
        self._grad_cam_visualizer = None
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
    def grad_cam_visualizer(self) -> GradCAMVisualizer:
        """Lazy-load Grad-CAM and its visualizer."""
        if self._grad_cam_visualizer is None:
            target_layer = self.model.features.denseblock4
            self._grad_cam = GradCAM(self.model, target_layer)
            self._grad_cam_visualizer = GradCAMVisualizer(
                self._grad_cam, self.class_names
            )
        return self._grad_cam_visualizer

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

        # Step 1: Preprocess image (delegated to ImagePreprocessor)
        preprocessed = self.image_preprocessor.preprocess(image_input)
        raw_tensor = preprocessed["tensor"]
        if isinstance(raw_tensor, torch.Tensor):
            tensor = raw_tensor.to(self.device)
        else:
            tensor = torch.from_numpy(raw_tensor).to(self.device)

        # Step 2: Inference
        with torch.no_grad():
            output = self.model(tensor)
            temperature = max(float(settings.cv_logit_temperature), 1e-6)
            scaled_logits = output[0] / temperature
            probabilities = torch.sigmoid(scaled_logits)

        probs = probabilities.cpu().numpy()

        # Step 3: Postprocess predictions (delegated to CVPostprocessor)
        result = self.postprocessor.postprocess(
            probs, self.class_names, self.class_thresholds
        )

        # Step 4: Generate Grad-CAM only for positive findings (delegated to GradCAMVisualizer)
        if result["finding_detected"]:
            predicted_class_idx = int(np.argmax(probs))
            gradcam_path = self.grad_cam_visualizer.generate_and_save(
                tensor, preprocessed["original_image"], predicted_class_idx
            )
        else:
            gradcam_path = ""

        # Step 5: Compute uncertainty (delegated to UncertaintyAnalyzer)
        uncertainty = self.uncertainty_analyzer.compute(
            probs, self.class_names, self.class_thresholds
        )

        # Step 6: Assemble final result
        result["uncertainty"] = uncertainty
        result["gradcam_path"] = gradcam_path
        result["image_metadata"] = preprocessed["metadata"]
        result["input_type"] = "image"

        cv_logger.info(
            f"CV prediction: {result['predicted_class']} "
            f"(confidence={result['confidence']:.3f})"
        )
        return result
