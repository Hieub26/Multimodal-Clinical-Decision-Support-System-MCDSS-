"""
CV Model: DenseNet-121 based medical image classifier.
Thin orchestrator that delegates preprocessing, postprocessing,
uncertainty analysis, and Grad-CAM to dedicated modules via Dependency Injection.
"""

import os
import json
import threading
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

from app.config import settings
from app.core.cv.image_preprocessor import ImagePreprocessor
from app.core.cv.cv_postprocessor import CVPostprocessor
from app.core.cv.uncertainty_analyzer import UncertaintyAnalyzer
from app.core.cv.grad_cam import GradCAM, GradCAMVisualizer
from app.utils.logger import cv_logger


class ModelWeightsNotFoundError(RuntimeError):
    """Raised when the fine-tuned CV weights file is missing."""


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
        # Requests run in worker threads. Grad-CAM keeps per-call state on the
        # model hooks and matplotlib is not thread-safe, so run one at a time.
        self._predict_lock = threading.Lock()
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
                if list(labels) != list(self.class_names):
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
        """Build DenseNet-121 with custom classification head.

        Raises:
            ModelWeightsNotFoundError: if the fine-tuned weights are missing.
                Without them the classifier head is random, so refusing to
                predict is the only safe behaviour for a diagnostic output.
        """
        model_path = settings.cv_model_path
        if not os.path.exists(model_path):
            raise ModelWeightsNotFoundError(
                f"No fine-tuned CV weights found at {model_path}. "
                "Set CV_MODEL_PATH to the trained DenseNet-121 checkpoint."
            )

        # The checkpoint holds every layer, so no ImageNet download is needed
        model = models.densenet121(weights=None)

        # Replace final FC layer for our classes
        # Modified to match the single Linear layer in the provided weights file
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, self.num_classes)

        cv_logger.info(f"Loading model weights from {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        cv_logger.info("Fine-tuned weights loaded successfully")

        return model

    def predict(self, image_input) -> dict:
        """
        Run inference on a medical image.

        Args:
            image_input: PIL Image, file path, or bytes

        Returns:
            CVDiagnosis dictionary
        """
        with self._predict_lock:
            return self._predict(image_input)

    def _predict(self, image_input) -> dict:
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

        # Step 3: Postprocess predictions 
        result = self.postprocessor.postprocess(
            probs, self.class_names, self.class_thresholds
        )

        # Step 4: Generate Grad-CAM only for positive findings 
        if result["finding_detected"]:
            # Explain the finding that is reported, not the highest raw
            # probability (which may sit below its own threshold)
            gradcam_path = self.grad_cam_visualizer.generate_and_save(
                tensor,
                preprocessed["original_image"],
                result["primary_class_index"],
                content_box=preprocessed["content_box"],
            )
        else:
            gradcam_path = ""

        # Step 5: Compute uncertainty 
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
