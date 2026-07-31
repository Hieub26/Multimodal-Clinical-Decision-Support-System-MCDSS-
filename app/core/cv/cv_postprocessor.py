"""
CV Postprocessor: Converts raw model probabilities into clinically structured predictions.
Handles threshold-based detection, negative screening, severity/urgency derivation,
and result assembly.
"""

import numpy as np
from app.config import settings
from app.utils.logger import cv_logger


# Clinically-grounded risk map for CXR findings.
# Maps finding class names to (severity, urgency) tuples.
# Any finding not listed defaults to ("moderate", "routine").
FINDING_RISK_MAP: dict[str, tuple[str, str]] = {
    "Pneumonia":          ("high",     "urgent"),
    "Consolidation":      ("high",     "urgent"),
    "Infiltration":       ("moderate", "urgent"),
    "Pneumothorax":       ("critical", "emergency"),
    "Edema":              ("critical", "emergency"),
    "Pleural_Thickening": ("moderate", "routine"),
    "Effusion":           ("high",     "urgent"),
    "Atelectasis":        ("moderate", "routine"),
    "Cardiomegaly":       ("high",     "urgent"),
    "Mass":               ("high",     "urgent"),
    "Nodule":             ("moderate", "routine"),
    "Hernia":             ("moderate", "routine"),
    "Emphysema":          ("moderate", "urgent"),
    "Fibrosis":           ("moderate", "routine"),
}

# Ranking order: higher index = more severe
_SEVERITY_RANK: dict[str, int] = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
_URGENCY_RANK: dict[str, int] = {"routine": 0, "urgent": 1, "emergency": 2}


class CVPostprocessor:
    """Converts raw sigmoid probabilities into structured clinical prediction results."""

    def postprocess(
        self,
        probs: np.ndarray,
        class_names: list[str],
        class_thresholds: list[float],
    ) -> dict:
        """Transform raw probabilities into a structured prediction result.

        Args:
            probs: 1-D array of per-class sigmoid probabilities.
            class_names: Ordered list of class label names.
            class_thresholds: Per-class decision thresholds.

        Returns:
            Dictionary with prediction details, severity, urgency, and explanation.
        """
        num_classes = len(probs)

        # Top-K predictions sorted by probability
        top_indices = np.argsort(probs)[::-1][:settings.cv_top_k]
        top_predictions = [
            {
                "class": class_names[i],
                "probability": float(probs[i]),
                "threshold": float(class_thresholds[i]),
            }
            for i in top_indices
        ]

        # Detected predictions (above per-class thresholds)
        predictions = [
            {
                "class": class_names[i],
                "probability": float(probs[i]),
                "threshold": float(class_thresholds[i]),
            }
            for i in range(num_classes)
            if probs[i] >= class_thresholds[i]
        ]

        max_probability = float(np.max(probs))
        top_class_idx = int(np.argmax(probs))
        top_class_threshold = float(class_thresholds[top_class_idx])
        threshold_ratios = probs / np.maximum(np.array(class_thresholds), 1e-6)
        max_threshold_ratio = float(np.max(threshold_ratios))

        # Multi-label formatting
        if len(predictions) == 0:
            detected_predictions = []
            finding_detected = False
            negative_screen = (
                max_threshold_ratio <= settings.cv_negative_screen_ratio_threshold
            )
            if negative_screen:
                predicted_class_name = "No confident abnormal finding detected"
                confidence = min(0.95, max(0.0, 1.0 - max_threshold_ratio))
                severity = "low"
                explanation = (
                    "The image model did not find any of its monitored disease "
                    "classes near the validated decision thresholds. This is a "
                    "negative screening result for the modeled conditions, not a "
                    "guarantee that the chest X-ray is completely normal."
                )
                recommended_actions = [
                    "Correlate with symptoms and clinical history",
                    "Seek medical care if symptoms are present, persistent, or worsening",
                ]
            else:
                predicted_class_name = "No confident CV finding"
                confidence = max_probability
                severity = "unknown"
                explanation = (
                    "The CV model did not produce any class probability above the "
                    "configured decision threshold, but at least one class is close "
                    "enough to require cautious review. Treat this as an uncertain "
                    "image-only result, not as a confirmed normal finding."
                )
                recommended_actions = [
                    "Consider clinician review or repeat imaging if clinically indicated",
                    "Correlate with symptoms and clinical history",
                ]
            urgency = "routine"
        else:
            predictions.sort(key=lambda x: x["probability"], reverse=True)
            predicted_class_name = ", ".join([p["class"] for p in predictions])
            confidence = predictions[0]["probability"]
            detected_predictions = predictions
            finding_detected = True
            negative_screen = False

            # Derive severity/urgency from detected findings dynamically
            severity, urgency = self._severity_from_findings(predictions)

            explanation = "Image analysis detected one or more modeled findings above the decision threshold."
            recommended_actions = []

        result = {
            "predicted_class": predicted_class_name,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "detected_predictions": detected_predictions,
            "finding_detected": finding_detected,
            "negative_screen": negative_screen,
            "abnormality_score": max_probability,
            "max_threshold_ratio": max_threshold_ratio,
            "decision_threshold": (
                settings.safety_threshold if negative_screen else top_class_threshold
            ),
            "positive_decision_threshold": top_class_threshold,
            "class_thresholds": {
                class_names[i]: float(class_thresholds[i])
                for i in range(num_classes)
            },
            "explanation": explanation,
            "severity": severity,
            "urgency": urgency,
            "recommended_actions": recommended_actions,
        }

        cv_logger.info(
            f"CV postprocess: {result['predicted_class']} "
            f"(confidence={confidence:.3f})"
        )
        return result

    @staticmethod
    def _severity_from_findings(predictions: list[dict]) -> tuple[str, str]:
        """Derive severity and urgency from the detected CV findings.

        Uses the highest-risk finding across all detected classes.
        Falls back to ("moderate", "routine") when no mapping is found.
        """
        best_severity = "moderate"
        best_urgency = "routine"

        for pred in predictions:
            cls_name = pred.get("class", "")
            sev, urg = FINDING_RISK_MAP.get(cls_name, ("moderate", "routine"))
            if _SEVERITY_RANK.get(sev, 0) > _SEVERITY_RANK.get(best_severity, 0):
                best_severity = sev
            if _URGENCY_RANK.get(urg, 0) > _URGENCY_RANK.get(best_urgency, 0):
                best_urgency = urg

        return best_severity, best_urgency
