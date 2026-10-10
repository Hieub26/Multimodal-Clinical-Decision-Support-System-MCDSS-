"""
CV Postprocessor: Converts raw model probabilities into clinically structured predictions.
Handles threshold-based detection, negative screening, severity/urgency derivation,
and result assembly.
"""

import numpy as np
from app.config import settings
from app.utils.logger import cv_logger


# Clinically-grounded risk map for CXR findings.
# Maps finding class names to (severity, urgency) tuples: what the finding
# means if it is really there. How far a flag can be trusted is a separate
# matter, see MIN_PRECISION_FOR_CRITICAL.
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

# The model's score is not the chance that a flag is right: it was trained
# with heavy positive-class weights, and on its validation set a pulmonary
# edema flag at a score above 0.92 was a true finding 17% of the time. A flag
# is therefore reported as critical or an emergency only when the validation
# figures back it: the score reaches the threshold those figures were measured
# at, and at least this share of flags there were true findings. Any other
# flag is capped at high / urgent, which still asks for prompt review.
MIN_PRECISION_FOR_CRITICAL = 0.25
_UNCONFIRMED_FLAG_CAP = ("high", "urgent")


class CVPostprocessor:
    """Converts raw sigmoid probabilities into structured clinical prediction results."""

    def postprocess(
        self,
        probs: np.ndarray,
        class_names: list[str],
        class_thresholds: list[float],
        validation_metrics: list[dict] | None = None,
    ) -> dict:
        """Transform raw probabilities into a structured prediction result.

        Args:
            probs: 1-D array of per-class sigmoid probabilities.
            class_names: Ordered list of class label names.
            class_thresholds: Per-class decision thresholds (possibly relaxed
                below the validated ones, see CV_THRESHOLD_RELAXATION).
            validation_metrics: Per class, the threshold the model was
                validated at and the precision and recall measured there
                ({"threshold", "precision", "recall"}), or None when the model
                metadata does not provide them.

        Returns:
            Dictionary with prediction details, severity, urgency, and explanation.
        """
        num_classes = len(probs)

        def describe(i: int) -> dict:
            prediction = {
                "class": class_names[i],
                "probability": float(probs[i]),
                "threshold": float(class_thresholds[i]),
            }
            if validation_metrics:
                validated = validation_metrics[i]
                prediction.update({
                    "validated_threshold": validated["threshold"],
                    "above_validated_threshold": bool(probs[i] >= validated["threshold"]),
                    "validated_precision": validated["precision"],
                    "validated_recall": validated["recall"],
                })
            return prediction

        # Top-K predictions sorted by probability
        top_indices = np.argsort(probs)[::-1][:settings.cv_top_k]
        top_predictions = [describe(i) for i in top_indices]

        # Detected predictions (above per-class thresholds)
        predictions = [
            describe(i) for i in range(num_classes)
            if probs[i] >= class_thresholds[i]
        ]

        max_probability = float(np.max(probs))
        # Class the decision is about: the strongest detected finding, or the
        # highest raw probability when nothing crossed its threshold. These
        # differ because thresholds are per-class (a 0.75 Emphysema can be
        # below its threshold while a 0.50 Infiltration is above its own).
        primary_class_idx = int(np.argmax(probs))
        top_class_threshold = float(class_thresholds[primary_class_idx])
        threshold_ratios = probs / np.maximum(np.array(class_thresholds), 1e-6)
        max_threshold_ratio = float(np.max(threshold_ratios))
        reliability = None

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
                    + self._sensitivity_note(validation_metrics)
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
            primary_class_idx = list(class_names).index(predictions[0]["class"])
            top_class_threshold = predictions[0]["threshold"]
            detected_predictions = predictions
            finding_detected = True
            negative_screen = False

            # Derive severity/urgency from detected findings dynamically
            severity, urgency = self._severity_from_findings(predictions)
            reliability = self._reliability(predictions[0])

            explanation = (
                "Image analysis detected one or more modeled findings above the "
                "decision threshold."
                + (f" {reliability['note']}" if reliability else "")
            )
            recommended_actions = []

        result = {
            "predicted_class": predicted_class_name,
            "primary_class_index": primary_class_idx,
            # The model's score for the reported class. It decides whether the
            # class is flagged; how often such a flag is right is in
            # "reliability".
            "confidence": confidence,
            "reliability": reliability,
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

    @classmethod
    def _severity_from_findings(cls, predictions: list[dict]) -> tuple[str, str]:
        """Derive severity and urgency from the detected CV findings.

        Uses the highest-risk finding across all detected classes.
        Falls back to ("moderate", "routine") when no mapping is found.
        """
        best_severity = "moderate"
        best_urgency = "routine"

        for pred in predictions:
            cls_name = pred.get("class", "")
            sev, urg = FINDING_RISK_MAP.get(cls_name, ("moderate", "routine"))
            if not cls._is_confirmed_by_validation(pred):
                cap_severity, cap_urgency = _UNCONFIRMED_FLAG_CAP
                if _SEVERITY_RANK[sev] > _SEVERITY_RANK[cap_severity]:
                    sev = cap_severity
                if _URGENCY_RANK[urg] > _URGENCY_RANK[cap_urgency]:
                    urg = cap_urgency
            if _SEVERITY_RANK.get(sev, 0) > _SEVERITY_RANK.get(best_severity, 0):
                best_severity = sev
            if _URGENCY_RANK.get(urg, 0) > _URGENCY_RANK.get(best_urgency, 0):
                best_urgency = urg

        return best_severity, best_urgency

    @staticmethod
    def _is_confirmed_by_validation(prediction: dict) -> bool:
        """Whether the validation figures back treating this flag at face value.

        Without validation figures there is nothing to judge by, and the flag
        keeps the risk its class carries.
        """
        if "validated_threshold" not in prediction:
            return True
        if not prediction["above_validated_threshold"]:
            return False
        precision = prediction["validated_precision"]
        return precision is None or precision >= MIN_PRECISION_FOR_CRITICAL

    @classmethod
    def _reliability(cls, prediction: dict) -> dict | None:
        """How far the reported flag can be trusted, from the validation set."""
        precision = prediction.get("validated_precision")
        if "validated_threshold" not in prediction or precision is None:
            return None

        name = prediction["class"].replace("_", " ")
        if prediction["above_validated_threshold"]:
            note = (
                f"The model's score is not the chance that it is right: on its "
                f"validation set, {precision:.0%} of the {name} flags raised "
                f"at the validated threshold were true findings. Treat it as "
                f"a prompt for review, not as a confirmed finding."
            )
        else:
            note = (
                f"The model's score is not the chance that it is right, and "
                f"for {name} it is below the threshold the model was "
                f"validated at ({prediction['validated_threshold']:.2f}). At "
                f"that threshold {precision:.0%} of flags were true findings; "
                f"below it the share is lower and was not measured."
            )
        return {
            "class": prediction["class"],
            "validated_precision": precision,
            "validated_recall": prediction.get("validated_recall"),
            "above_validated_threshold": prediction["above_validated_threshold"],
            "confirmed_by_validation": cls._is_confirmed_by_validation(prediction),
            "note": note,
        }

    @staticmethod
    def _sensitivity_note(validation_metrics: list[dict] | None) -> str:
        """What a negative screen is worth, from the validated recall."""
        recalls = [
            m["recall"] for m in validation_metrics or [] if m.get("recall") is not None
        ]
        if not recalls:
            return ""
        return (
            f" At its validated thresholds the model finds between "
            f"{min(recalls):.0%} and {max(recalls):.0%} of the cases of each "
            f"condition, so a negative result does not rule any of them out."
        )
