"""
Safety & Risk Control module.
Evaluates diagnosis confidence, detects contradictions and emergency conditions,
and determines whether output is safe to present or requires doctor consultation.
"""

from app.config import settings
from app.utils.logger import safety_logger

# Red-flag conditions that always require urgent medical attention
RED_FLAG_KEYWORDS = [
    "myocardial infarction", "heart attack", "stroke", "cerebrovascular",
    "pulmonary embolism", "anaphylaxis", "sepsis", "meningitis",
    "cardiac arrest", "respiratory failure", "hemorrhage", "bleeding",
    "seizure", "unconscious", "critical", "emergency", "severe",
    "life-threatening", "acute abdomen", "tension pneumothorax",
]


class SafetyController:
    """Evaluates safety of diagnosis output and controls release."""

    def __init__(self):
        self.confidence_threshold_cv = settings.confidence_threshold_cv
        self.confidence_threshold_nlp = settings.confidence_threshold_nlp
        self.safety_threshold = settings.safety_threshold
        safety_logger.info(
            f"SafetyController initialized "
            f"(cv_threshold={self.confidence_threshold_cv}, "
            f"nlp_threshold={self.confidence_threshold_nlp})"
        )

    def evaluate(self, validated_output: dict) -> dict:
        """
        Evaluate the safety of a validated diagnosis output.

        Args:
            validated_output: Output from GuidelineValidator

        Returns:
            SafetyResult with approval status and risk factors
        """
        risk_factors = []
        is_safe = True

        confidence = validated_output.get("combined_confidence", 0)
        primary = validated_output.get("primary_diagnosis", "").lower()
        severity = validated_output.get("severity", "moderate")
        urgency = validated_output.get("urgency", "routine")
        modality = validated_output.get("modality", "unknown")

        # Chọn ngưỡng tin cậy theo phân loại
        if modality == "nlp":
            threshold = self.confidence_threshold_nlp
        else:
            # Cho ảnh và đa phương thức
            threshold = self.confidence_threshold_cv
        threshold = validated_output.get("confidence_threshold", threshold)

        # Check 1: Confidence threshold
        if confidence < threshold:
            risk_factors.append(
                f"Low confidence ({confidence:.0%}) below threshold "
                f"({threshold:.0%})"
            )
            is_safe = False

        # Check 2: Guideline consistency
        if not validated_output.get("is_guideline_consistent", True):
            risk_factors.append("Diagnosis not fully consistent with clinical guidelines")
            is_safe = False

        # Check 3: Conflict detection
        if validated_output.get("conflict_flag"):
            risk_factors.append("Conflict between NLP and CV modalities detected")
            is_safe = False

        # Check 4: Red-flag conditions
        red_flags_found = [kw for kw in RED_FLAG_KEYWORDS if kw in primary]
        if red_flags_found:
            risk_factors.append(
                f"Red-flag conditions detected: {', '.join(red_flags_found)}"
            )
            # Red flags ALWAYS require professional review and emergency consultation
            is_safe = False

        # Check 5: Severity/urgency — only a warning note, not a blocker
        warning_notes = []
        if severity in ("high", "critical") or urgency in ("urgent", "emergency"):
            warning_notes.append(
                f"⚠️ High severity ({severity}) / urgency ({urgency}) — seek medical attention"
            )

        # Determine safety status
        if is_safe:
            safety_status = "approved"
            warning_str = (
                " Note: " + "; ".join(warning_notes) + "."
                if warning_notes else ""
            )
            safety_message = (
                "✅ Output approved. This analysis has passed safety checks."
                + warning_str
                + " Always consult a qualified healthcare professional for medical decisions."
            )
        else:
            safety_status = "fallback"
            all_issues = risk_factors + warning_notes
            safety_message = (
                "⚠️ CONSULT A DOCTOR: This analysis has been flagged for "
                "professional review due to: " + "; ".join(all_issues) + ". "
                "Please seek immediate medical consultation."
            )

        safety_result = {
            **validated_output,
            "safety_status": safety_status,
            "is_approved": is_safe,
            "risk_factors": risk_factors,
            "safety_message": safety_message,
            "confidence_threshold_used": threshold,
        }

        safety_logger.info(
            f"Safety evaluation: status={safety_status}, "
            f"risks={len(risk_factors)}"
        )
        return safety_result
