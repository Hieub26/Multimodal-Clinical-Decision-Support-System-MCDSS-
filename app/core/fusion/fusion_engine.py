"""
Fusion Engine: Combines NLP and CV diagnosis results
using weighted strategies for concordant, complementary, and discordant cases.
"""

from app.utils.logger import fusion_logger
from app.config import settings

class FusionEngine:
    """Fuses NLP-based and CV-based diagnoses into a combined result."""

    # Weight allocation for each modality (CV is stronger for X-ray diagnosis)
    NLP_WEIGHT = 0.3
    CV_WEIGHT = 0.7

    # Ontology mapping for more robust concordant matching
    ONTOLOGY_MAPPING = {
        "pneumonia": ["infiltration", "consolidation", "pneumonia", "lung infection"],
        "effusion": ["effusion", "pleural_thickening", "fluid"],
        "cardiomegaly": ["cardiomegaly", "enlarged heart", "heart"],
        "nodule": ["nodule", "mass", "lesion", "tumor"],
        "edema": ["edema", "fluid in lungs", "pulmonary edema"],
    }

    def __init__(self):
        fusion_logger.info("FusionEngine initialized")

    def fuse(self, nlp_diagnosis: dict = None, cv_diagnosis: dict = None) -> dict:
        """
        Fuse NLP and CV diagnoses into a combined diagnosis.

        Args:
            nlp_diagnosis: NLP pipeline output (or None)
            cv_diagnosis: CV pipeline output (or None)

        Returns:
            Combined diagnosis dictionary
        """
        # Single modality cases
        if nlp_diagnosis and not cv_diagnosis:
            return self._single_modality_result(nlp_diagnosis, "nlp")
        if cv_diagnosis and not nlp_diagnosis:
            return self._single_modality_result(cv_diagnosis, "cv")
        if not nlp_diagnosis and not cv_diagnosis:
            return self._empty_result()

        if nlp_diagnosis and cv_diagnosis and self._is_low_signal_nlp(nlp_diagnosis):
            result = self._single_modality_result(cv_diagnosis, "cv")
            result["nlp_diagnosis"] = nlp_diagnosis
            result["fusion_strategy"] = "cv_primary"
            result["explanation"] = (
                result.get("explanation", "")
                + " Low-confidence text analysis was ignored for safety."
            ).strip()
            return result

        # Multimodal fusion
        fusion_logger.info("Performing multimodal fusion")

        nlp_conf = nlp_diagnosis.get("confidence", 0)
        cv_conf = cv_diagnosis.get("confidence", 0)
        nlp_primary = nlp_diagnosis.get("primary_diagnosis", "")
        cv_primary = cv_diagnosis.get("predicted_class", "")

        # Determine fusion strategy
        strategy = self._determine_strategy(nlp_primary, cv_primary, nlp_conf, cv_conf)

        if strategy == "concordant":
            result = self._fuse_concordant(nlp_diagnosis, cv_diagnosis)
        elif strategy == "complementary":
            result = self._fuse_complementary(nlp_diagnosis, cv_diagnosis)
        else:
            result = self._fuse_discordant(nlp_diagnosis, cv_diagnosis)

        result["fusion_strategy"] = strategy
        result["nlp_diagnosis"] = nlp_diagnosis
        result["cv_diagnosis"] = cv_diagnosis
        result["modality"] = "multimodal"

        # Missing uncertainty reasoning logic (Confidence Gap)
        confidence_gap = abs(nlp_conf - cv_conf)
        if confidence_gap > 0.4:
            result["explanation"] += f"\n\n⚠️ Note: There is a significant confidence gap ({confidence_gap:.2f}) between text and image analyses, indicating potential uncertainty."
            result["uncertainty_flag"] = True

        fusion_logger.info(
            f"Fusion complete: strategy={strategy}, "
            f"confidence={result['combined_confidence']:.3f}, gap={confidence_gap:.3f}"
        )
        return result

    def _is_low_signal_nlp(self, nlp_diagnosis: dict) -> bool:
        """Detect generic fallback NLP output that should not dilute CV results."""
        confidence = nlp_diagnosis.get("confidence", 0)
        primary = nlp_diagnosis.get("primary_diagnosis", "").lower()
        evidence = nlp_diagnosis.get("supporting_evidence", [])

        generic_outputs = {
            "general clinical assessment needed",
            "unable to provide diagnosis",
            "analysis completed",
        }

        return (
            confidence < settings.confidence_threshold_nlp
            and (primary in generic_outputs or not evidence)
        )

    def _determine_strategy(self, nlp_primary: str, cv_primary: str,
                            nlp_conf: float, cv_conf: float) -> str:
        """Determine the appropriate fusion strategy."""
        nlp_lower = nlp_primary.lower()
        cv_lower = cv_primary.lower()

        # Check for agreement using ontology mapping
        is_concordant = False
        for key, related_terms in self.ONTOLOGY_MAPPING.items():
            if key in cv_lower:
                if any(term in nlp_lower for term in related_terms):
                    is_concordant = True
                    break

        # Fallback check
        if not is_concordant and any(term in nlp_lower for term in cv_lower.replace(",", " ").split()):
            is_concordant = True

        if is_concordant:
            return "concordant"

        # Conflict happens when both modalities pass their own confidence threshold
        if nlp_conf >= settings.confidence_threshold_nlp and cv_conf >= settings.confidence_threshold_cv:
            return "discordant"

        return "complementary"

    def _fuse_concordant(self, nlp_diag: dict, cv_diag: dict) -> dict:
        """Both modalities agree — boost confidence safely."""
        nlp_conf = nlp_diag.get("confidence", 0)
        cv_conf = cv_diag.get("confidence", 0)
        question_answer = self._clean_question_answer(
            nlp_diag.get("question_answer", "")
        )

        # Boost confidence safely without breaking calibration
        combined_conf = min(1.0, max(nlp_conf, cv_conf) + 0.05)
        explanation = (
            f"Both text analysis and image analysis agree. "
            f"NLP suggests '{nlp_diag.get('primary_diagnosis', '')}' "
            f"(confidence: {nlp_conf:.0%}), "
            f"CV confirms '{cv_diag.get('predicted_class', '')}' "
            f"(confidence: {cv_conf:.0%}). "
            f"Combined confidence is boosted due to agreement."
        )

        return {
            "primary_diagnosis": nlp_diag.get("primary_diagnosis", ""),
            "combined_confidence": combined_conf,
            "agreement": "concordant",
            "explanation": explanation,
            "question_answer": question_answer,
            "supporting_evidence": nlp_diag.get("supporting_evidence", []),
            "recommended_actions": nlp_diag.get("recommended_actions", []),
            "severity": nlp_diag.get("severity", "moderate"),
            "urgency": nlp_diag.get("urgency", "routine"),
        }

    def _fuse_complementary(self, nlp_diag: dict, cv_diag: dict) -> dict:
        """Modalities provide different but non-conflicting information."""
        nlp_conf = nlp_diag.get("confidence", 0)
        cv_conf = cv_diag.get("confidence", 0)
        question_answer = self._clean_question_answer(
            nlp_diag.get("question_answer", "")
        )

        combined_conf = nlp_conf * self.NLP_WEIGHT + cv_conf * self.CV_WEIGHT
        explanation = (
            f"Text and image analyses provide complementary insights. "
            f"NLP analysis: '{nlp_diag.get('primary_diagnosis', '')}' ({nlp_conf:.0%}). "
            f"Image analysis: '{cv_diag.get('predicted_class', '')}' ({cv_conf:.0%}). "
            f"Both perspectives are considered in the combined assessment."
        )

        return {
            "primary_diagnosis": (
                nlp_diag.get("primary_diagnosis", "")
                if nlp_conf >= cv_conf
                else cv_diag.get("predicted_class", "")
            ),
            "combined_confidence": combined_conf,
            "agreement": "complementary",
            "explanation": explanation,
            "question_answer": question_answer,
            "supporting_evidence": nlp_diag.get("supporting_evidence", []),
            "recommended_actions": nlp_diag.get("recommended_actions", []),
            "severity": nlp_diag.get("severity", "moderate"),
            "urgency": nlp_diag.get("urgency", "routine"),
        }

    def _fuse_discordant(self, nlp_diag: dict, cv_diag: dict) -> dict:
        """Modalities conflict — flag for review but retain strong signals."""
        nlp_conf = nlp_diag.get("confidence", 0)
        cv_conf = cv_diag.get("confidence", 0)
        question_answer = self._clean_question_answer(
            nlp_diag.get("question_answer", "")
        )

        # Industry-style handling: Retain the strongest signal with a slight penalty
        combined_conf = max(nlp_conf, cv_conf) * 0.85
        explanation = (
            "CONFLICT DETECTED: Text analysis suggests "
            f"'{nlp_diag.get('primary_diagnosis', '')}' ({nlp_conf:.0%}), "
            f"but image analysis suggests '{cv_diag.get('predicted_class', '')}' "
            f"({cv_conf:.0%}). Professional review is recommended to resolve "
            "the discrepancy."
        )

        return {
            "primary_diagnosis": (
                f"CONFLICTING: NLP='{nlp_diag.get('primary_diagnosis', '')}' "
                f"vs CV='{cv_diag.get('predicted_class', '')}'"
            ),
            "combined_confidence": combined_conf,
            "agreement": "discordant",
            "explanation": (
                f"⚠️ CONFLICT DETECTED: Text analysis suggests "
                f"'{nlp_diag.get('primary_diagnosis', '')}' ({nlp_conf:.0%}), "
                f"but image analysis suggests '{cv_diag.get('predicted_class', '')}' "
                f"({cv_conf:.0%}). "
                f"Professional review is recommended to resolve the discrepancy."
            ),
            "question_answer": question_answer,
            "conflict_flag": True,
            "supporting_evidence": nlp_diag.get("supporting_evidence", []),
            "recommended_actions": nlp_diag.get("recommended_actions", []),
            "severity": "moderate",
            "urgency": "follow_up",
        }

    def _single_modality_result(self, diagnosis: dict, modality: str) -> dict:
        """Wrap a single-modality result."""
        conf = diagnosis.get("confidence", 0)
        primary = diagnosis.get(
            "primary_diagnosis",
            diagnosis.get("predicted_class", "Unknown"),
        )
        return {
            "primary_diagnosis": primary,
            "combined_confidence": conf,
            "confidence_threshold": diagnosis.get(
                "decision_threshold",
                settings.confidence_threshold_nlp
                if modality == "nlp"
                else settings.confidence_threshold_cv,
            ),
            "agreement": "single_modality",
            "fusion_strategy": "single",
            "explanation": diagnosis.get("explanation", f"Analysis based on {modality.upper()} modality only."),
            "question_answer": diagnosis.get("question_answer", ""),
            "recommended_actions": diagnosis.get("recommended_actions", []),
            "nlp_diagnosis": diagnosis if modality == "nlp" else None,
            "cv_diagnosis": diagnosis if modality == "cv" else None,
            "modality": modality,
            "severity": diagnosis.get("severity", "moderate"),
            "urgency": diagnosis.get("urgency", "routine"),
        }

    def _clean_question_answer(self, answer: str) -> str:
        """Return a display-worthy clinical question answer."""
        if not answer:
            return ""
        answer = str(answer).strip()
        if answer.upper() in {"N/A", "NA", "NONE", "NULL"}:
            return ""
        return answer

    def _empty_result(self) -> dict:
        return {
            "primary_diagnosis": "No input provided",
            "combined_confidence": 0.0,
            "agreement": "none",
            "fusion_strategy": "none",
            "explanation": "No diagnosis data available.",
            "modality": "none",
            "severity": "unknown",
            "urgency": "routine",
        }
