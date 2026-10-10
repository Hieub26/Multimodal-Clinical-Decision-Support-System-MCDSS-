"""
Fusion Engine: Combines NLP and CV diagnosis results
using weighted strategies for concordant, complementary, and discordant cases.
"""

import re

from app.utils.logger import fusion_logger
from app.config import settings

class FusionEngine:
    """Fuses NLP-based and CV-based diagnoses into a combined result."""

    # Weight allocation for each modality (CV is stronger for X-ray diagnosis)
    NLP_WEIGHT = 0.3
    CV_WEIGHT = 0.7

    # Concordant fusion confidence cap — prevents overconfidence in medical AI
    CONCORDANT_CONFIDENCE_CAP = 0.95

    # Uncertainty penalty: reduce confidence when CV entropy is high.
    # Resolves the contradiction of "94% confidence + entropy_mean=0.77".
    # penalty = entropy_mean * UNCERTAINTY_PENALTY_FACTOR
    UNCERTAINTY_PENALTY_FACTOR = 0.08

    # Severity formula weights — configurable instead of hardcoded.
    # Recruiter question: "why symptom 50%?" → because symptom presentation
    # is the primary clinical driver, imaging confirms/refines.
    SEVERITY_FORMULA_WEIGHTS = {
        "symptom": 0.5,
        "image": 0.3,
        "risk": 0.2,
    }

    # Severity scoring: maps NLP severity labels to numeric weights
    SEVERITY_WEIGHT_MAP = {
        "critical": 1.0,
        "high": 0.75,
        "moderate": 0.50,
        "low": 0.25,
        "unknown": 0.35,
    }

    # Label ranks, used to keep the fused label at least as severe as the
    # label each modality reported on its own.
    SEVERITY_RANK = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
    URGENCY_RANK = {"routine": 0, "urgent": 1, "emergent": 2, "emergency": 2}

    # Risk finding bonuses: high-risk radiological findings
    RISK_FINDING_BONUSES = {
        "pneumothorax": 0.30,
        "mass": 0.25,
        "nodule": 0.25,
        "effusion": 0.20,
        "edema": 0.20,
        "cardiomegaly": 0.15,
        "pneumonia": 0.15,
        "consolidation": 0.15,
        "atelectasis": 0.10,
        "emphysema": 0.10,
        "fibrosis": 0.10,
        "infiltration": 0.10,
        "pleural_thickening": 0.05,
        "hernia": 0.05,
    }

    # Ontology mapping for more robust concordant matching.
    # Key: a detected CV class. Value: terms of a text diagnosis that the
    # finding supports. Consolidation and infiltration are how pneumonia shows
    # on a chest X-ray, edema and cardiomegaly are signs of heart failure, so
    # those pairs agree rather than conflict.
    ONTOLOGY_MAPPING = {
        "pneumonia": ["pneumonia", "lung infection", "chest infection",
                      "infiltration", "consolidation"],
        "consolidation": ["consolidation", "pneumonia", "lung infection",
                          "chest infection"],
        "infiltration": ["infiltration", "infiltrate", "pneumonia",
                         "lung infection", "chest infection"],
        "effusion": ["effusion", "pleural_thickening", "fluid", "heart failure"],
        "edema": ["edema", "fluid in lungs", "pulmonary edema", "heart failure",
                  "fluid overload"],
        "cardiomegaly": ["cardiomegaly", "enlarged heart", "heart",
                         "cardiomyopathy"],
        "mass": ["mass", "nodule", "lesion", "tumor", "cancer", "malignancy",
                 "carcinoma"],
        "nodule": ["nodule", "mass", "lesion", "tumor", "cancer", "malignancy",
                   "carcinoma"],
        "pneumothorax": ["pneumothorax", "collapsed lung"],
        "atelectasis": ["atelectasis", "lung collapse"],
        "emphysema": ["emphysema", "copd",
                      "chronic obstructive pulmonary disease"],
        "fibrosis": ["fibrosis", "interstitial lung disease"],
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
        strategy = self._determine_strategy(
            nlp_primary, cv_primary, nlp_conf, cv_conf,
            cv_finding_detected=cv_diagnosis.get("finding_detected", True),
        )

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

        # What the image flag is worth on the model's validation set
        reliability = cv_diagnosis.get("reliability")
        if reliability:
            result["explanation"] += f" {reliability['note']}"

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
                            nlp_conf: float, cv_conf: float,
                            cv_finding_detected: bool = True) -> str:
        """Determine the appropriate fusion strategy."""
        nlp_lower = nlp_primary.lower()

        # Detected CV classes, e.g. "Pleural_Thickening, Mass". Without a
        # detected finding cv_primary is a sentence ("No confident abnormal
        # finding detected"), which cannot agree with a diagnosis.
        cv_classes = (
            [c.strip().lower() for c in cv_primary.split(",") if c.strip()]
            if cv_finding_detected
            else []
        )

        # Check for agreement using ontology mapping
        is_concordant = False
        for key, related_terms in self.ONTOLOGY_MAPPING.items():
            if key in cv_classes:
                if any(self._mentions(nlp_lower, term) for term in related_terms):
                    is_concordant = True
                    break

        # Fallback check
        if not is_concordant and any(self._mentions(nlp_lower, cls) for cls in cv_classes):
            is_concordant = True

        if is_concordant:
            return "concordant"

        # Conflict happens when both modalities pass their own confidence threshold
        if nlp_conf >= settings.confidence_threshold_nlp and cv_conf >= settings.confidence_threshold_cv:
            return "discordant"

        return "complementary"

    @staticmethod
    def _mentions(text: str, term: str) -> bool:
        """Whole-word match (plural allowed): "mass" must not match "massive"."""
        term = re.escape(term.replace("_", " "))
        return re.search(r"\b" + term + r"(?:e?s)?\b", text) is not None

    def _fuse_concordant(self, nlp_diag: dict, cv_diag: dict) -> dict:
        """Both modalities agree — boost confidence safely."""
        nlp_conf = nlp_diag.get("confidence", 0)
        cv_conf = cv_diag.get("confidence", 0)
        question_answer = self._clean_question_answer(
            nlp_diag.get("question_answer", "")
        )

        # Calibrated fusion: weighted average + dynamic concordance bonus
        base = nlp_conf * 0.4 + cv_conf * 0.6
        agreement = 1 - abs(nlp_conf - cv_conf)
        concordance_bonus = 0.03 * agreement
        combined_conf = min(self.CONCORDANT_CONFIDENCE_CAP, base + concordance_bonus)

        # Uncertainty penalty: penalize high confidence when CV entropy is high
        combined_conf = self._apply_uncertainty_penalty(combined_conf, cv_diag)

        # Explainable severity scoring
        severity_result = self._compute_severity_score(nlp_diag, cv_diag)

        explanation = (
            f"Both text analysis and image analysis agree. "
            f"NLP suggests '{nlp_diag.get('primary_diagnosis', '')}' "
            f"(confidence: {nlp_conf:.0%}), "
            f"CV confirms '{cv_diag.get('predicted_class', '')}' "
            f"(model score: {cv_conf:.0%}). "
            f"Calibrated combined confidence: {combined_conf:.0%} "
            f"(agreement factor: {agreement:.2f})."
        )
        return {
            "primary_diagnosis": nlp_diag.get("primary_diagnosis", ""),
            "combined_confidence": combined_conf,
            "agreement": "concordant",
            "explanation": explanation,
            "question_answer": question_answer,
            "supporting_evidence": nlp_diag.get("supporting_evidence", []),
            "recommended_actions": nlp_diag.get("recommended_actions", []),
            "severity": severity_result["severity"],
            "urgency": severity_result["urgency"],
            "severity_score": severity_result["severity_score"],
            "severity_breakdown": severity_result["breakdown"],
        }

    def _fuse_complementary(self, nlp_diag: dict, cv_diag: dict) -> dict:
        """Modalities provide different but non-conflicting information."""
        nlp_conf = nlp_diag.get("confidence", 0)
        cv_conf = cv_diag.get("confidence", 0)
        question_answer = self._clean_question_answer(
            nlp_diag.get("question_answer", "")
        )

        combined_conf = nlp_conf * self.NLP_WEIGHT + cv_conf * self.CV_WEIGHT

        # Uncertainty penalty: penalize high confidence when CV entropy is high
        combined_conf = self._apply_uncertainty_penalty(combined_conf, cv_diag)

        # Explainable severity scoring
        severity_result = self._compute_severity_score(nlp_diag, cv_diag)

        explanation = (
            f"Text and image analyses provide complementary insights. "
            f"NLP analysis: '{nlp_diag.get('primary_diagnosis', '')}' ({nlp_conf:.0%}). "
            f"Image analysis: '{cv_diag.get('predicted_class', '')}' "
            f"(model score: {cv_conf:.0%}). "
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
            "severity": severity_result["severity"],
            "urgency": severity_result["urgency"],
            "severity_score": severity_result["severity_score"],
            "severity_breakdown": severity_result["breakdown"],
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

        # Uncertainty penalty: penalize high confidence when CV entropy is high
        combined_conf = self._apply_uncertainty_penalty(combined_conf, cv_diag)

        # Severity scoring — conservative for conflicts
        severity_result = self._compute_severity_score(nlp_diag, cv_diag)
        # Conflicts always escalate to at least "moderate" / "urgent"
        if severity_result["severity"] == "low":
            severity_result["severity"] = "moderate"
            severity_result["urgency"] = "urgent"

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
                f"(model score: {cv_conf:.0%}). "
                f"Professional review is recommended to resolve the discrepancy."
            ),
            "question_answer": question_answer,
            "conflict_flag": True,
            "supporting_evidence": nlp_diag.get("supporting_evidence", []),
            "recommended_actions": nlp_diag.get("recommended_actions", []),
            "severity": severity_result["severity"],
            "urgency": severity_result["urgency"],
            "severity_score": severity_result["severity_score"],
            "severity_breakdown": severity_result["breakdown"],
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
            "finding_detected": diagnosis.get("finding_detected"),
            "negative_screen": diagnosis.get("negative_screen", False),
            "abnormality_score": diagnosis.get("abnormality_score"),
            "max_threshold_ratio": diagnosis.get("max_threshold_ratio"),
            "modality": modality,
            "severity": diagnosis.get("severity", "moderate"),
            "urgency": diagnosis.get("urgency", "routine"),
        }

    def _compute_severity_score(
        self, nlp_diag: dict, cv_diag: dict
    ) -> dict:
        """
        Compute explainable severity score based on clinical factors.

        Formula: 0.5 * symptom_weight + 0.3 * image_burden + 0.2 * risk_bonus
        Does NOT use confidence — confidence ≠ severity.
        """
        # 1. Symptom severity weight (from NLP)
        nlp_severity = str(nlp_diag.get("severity") or "moderate").lower()
        symptom_weight = self.SEVERITY_WEIGHT_MAP.get(nlp_severity, 0.35)

        # 2. Image burden: normalized count of detected CV findings
        detected = cv_diag.get("detected_predictions", [])
        num_detected = len(detected)
        # Normalize: 1 finding = 0.3, 2 = 0.5, 3+ = 0.7, 5+ = 1.0
        if num_detected == 0:
            image_burden = 0.1
        elif num_detected == 1:
            image_burden = 0.3
        elif num_detected == 2:
            image_burden = 0.5
        elif num_detected <= 4:
            image_burden = 0.7
        else:
            image_burden = 1.0

        # 3. Risk finding bonus: check for high-risk radiological findings
        risk_bonus = 0.0
        risk_findings_found = []
        all_cv_classes = [
            p.get("class", "").lower().replace(" ", "_")
            for p in detected
        ]
        for finding, bonus in self.RISK_FINDING_BONUSES.items():
            if finding in all_cv_classes:
                risk_bonus = max(risk_bonus, bonus)
                risk_findings_found.append(finding)

        # Also check NLP primary for risk keywords (using word boundaries)
        nlp_primary = nlp_diag.get("primary_diagnosis", "").lower()
        for finding, bonus in self.RISK_FINDING_BONUSES.items():
            pattern = r"\b" + re.escape(finding.replace("_", " ")) + r"\b|\b" + re.escape(finding) + r"\b"
            if re.search(pattern, nlp_primary) and finding not in risk_findings_found:
                risk_bonus = max(risk_bonus, bonus)
                risk_findings_found.append(finding)

        # Composite score (configurable weights)
        w = self.SEVERITY_FORMULA_WEIGHTS
        severity_score = (
            w["symptom"] * symptom_weight
            + w["image"] * image_burden
            + w["risk"] * risk_bonus
        )
        severity_score = min(1.0, severity_score)

        # Map to labels
        if severity_score >= 0.75:
            severity = "critical"
            urgency = "emergency"
        elif severity_score >= 0.60:
            severity = "high"
            urgency = "urgent"
        elif severity_score >= 0.40:
            severity = "moderate"
            urgency = "routine"
        else:
            severity = "low"
            urgency = "routine"

        # The composite score blends the modalities, so on its own it can
        # land below what either one reported (NLP "high" + CV pneumothorax
        # scored "moderate"). Fusion must never de-escalate: keep the most
        # severe label among the score and each modality's own assessment.
        score_severity, score_urgency = severity, urgency
        for diag in (nlp_diag, cv_diag):
            diag_severity = str(diag.get("severity") or "").lower()
            if self.SEVERITY_RANK.get(diag_severity, -1) > self.SEVERITY_RANK[severity]:
                severity = diag_severity
            diag_urgency = str(diag.get("urgency") or "").lower()
            if self.URGENCY_RANK.get(diag_urgency, -1) > self.URGENCY_RANK[urgency]:
                urgency = diag_urgency

        return {
            "severity": severity,
            "urgency": urgency,
            "severity_score": round(severity_score, 4),
            "breakdown": {
                "score_severity": score_severity,
                "score_urgency": score_urgency,
                "nlp_severity": nlp_diag.get("severity"),
                "cv_severity": cv_diag.get("severity"),
                "symptom_weight": round(symptom_weight, 2),
                "image_burden": round(image_burden, 2),
                "risk_finding_bonus": round(risk_bonus, 2),
                "risk_findings_found": risk_findings_found,
                "weights": self.SEVERITY_FORMULA_WEIGHTS,
                "formula": (
                    f"{w['symptom']}*symptom + "
                    f"{w['image']}*image_burden + "
                    f"{w['risk']}*risk_findings"
                ),
            },
        }

    def _apply_uncertainty_penalty(
        self, combined_conf: float, cv_diag: dict
    ) -> float:
        """Reduce fusion confidence when CV model reports high uncertainty.

        Uses two complementary signals:
        - entropy_positive: entropy of detected (above-threshold) findings only
          — more clinically relevant than entropy_mean (all 14 classes)
        - near_threshold_count: number of classes near decision boundary
          — indicates how many predictions are borderline

        penalty = entropy_positive * 0.08 + near_threshold_count * 0.03
        final_conf *= (1 - penalty)
        """
        uncertainty = cv_diag.get("uncertainty", {})
        entropy_pos = uncertainty.get("entropy_positive", 0.0)
        near_count = uncertainty.get("near_threshold_count", 0)

        if entropy_pos <= 0 and near_count == 0:
            return combined_conf

        penalty_rate = (
            entropy_pos * self.UNCERTAINTY_PENALTY_FACTOR
            + near_count * 0.03
        )
        # Cap total penalty rate at 20% to prevent over-correction
        penalty_rate = min(penalty_rate, 0.20)
        adjusted = combined_conf * (1 - penalty_rate)

        if penalty_rate > 0.01:
            fusion_logger.info(
                f"Uncertainty penalty: entropy_pos={entropy_pos:.3f}, "
                f"near_threshold={near_count}, penalty_rate={penalty_rate:.3f}, "
                f"conf {combined_conf:.3f} -> {adjusted:.3f}"
            )
        return round(adjusted, 4)

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
