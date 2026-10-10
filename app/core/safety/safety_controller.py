"""
Safety & Risk Control module.
Evaluates diagnosis confidence, detects contradictions and emergency conditions,
and determines whether output is safe to present or requires doctor consultation.
"""

import json
import re
from pathlib import Path
from PIL import Image

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    genai = None
    types = None

from app.config import settings
from app.core.nlp.text_preprocessor import (
    STATED_FAMILY_HISTORY_CONTEXTS, family_context, is_negated,
    normalize_contractions,
)
from app.utils.logger import describe_error, safety_logger

# Red-flag conditions that always require urgent medical attention
RED_FLAG_CONDITIONS = [
    "myocardial infarction", "heart attack", "stroke", "cerebrovascular",
    "pulmonary embolism", "anaphylaxis", "sepsis", "meningitis",
    "cardiac arrest", "respiratory failure", "hemorrhage", "bleeding",
    "seizure", "unconscious", "life-threatening", "acute abdomen",
    "tension pneumothorax",
]

# Generic alarm words. A red flag in what the patient wrote or in the
# diagnosis label, but routine vocabulary in a generated explanation
# ("return if symptoms become severe"), so they are not scanned there.
RED_FLAG_QUALIFIERS = ["critical", "emergency", "severe"]

RED_FLAG_KEYWORDS = RED_FLAG_CONDITIONS + RED_FLAG_QUALIFIERS

# Severity ranking hierarchy for risk levels
RISK_ORDER = {
    "low": 0,
    "moderate": 1,
    "high": 2,
    "critical": 3,
}


class SafetyController:
    """Evaluates safety of diagnosis output and controls release."""

    def __init__(self):
        self.confidence_threshold_cv = settings.confidence_threshold_cv
        self.confidence_threshold_nlp = settings.confidence_threshold_nlp
        self.safety_threshold = settings.safety_threshold
        self._client = None
        safety_logger.info(
            f"SafetyController initialized "
            f"(cv_threshold={self.confidence_threshold_cv}, "
            f"nlp_threshold={self.confidence_threshold_nlp})"
        )

    def _ensure_vlm(self):
        """Configure Gemini VLM on first use."""
        if self._client is None and settings.gemini_api_key_str and HAS_GENAI:
            self._client = genai.Client(api_key=settings.gemini_api_key_str)
            safety_logger.info("VLM safety verifier configured")

    def evaluate(
        self,
        validated_output: dict,
        image_path: str = None,
        symptoms_text: str = None,
        clinical_question: str = None,
    ) -> dict:
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
        primary = str(validated_output.get("primary_diagnosis") or "")
        severity = validated_output.get("severity", "moderate")
        urgency = validated_output.get("urgency", "routine")
        modality = validated_output.get("modality", "unknown")
        negative_screen = bool(validated_output.get("negative_screen"))

        # Chọn ngưỡng tin cậy theo phân loại
        if modality == "nlp":
            threshold = self.confidence_threshold_nlp
        else:
            # Cho ảnh và đa phương thức
            threshold = self.confidence_threshold_cv
        threshold = validated_output.get("confidence_threshold", threshold)

        # Check 1: Confidence threshold
        if confidence < threshold and not (modality == "cv" and negative_screen):
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
        # Scan ALL text sources — not just primary_diagnosis — to prevent
        # bypass when NLP abstracts the user's input into a generic label
        # (e.g. "Inquiry about disease etiology" instead of "heart attack").
        # Each source is scanned on its own so a negation in one cannot
        # reach into the next.
        scan_sources = [
            (primary, RED_FLAG_KEYWORDS),
            (symptoms_text or "", RED_FLAG_KEYWORDS),
            (clinical_question or "", RED_FLAG_KEYWORDS),
            (str(validated_output.get("explanation") or ""), RED_FLAG_CONDITIONS),
        ]
        red_flags_found = list(dict.fromkeys(
            kw
            for text, keywords in scan_sources
            for kw in self._find_red_flags(text, keywords)
        ))
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

        vlm_review = None
        if self._should_run_vlm_review(validated_output, image_path, risk_factors):
            vlm_review = self._run_vlm_safety_review(
                validated_output=validated_output,
                image_path=image_path,
                symptoms_text=symptoms_text,
                clinical_question=clinical_question,
                risk_factors=risk_factors,
                warning_notes=warning_notes,
            )
            risk_factors, warning_notes, is_safe = self._apply_vlm_review(
                vlm_review, validated_output, risk_factors, warning_notes, is_safe
            )

        # Determine safety status
        if is_safe:
            safety_status = "approved"
            safety_message = (
                "✅ Output approved. This analysis has passed safety checks."
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

        explanation = validated_output.get("explanation", "")
        if warning_notes:
            safety_note_text = "\n".join(f"- {note}" for note in warning_notes)
            explanation = (
                f"{explanation}\n\nSafety notes:\n{safety_note_text}"
                if explanation
                else f"Safety notes:\n{safety_note_text}"
            )

        safety_result = {
            **validated_output,
            "explanation": explanation,
            "safety_status": safety_status,
            "is_approved": is_safe,
            "risk_factors": risk_factors,
            "safety_warning_notes": warning_notes,
            "safety_message": safety_message,
            "confidence_threshold_used": threshold,
            "vlm_safety_review": vlm_review,
        }

        safety_logger.info(
            f"Safety evaluation: status={safety_status}, "
            f"risks={len(risk_factors)}"
        )
        return safety_result

    @staticmethod
    def _find_red_flags(text: str, keywords: list[str]) -> list[str]:
        """Keywords mentioned in text as whole words, not negated, and not
        stated family history.

        Whole-word matching keeps "keystrokes" from matching "stroke"; the
        negation check keeps "no bleeding" / "not life-threatening" from
        escalating a case.

        A relative's condition is skipped only when the text states or dates
        it as history ("family history of stroke", "mother had a stroke last
        year"). An undated "my mother had a stroke" still counts: it can be
        someone reporting an emergency, and this gate errs towards escalation.
        """
        text = normalize_contractions(text.lower())
        found = []
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"(?:e?s)?\b"
            if any(
                not is_negated(text, match.start(), match.end())
                and family_context(text, match.start(), match.end())
                not in STATED_FAMILY_HISTORY_CONTEXTS
                for match in re.finditer(pattern, text)
            ):
                found.append(kw)
        return found

    def _should_run_vlm_review(
        self,
        validated_output: dict,
        image_path: str = None,
        risk_factors: list[str] = None,
    ) -> bool:
        """Run VLM verification only when image context can change the decision."""
        if not settings.vlm_safety_enabled or not image_path:
            return False
        if not settings.gemini_api_key_str or not HAS_GENAI:
            return False
        if not Path(image_path).exists():
            return False

        modality = validated_output.get("modality", "unknown")
        return (
            modality in ("cv", "multimodal")
            and (
                validated_output.get("conflict_flag")
                or not validated_output.get("is_guideline_consistent", True)
                or bool(risk_factors)
                or modality == "multimodal"
            )
        )

    def _run_vlm_safety_review(
        self,
        validated_output: dict,
        image_path: str,
        symptoms_text: str = None,
        clinical_question: str = None,
        risk_factors: list[str] = None,
        warning_notes: list[str] = None,
    ) -> dict | None:
        """Use Gemini multimodal reasoning as cross-modal safety verifier."""
        self._ensure_vlm()
        if not self._client:
            return None

        prompt = self._build_vlm_prompt(
            validated_output=validated_output,
            symptoms_text=symptoms_text,
            clinical_question=clinical_question,
            risk_factors=risk_factors or [],
            warning_notes=warning_notes or [],
        )

        try:
            image = Image.open(image_path).convert("RGB")
            response = self._client.models.generate_content(
                model=settings.vlm_safety_model,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            review = json.loads(response.text)
            safety_logger.info(
                "VLM safety review: decision=%s risk=%s",
                review.get("decision"),
                review.get("risk_level"),
            )
            return review
        except Exception as e:
            safety_logger.error("VLM safety review failed: %s", describe_error(e))
            return self._fallback_safety_review(validated_output, describe_error(e))

    def _fallback_safety_review(
        self, validated_output: dict, error_msg: str
    ) -> dict:
        """Rule-based safety review when VLM is unavailable.

        Ensures the safety layer never 'disappears' — produces a conservative
        assessment using available structured data (confidence, severity,
        urgency, conflict_flag) plus a rule-based cross-modal sanity check.
        """
        safety_logger.info("Running fallback safety review (VLM unavailable)")
        # Log raw error internally — never expose to patient output
        safety_logger.debug("VLM error detail (internal): %s", error_msg)

        confidence = validated_output.get("combined_confidence", 0)
        severity = validated_output.get("severity", "moderate")
        urgency = validated_output.get("urgency", "routine")
        conflict = validated_output.get("conflict_flag", False)

        # Cross-modal sanity check: compare NLP and CV findings
        cross_modal = self._cross_modal_sanity_check(validated_output)

        # Conservative risk assessment
        if conflict or severity in ("critical",) or urgency == "emergency":
            risk_level = "high"
            decision = "escalate"
        elif confidence < 0.5 or severity == "high" or urgency == "urgent":
            risk_level = "moderate"
            decision = "caution"
        else:
            risk_level = "low"
            decision = "caution"  # Never auto-approve without VLM

        # If cross-modal check detects strong disagreement, escalate
        if cross_modal["agreement_level"] == "conflicting" and decision != "escalate":
            decision = "caution"
            if RISK_ORDER[risk_level] < RISK_ORDER["moderate"]:
                risk_level = "moderate"

        return {
            "decision": decision,
            "risk_level": risk_level,
            "image_text_alignment": cross_modal["agreement_level"],
            "safety_risk_assessed": True,
            "conflict_resolution": (
                "VLM unavailable — rule-based cross-modal check applied"
            ),
            "safety_context_note": None,
            "reasoning": (
                f"External verification temporarily unavailable. "
                f"Rule-based fallback applied: confidence={confidence:.0%}, "
                f"severity={severity}, urgency={urgency}, conflict={conflict}. "
                f"Cross-modal check: {cross_modal['agreement_level']} "
                f"(overlap={cross_modal['semantic_overlap']:.0%}). "
                f"Additional human review recommended."
            ),
            "recommended_safety_action": (
                "Human review advised — VLM verification was not available"
            ),
            "cross_modal_check": cross_modal,
            "_vlm_fallback": True,
        }

    # Semantic groupings for cross-modal overlap computation.
    # consolidation/infiltration clinically support pneumonia, etc.
    SEMANTIC_GROUPS = {
        "pneumonia": {"pneumonia", "consolidation", "infiltration"},
        "heart_failure": {"cardiomegaly", "edema", "effusion"},
        "pleural_disease": {"effusion", "pleural_thickening"},
        "lung_mass": {"mass", "nodule"},
        "airway_disease": {"emphysema", "atelectasis"},
        "fibrotic_disease": {"fibrosis", "pleural_thickening"},
    }

    def _cross_modal_sanity_check(self, validated_output: dict) -> dict:
        """Rule-based cross-modal agreement check (no VLM needed).

        Uses semantic groupings to recognize that consolidation/infiltration
        support pneumonia, cardiomegaly/edema support heart_failure, etc.
        """
        nlp_diag = validated_output.get("nlp_diagnosis") or {}
        cv_diag = validated_output.get("cv_diagnosis") or {}

        nlp_primary = nlp_diag.get("primary_diagnosis", "").lower()
        cv_detected = cv_diag.get("detected_predictions", [])
        cv_classes = {p.get("class", "").lower() for p in cv_detected}

        if not nlp_primary or not cv_classes:
            return {
                "finding_overlap": 0.0,
                "semantic_overlap": 0.0,
                "agreement_level": "unknown",
                "nlp_terms": [],
                "cv_classes": sorted(cv_classes),
                "matched_terms": [],
                "semantic_matches": [],
            }

        # Step 1: Direct keyword overlap
        nlp_terms = set(nlp_primary.replace(",", " ").split())
        direct_matched = set()
        for cv_cls in cv_classes:
            cv_words = set(cv_cls.replace("_", " ").split())
            common = nlp_terms & cv_words
            if common:
                direct_matched.update(common)
            if cv_cls.replace("_", " ") in nlp_primary:
                direct_matched.add(cv_cls)

        # Step 2: Semantic overlap via ontology groupings
        # Find which semantic groups the NLP diagnosis belongs to
        nlp_groups = set()
        for group_name, members in self.SEMANTIC_GROUPS.items():
            if any(m in nlp_primary for m in members):
                nlp_groups.add(group_name)

        # Count CV classes that share a semantic group with NLP
        semantic_matched = set()
        for cv_cls in cv_classes:
            for group_name, members in self.SEMANTIC_GROUPS.items():
                if group_name in nlp_groups and cv_cls in members:
                    semantic_matched.add(cv_cls)

        # Combine: semantic_overlap counts both direct and semantic matches
        all_matched = direct_matched | semantic_matched
        direct_overlap = len(direct_matched) / max(len(cv_classes), 1)
        semantic_overlap = len(all_matched) / max(len(cv_classes), 1)

        # Agreement level based on semantic overlap (stronger signal)
        if semantic_overlap >= 0.5:
            agreement_level = "strong"
        elif semantic_overlap > 0 or direct_overlap > 0:
            agreement_level = "partial"
        elif validated_output.get("conflict_flag"):
            agreement_level = "conflicting"
        else:
            agreement_level = "weak"

        return {
            "finding_overlap": round(direct_overlap, 2),
            "semantic_overlap": round(semantic_overlap, 2),
            "agreement_level": agreement_level,
            "nlp_terms": sorted(nlp_terms),
            "cv_classes": sorted(cv_classes),
            "matched_terms": sorted(direct_matched),
            "semantic_matches": sorted(semantic_matched),
        }

    def _build_vlm_prompt(
        self,
        validated_output: dict,
        symptoms_text: str = None,
        clinical_question: str = None,
        risk_factors: list[str] = None,
        warning_notes: list[str] = None,
    ) -> str:
        """Build a constrained cross-modal verification prompt."""
        nlp_diag = validated_output.get("nlp_diagnosis")
        cv_diag = validated_output.get("cv_diagnosis")
        payload = {
            "patient_text": {
                "symptoms": symptoms_text or "",
                "clinical_question": clinical_question or "",
            },
            "current_diagnosis": {
                "primary_diagnosis": validated_output.get("primary_diagnosis"),
                "combined_confidence": validated_output.get("combined_confidence"),
                "fusion_strategy": validated_output.get("fusion_strategy"),
                "agreement": validated_output.get("agreement"),
                "explanation": validated_output.get("explanation"),
                "severity": validated_output.get("severity"),
                "urgency": validated_output.get("urgency"),
            },
            "nlp_diagnosis": nlp_diag,
            "cv_diagnosis": cv_diag,
            "validation": {
                "is_guideline_consistent": validated_output.get("is_guideline_consistent"),
                "guideline_support_score": validated_output.get("guideline_support_score"),
                "validation_notes": validated_output.get("validation_notes"),
            },
            "rule_based_risks": risk_factors or [],
            "warning_notes": warning_notes or [],
        }

        return (
            "You are a conservative clinical safety verifier for an educational "
            "multimodal clinical decision support system. Review the attached "
            "chest X-ray image together with the structured NLP/CV/fusion outputs. "
            "Perform cross-modal consistency verification, patient-facing safety assessment, "
            "and escalation risk analysis. Do not provide or judge a final clinical "
            "diagnosis for correctness (that is handled by the guideline validator); "
            "instead, focus purely on whether the integrated AI output is safe to display "
            "or should be escalated/cautioned for professional review.\n\n"
            "Return JSON only with this schema:\n"
            "{\n"
            '  "decision": "approve|caution|block|escalate",\n'
            '  "risk_level": "low|moderate|high|critical",\n'
            '  "image_text_alignment": "aligned|partially_aligned|conflicting|unclear",\n'
            '  "safety_risk_assessed": true,\n'
            '  "conflict_resolution": "short explanation of consistency/conflicts found",\n'
            '  "safety_context_note": "short qualified safety statement/notes if needed",\n'
            '  "reasoning": "brief safety and escalation reasoning",\n'
            '  "recommended_safety_action": "what safety precautions or escalation actions the app should take"\n'
            "}\n\n"
            "Be conservative: choose block/escalate for possible emergency signs, "
            "major image-text conflict, hallucinated anatomy/location, or low-quality "
            "image. Choose caution when the output is plausible but needs qualification. "
            "Choose approve only when the image, text, and generated explanation are "
            "coherent and no safety concern remains.\n\n"
            "LANGUAGE RULES (MANDATORY):\n"
            "- NEVER use definitive visual claims: 'clearly shows', 'definitely', "
            "'obviously visible', 'undoubtedly', 'unmistakable', 'certain'.\n"
            "- ALWAYS use hedged clinical language: 'findings consistent with', "
            "'radiographic features suggestive of', 'image analysis suggests', "
            "'pattern compatible with', 'may represent'.\n"
            "- Frame ALL observations as AI-assisted interpretations, "
            "NOT as clinical diagnoses.\n"
            "- Use qualifying phrases: 'based on the automated analysis', "
            "'the model output indicates', 'computational assessment suggests'.\n\n"
            f"Structured case data:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def _apply_vlm_review(
        self,
        vlm_review: dict | None,
        validated_output: dict,
        risk_factors: list[str],
        warning_notes: list[str],
        is_safe: bool,
    ) -> tuple[list[str], list[str], bool]:
        """Let VLM soften soft risks or escalate unsafe cases."""
        if not vlm_review:
            return risk_factors, warning_notes, is_safe

        decision = str(vlm_review.get("decision", "")).lower()
        reasoning = vlm_review.get("reasoning", "")

        if vlm_review.get("_vlm_fallback"):
            # The VLM never looked at this case. A rule-based stand-in may add
            # a note, but it must not clear risks the way a real review can —
            # otherwise an API outage would relax the safety gate.
            note = "VLM safety verification unavailable — rule-based review applied"
            if reasoning:
                note += f": {reasoning}"
            warning_notes = list(dict.fromkeys(warning_notes + [note]))
            return risk_factors, warning_notes, is_safe

        hard_risks = [
            r for r in risk_factors
            if r.startswith("Low confidence") or r.startswith("Red-flag")
        ]

        if decision == "block":
            reason = f"VLM safety verifier recommends {decision}"
            if reasoning:
                reason += f": {reasoning}"
            risk_factors = list(dict.fromkeys(risk_factors + [reason]))
            return risk_factors, warning_notes, False

        if decision == "escalate":
            can_warn_instead = (
                validated_output.get("fusion_strategy") == "concordant"
                and validated_output.get("agreement") == "concordant"
                and validated_output.get("is_guideline_consistent", False)
                and not hard_risks
                and not validated_output.get("conflict_flag")
            )
            note = "VLM safety verifier recommends escalation"
            if reasoning:
                note += f": {reasoning}"

            if can_warn_instead:
                warning_notes = list(dict.fromkeys(warning_notes + [note]))
                risk_factors = hard_risks
                return risk_factors, warning_notes, True

            risk_factors = list(dict.fromkeys(risk_factors + [note]))
            return risk_factors, warning_notes, False

        if decision in ("approve", "caution"):
            if decision == "caution":
                note = "VLM safety verifier advises caution"
                if reasoning:
                    note += f": {reasoning}"
                warning_notes = list(dict.fromkeys(warning_notes + [note]))

            # VLM may resolve soft risks such as guideline retrieval gaps or
            # NLP/CV conflict, but never clears hard low-confidence/red-flag risks.
            risk_factors = hard_risks
            return risk_factors, warning_notes, len(risk_factors) == 0

        return risk_factors, warning_notes, is_safe
