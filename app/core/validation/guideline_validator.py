"""
Guideline Validation Layer.
Cross-references combined diagnosis against retrieved clinical guidelines
using LLM-as-a-Judge for semantic understanding (negation, synonyms, context).
Falls back to token-based matching when the judge is unavailable.
"""

import re

from app.config import settings
from app.core.nlp.vector_store import VectorStore
from app.core.validation.guideline_judge import (
    GeminiJudge, GuidelineJudge, OpenAIJudge,
)
from app.utils.logger import nlp_logger


# Unified threshold for guideline consistency — same for the judge and fallback
VALIDATION_THRESHOLD = 0.4

# How a validation result was reached (the "validation_method" output field)
METHOD_JUDGE = "llm_judge"
METHOD_TOKENS = "token_matching"
METHOD_SKIPPED = "skipped"
METHOD_NO_GUIDELINES = "no_guidelines"


# Words that say nothing about which condition a diagnosis names
_GENERIC_WORDS = frozenset({
    "a", "an", "and", "of", "or", "the", "to", "with", "no", "not",
    "acute", "chronic", "mild", "moderate", "severe", "exacerbation",
    "suspected", "possible", "probable", "likely",
})


def _words(text: str) -> set[str]:
    """Lower-cased words of two or more characters, plural "s" dropped."""
    return {
        word[:-1] if len(word) > 3 and word.endswith("s") else word
        for word in re.findall(r"[a-z0-9]{2,}", text.lower())
    }


def default_judge() -> GuidelineJudge:
    """The OpenAI judge when a key is configured, otherwise the generator's model.

    With a key configured the choice is final: if that judge fails, validation
    falls back to token matching, never quietly to the generator grading
    itself.
    """
    if settings.openai_api_key.get_secret_value():
        return OpenAIJudge()
    return GeminiJudge()


class GuidelineValidator:
    """Validates diagnoses against retrieved clinical guidelines."""

    def __init__(self, vector_store: VectorStore = None, judge: GuidelineJudge = None):
        self.vector_store = vector_store or VectorStore()
        self.judge = judge or default_judge()
        nlp_logger.info(
            f"GuidelineValidator initialized (judge={self.judge.provider}:"
            f"{self.judge.model}, available={self.judge.available})"
        )

    def validate(self, combined_diagnosis: dict) -> dict:
        """
        Validate a combined diagnosis against clinical guidelines.

        Args:
            combined_diagnosis: Output from FusionEngine

        Returns:
            ValidatedOutput with validation status and notes
        """
        primary = combined_diagnosis.get("primary_diagnosis", "")
        confidence = combined_diagnosis.get("combined_confidence", 0)
        modality = combined_diagnosis.get("modality", "unknown")

        # CV-only outputs are image classifier labels, not text-derived clinical
        # claims. Guideline retrieval is not a meaningful gate for those labels.
        if modality == "cv":
            return {
                **combined_diagnosis,
                "validation_status": "validated",
                "is_guideline_consistent": True,
                "guideline_support_score": 0.0,
                "validation_notes": [
                    "Guideline validation skipped for CV-only image classification."
                ],
                "validation_method": METHOD_SKIPPED,
                "judge_model": None,
                "referenced_guidelines": [],
            }

        # Retrieve guidelines related to the diagnosis
        if primary and primary != "No input provided":
            related_docs = self.vector_store.search(primary, k=3)
        else:
            related_docs = []

        judge_model = None
        if not related_docs:
            validation_notes = [
                "No relevant guidelines found for validation. "
                "Consider expanding the guideline database."
            ]
            is_consistent = False
            guideline_support = 0.0
            method = METHOD_NO_GUIDELINES
        else:
            # Try LLM-as-a-Judge first, fall back to token matching
            verdict = self.judge.judge(
                primary, [doc.get("text", "") for doc in related_docs]
            )
            if verdict is not None:
                method, judge_model = METHOD_JUDGE, self.judge.model
                is_consistent, guideline_support, validation_notes = (
                    self._read_verdict(verdict)
                )
            else:
                method = METHOD_TOKENS
                is_consistent, guideline_support, validation_notes = (
                    self._validate_with_tokens(primary, related_docs)
                )

        # Check for conflict flags from fusion
        if combined_diagnosis.get("conflict_flag"):
            validation_notes.append(
                "⚠️ NLP and CV modalities produced conflicting results."
            )
            is_consistent = False

        validated_output = {
            **combined_diagnosis,
            "validation_status": "validated" if is_consistent else "needs_review",
            "is_guideline_consistent": is_consistent,
            "guideline_support_score": guideline_support,
            "validation_notes": validation_notes,
            "validation_method": method,
            "judge_model": judge_model,
            "referenced_guidelines": [
                {
                    "text": d.get("text", "")[:200],
                    "distance": d.get("distance", 0),
                }
                for d in related_docs
            ],
        }

        nlp_logger.info(
            f"Validation: consistent={is_consistent}, "
            f"support={guideline_support:.2f}, method={method}"
        )
        return validated_output

    # ------------------------------------------------------------------
    # LLM-as-a-Judge (primary)
    # ------------------------------------------------------------------

    def _read_verdict(self, verdict: dict) -> tuple[bool, float, list[str]]:
        """Turn a judge verdict into the consistency decision and its notes."""
        support_score = verdict["support_score"]
        # Deterministic: consistency is ALWAYS derived from the score, never
        # from a boolean the model could return in contradiction with it.
        is_consistent = support_score >= VALIDATION_THRESHOLD
        notes = [f"Guideline judge ({self.judge.model}): {verdict['reasoning']}"]

        if verdict.get("negation_detected"):
            notes.append(
                "⚠️ Negation detected: guidelines may contradict this diagnosis."
            )

        if verdict.get("synonym_matches"):
            notes.append(
                f"Synonym matches found: {', '.join(verdict['synonym_matches'])}"
            )

        if support_score >= 0.6:
            notes.append(
                f"Diagnosis is supported by clinical guidelines "
                f"(support score: {support_score:.0%})"
            )
        elif support_score >= VALIDATION_THRESHOLD:
            notes.append(
                "Partial guideline support found. Additional review recommended."
            )
        else:
            notes.append(
                "Limited guideline support. Diagnosis may require further investigation."
            )

        return is_consistent, support_score, notes

    # ------------------------------------------------------------------
    # Token-based matching (fallback)
    # ------------------------------------------------------------------

    def _validate_with_tokens(
        self, diagnosis: str, docs: list[dict]
    ) -> tuple[bool, float, list[str]]:
        """Fallback: whole-word matching when the judge is unavailable.

        The score is the share of the diagnosis's words found in the single
        passage that has most of them. It errs towards review: word matching
        knows no synonyms, so it rejects some supported diagnoses, but it
        should not pass one the guidelines do not describe.
        """
        nlp_logger.info("Using token-based fallback for validation")

        # A parenthesis qualifies the diagnosis ("Stroke (CVA)", "(possible
        # STEMI)") and is not required in the passage. One that states a
        # measurement cannot be checked against the guideline's criteria by
        # matching words, so such a label is never passed.
        qualifiers = re.findall(r"\(([^)]*)\)", diagnosis)
        unverifiable = any(re.search(r"\d", q) for q in qualifiers)
        diagnosis_terms = _words(re.sub(r"\([^)]*\)", " ", diagnosis)) - _GENERIC_WORDS

        guideline_support = 0.0
        for doc in docs:
            # Cosine distance of 1 or more: the passage is unrelated
            if doc.get("distance", 0.0) >= 1.0 or not diagnosis_terms:
                continue
            coverage = (
                len(diagnosis_terms & _words(doc.get("text", "")))
                / len(diagnosis_terms)
            )
            # Half the words or fewer ("Heart block" in a passage on heart
            # failure) means the passage is about something else. Passages
            # are not added up: one of them has to describe the diagnosis.
            if coverage > 0.5:
                guideline_support = max(guideline_support, coverage)

        is_consistent = guideline_support >= VALIDATION_THRESHOLD and not unverifiable
        notes = ["Validation method: token matching (guideline judge unavailable)"]

        if unverifiable:
            notes.append(
                "The diagnosis states a measurement that word matching cannot "
                "check against the guidelines. Review recommended."
            )
        elif guideline_support >= VALIDATION_THRESHOLD:
            notes.append(
                f"Diagnosis is supported by clinical guidelines "
                f"(support score: {guideline_support:.0%})"
            )
        elif guideline_support >= 0.2:
            notes.append(
                "Partial guideline support found. Additional review recommended."
            )
        else:
            notes.append(
                "Limited guideline support. Diagnosis may require further investigation."
            )

        return is_consistent, guideline_support, notes
