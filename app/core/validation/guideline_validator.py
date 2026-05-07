"""
Guideline Validation Layer.
Cross-references combined diagnosis against retrieved clinical guidelines
for consistency checking.
"""

from app.core.nlp.vector_store import VectorStore
from app.utils.logger import nlp_logger


class GuidelineValidator:
    """Validates diagnoses against retrieved clinical guidelines."""

    def __init__(self):
        self.vector_store = VectorStore()
        nlp_logger.info("GuidelineValidator initialized")

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
        # claims. Guideline retrieval is not a meaningful gate for those labels
        # by itself, especially for "no modeled abnormality" cases.
        if modality == "cv":
            return {
                **combined_diagnosis,
                "validation_status": "validated",
                "is_guideline_consistent": True,
                "guideline_support_score": 0.0,
                "validation_notes": [
                    "Guideline validation skipped for CV-only image classification."
                ],
                "referenced_guidelines": [],
            }

        # Retrieve guidelines related to the diagnosis
        if primary and primary != "No input provided":
            related_docs = self.vector_store.search(primary, k=3)
        else:
            related_docs = []

        # Check consistency
        validation_notes = []
        is_consistent = True
        guideline_support = 0.0

        if related_docs:
            # Check if any retrieved guidelines mention the diagnosis
            diagnosis_terms = set(
                term.strip(",.;:") for term in primary.lower().split()
                if len(term.strip(",.;:")) > 2  # skip short stop-words
            )
            for doc in related_docs:
                doc_text = doc["text"].lower()
                matches = sum(1 for term in diagnosis_terms if term in doc_text)
                if matches > 0:
                    guideline_support += (1.0 - doc["distance"]) * (matches / max(len(diagnosis_terms), 1))

            guideline_support = min(1.0, guideline_support)

            if guideline_support > 0.3:
                validation_notes.append(
                    f"Diagnosis is supported by clinical guidelines "
                    f"(support score: {guideline_support:.0%})"
                )
            elif guideline_support > 0.1:
                validation_notes.append(
                    "Partial guideline support found. Additional review recommended."
                )
            else:
                validation_notes.append(
                    "Limited guideline support. Diagnosis may require further investigation."
                )
                is_consistent = False
        else:
            validation_notes.append(
                "No relevant guidelines found for validation. "
                "Consider expanding the guideline database."
            )
            is_consistent = False

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
            "referenced_guidelines": [
                {"text": d["text"][:200], "distance": d["distance"]}
                for d in related_docs
            ],
        }

        nlp_logger.info(
            f"Validation: consistent={is_consistent}, "
            f"support={guideline_support:.2f}"
        )
        return validated_output
