"""
Guideline Validation Layer.
Cross-references combined diagnosis against retrieved clinical guidelines
using LLM-as-a-Judge for semantic understanding (negation, synonyms, context).
Falls back to token-based matching when LLM is unavailable.
"""

import json

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    genai = None
    types = None

from app.config import settings
from app.core.nlp.vector_store import VectorStore
from app.utils.logger import nlp_logger


VALIDATION_PROMPT = """You are a clinical guideline validation expert. Your task is to determine whether a given diagnosis is SUPPORTED or CONTRADICTED by the provided clinical guidelines.

## Diagnosis to validate:
{diagnosis}

## Retrieved Clinical Guidelines:
{guidelines}

## Instructions:
1. Read the diagnosis and each guideline carefully.
2. Pay close attention to NEGATIONS (e.g., "no evidence of", "rules out", "negative for").
3. Recognize medical SYNONYMS and ACRONYMS (e.g., "CVA" = "Stroke", "MI" = "Heart Attack").
4. Evaluate whether the guidelines genuinely support, partially support, or contradict the diagnosis.

Respond in this exact JSON format ONLY (no extra text):
{{
    "is_consistent": true,
    "support_score": 0.75,
    "reasoning": "Brief explanation of why the diagnosis is or is not supported",
    "negation_detected": false,
    "synonym_matches": ["term1 = term2"]
}}

Rules for support_score:
- 0.0 to 0.2: Guidelines CONTRADICT or explicitly rule out the diagnosis
- 0.2 to 0.4: No meaningful support found in guidelines
- 0.4 to 0.6: Partial or indirect support
- 0.6 to 0.8: Good support with relevant guideline evidence
- 0.8 to 1.0: Strong, direct support from guidelines

Rules for is_consistent:
- true: support_score >= 0.4
- false: support_score < 0.4
"""


# Unified threshold for guideline consistency — same for LLM and fallback
VALIDATION_THRESHOLD = 0.4


class GuidelineValidator:
    """Validates diagnoses against retrieved clinical guidelines."""

    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store or VectorStore()
        self._client = None
        nlp_logger.info("GuidelineValidator initialized")

    def _ensure_llm(self):
        """Configure the LLM client on first use."""
        if self._client is None and settings.gemini_api_key_str and HAS_GENAI:
            self._client = genai.Client(api_key=settings.gemini_api_key_str)

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
                "referenced_guidelines": [],
            }

        # Retrieve guidelines related to the diagnosis
        if primary and primary != "No input provided":
            related_docs = self.vector_store.search(primary, k=3)
        else:
            related_docs = []

        if not related_docs:
            validation_notes = [
                "No relevant guidelines found for validation. "
                "Consider expanding the guideline database."
            ]
            is_consistent = False
            guideline_support = 0.0
        else:
            # Try LLM-as-a-Judge first, fall back to token matching
            is_consistent, guideline_support, validation_notes = (
                self._validate_with_llm(primary, related_docs)
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
            f"support={guideline_support:.2f}"
        )
        return validated_output

    # ------------------------------------------------------------------
    # LLM-as-a-Judge (primary)
    # ------------------------------------------------------------------

    def _validate_with_llm(
        self, diagnosis: str, docs: list[dict]
    ) -> tuple[bool, float, list[str]]:
        """Use Gemini to semantically validate the diagnosis against guidelines."""
        self._ensure_llm()

        if not (settings.gemini_api_key_str and HAS_GENAI and self._client):
            nlp_logger.warning("LLM unavailable — falling back to token matching")
            return self._validate_with_tokens(diagnosis, docs)

        guidelines_text = "\n\n".join(
            f"[Guideline {i+1}] {doc['text']}" for i, doc in enumerate(docs)
        )
        prompt = VALIDATION_PROMPT.format(
            diagnosis=diagnosis, guidelines=guidelines_text
        )

        try:
            response = self._client.models.generate_content(
                model=settings.llm_model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            result = self._parse_llm_response(response.text)

            is_consistent = result["is_consistent"]
            support_score = result["support_score"]
            notes = [f"LLM validation: {result['reasoning']}"]

            if result.get("negation_detected"):
                notes.append(
                    "⚠️ Negation detected: guidelines may contradict this diagnosis."
                )

            if result.get("synonym_matches"):
                notes.append(
                    f"Synonym matches found: {', '.join(result['synonym_matches'])}"
                )

            if support_score >= 0.6:
                notes.append(
                    f"Diagnosis is supported by clinical guidelines "
                    f"(support score: {support_score:.0%})"
                )
            elif support_score >= 0.4:
                notes.append(
                    "Partial guideline support found. Additional review recommended."
                )
            else:
                notes.append(
                    "Limited guideline support. Diagnosis may require further investigation."
                )

            nlp_logger.info(f"LLM validation complete: consistent={is_consistent}, score={support_score:.2f}")
            return is_consistent, support_score, notes

        except Exception as e:
            nlp_logger.error(f"LLM validation failed: {e} — falling back to token matching")
            return self._validate_with_tokens(diagnosis, docs)

    def _parse_llm_response(self, response_text: str) -> dict:
        """Parse the JSON response from the validation LLM."""
        import re
        text = response_text.strip()

        # Strip markdown code fences
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Try direct parse
        try:
            parsed = json.loads(text)
            return self._sanitize_llm_result(parsed)
        except json.JSONDecodeError:
            pass

        # Try regex extraction
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return self._sanitize_llm_result(parsed)
            except json.JSONDecodeError:
                pass

        nlp_logger.warning("Could not parse LLM validation response")
        return {
            "is_consistent": False,
            "support_score": 0.3,
            "reasoning": "Could not parse LLM response — treating as uncertain.",
            "negation_detected": False,
            "synonym_matches": [],
        }

    def _sanitize_llm_result(self, parsed: dict) -> dict:
        """Ensure all expected fields are present with correct types."""
        score = float(parsed.get("support_score", 0.3))
        score = max(0.0, min(1.0, score))
        # Deterministic: is_consistent is ALWAYS derived from score,
        # never trusting the LLM's boolean to avoid contradictory outputs
        # (e.g. score=0.3 but is_consistent=true).
        return {
            "is_consistent": score >= VALIDATION_THRESHOLD,
            "support_score": score,
            "reasoning": str(parsed.get("reasoning", "No reasoning provided.")),
            "negation_detected": bool(parsed.get("negation_detected", False)),
            "synonym_matches": list(parsed.get("synonym_matches", [])),
        }

    # ------------------------------------------------------------------
    # Token-based matching (fallback)
    # ------------------------------------------------------------------

    def _validate_with_tokens(
        self, diagnosis: str, docs: list[dict]
    ) -> tuple[bool, float, list[str]]:
        """Fallback: simple keyword matching when LLM is unavailable."""
        nlp_logger.info("Using token-based fallback for validation")

        diagnosis_terms = set(
            term.strip(",.;:")
            for term in diagnosis.lower().split()
            if len(term.strip(",.;:")) > 2
        )

        guideline_support = 0.0
        for doc in docs:
            doc_text = doc.get("text", "").lower()
            matches = sum(1 for term in diagnosis_terms if term in doc_text)
            if matches > 0:
                # Cosine distance can exceed 1; a dissimilar doc must not
                # subtract from the support of the others.
                similarity = max(0.0, 1.0 - doc.get("distance", 0.0))
                guideline_support += (
                    similarity * (matches / max(len(diagnosis_terms), 1))
                )

        guideline_support = min(1.0, guideline_support)
        is_consistent = guideline_support >= VALIDATION_THRESHOLD
        notes = ["Validation method: token matching (LLM unavailable)"]

        if guideline_support >= VALIDATION_THRESHOLD:
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

