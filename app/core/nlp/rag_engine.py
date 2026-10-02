"""
RAG Engine: Combines ChromaDB retrieval with LLM reasoning
for clinical diagnosis generation.
"""

import json
import math
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
from app.core.nlp.text_preprocessor import TextPreprocessor
from app.core.nlp.question_understanding import QuestionUnderstanding
from app.core.nlp.clinical_fallback import ClinicalFallbackEngine
from app.utils.logger import nlp_logger


RAG_PROMPT_TEMPLATE = """You are an expert clinical decision support system. Based on the clinical guidelines provided and the patient's input, provide a detailed clinical analysis.

## Retrieved Clinical Guidelines:
{context}

## Patient Input:
{user_input}

## Instructions:
Analyze the patient's symptoms/question using the clinical guidelines above. Provide your response in the following JSON format ONLY (no extra text):

{{
    "primary_diagnosis": "Most likely diagnosis",
    "confidence": 0.85,
    "differential_diagnoses": [
        {{"condition": "Alternative diagnosis 1", "probability": 0.1}},
        {{"condition": "Alternative diagnosis 2", "probability": 0.05}}
    ],
    "explanation": "Detailed clinical reasoning explaining your diagnosis",
    "question_answer": "If a clinical question was asked, provide a direct, thorough answer here. If no question was asked, write N/A.",
    "supporting_evidence": ["Evidence point 1 from guidelines", "Evidence point 2"],
    "recommended_actions": ["Action 1", "Action 2"],
    "severity": "low|moderate|high|critical",
    "urgency": "routine|urgent|emergency"
}}

IMPORTANT: 
- Base your analysis ONLY on the provided clinical guidelines.
- Assign realistic confidence scores.
- Always recommend consulting a healthcare professional.
- Be thorough but concise in your explanation.
- In supporting_evidence, do NOT use double-quote characters inside string values. Use single quotes instead.
- If the patient asked a clinical question, provide a clear, direct answer in the question_answer field.
"""


class RAGEngine:
    """RAG pipeline combining ChromaDB retrieval with LLM generation."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        question_understanding: QuestionUnderstanding | None = None,
        fallback_engine: ClinicalFallbackEngine | None = None,
    ):
        """Initialize RAGEngine using Dependency Injection for all core NLP components."""
        self.vector_store = vector_store or VectorStore()
        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        self.question_understanding = question_understanding or QuestionUnderstanding()
        self._fallback = fallback_engine or ClinicalFallbackEngine()
        self._client = None
        nlp_logger.info("RAGEngine initialized")

    def _ensure_llm(self):
        """Configure the LLM client on first use."""
        if self._client is None and settings.gemini_api_key_str and HAS_GENAI:
            self._client = genai.Client(api_key=settings.gemini_api_key_str)
            nlp_logger.info(f"LLM configured: {settings.llm_model_name}")

    def diagnose_from_text(self, symptoms_text: str = None,
                           clinical_question: str = None) -> dict:
        """
        Perform NLP-based diagnosis using RAG.

        Args:
            symptoms_text: Patient symptom description
            clinical_question: Clinical question

        Returns:
            NLP diagnosis dictionary
        """
        nlp_logger.info("Starting NLP diagnosis pipeline")

        # Preprocess inputs
        search_query = ""
        processed_input = ""

        if symptoms_text:
            preprocessed = self.text_preprocessor.preprocess(symptoms_text)
            search_query = preprocessed["search_query"]
            processed_input = preprocessed["expanded_text"]

        if clinical_question:
            question_analysis = self.question_understanding.analyze(clinical_question)
            if search_query:
                search_query += " " + question_analysis["search_query"]
            else:
                search_query = question_analysis["search_query"]
            processed_input += (" " + clinical_question) if processed_input else clinical_question

        if not search_query:
            return self._empty_diagnosis("No input provided")

        # Retrieve relevant guidelines
        retrieved_docs = self.vector_store.search(search_query, k=5)
        if not retrieved_docs:
            return self._empty_diagnosis("No relevant guidelines found")

        # Build context from retrieved documents
        context = "\n\n".join([
            f"[Source: {doc['metadata'].get('source', 'unknown')}]\n{doc['text']}"
            for doc in retrieved_docs
        ])

        # Generate diagnosis using LLM (pass retrieved_docs for fallback)
        diagnosis = self._generate_diagnosis(context, processed_input, retrieved_docs)
        diagnosis["retrieved_documents"] = [
            {"text": d["text"][:200], "distance": d["distance"]}
            for d in retrieved_docs
        ]
        diagnosis["input_type"] = "text"

        nlp_logger.info(f"NLP diagnosis complete: {diagnosis.get('primary_diagnosis', 'N/A')}")
        return diagnosis

    def _generate_diagnosis(self, context: str, user_input: str,
                            retrieved_docs: list[dict] = None) -> dict:
        """Call LLM to generate diagnosis from context and input.

        Falls back to ClinicalFallbackEngine when LLM is unavailable
        """
        self._ensure_llm()

        prompt = RAG_PROMPT_TEMPLATE.format(context=context, user_input=user_input)

        try:
            if self._client and HAS_GENAI:
                response = self._client.models.generate_content(
                    model=settings.llm_model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                result = self._parse_llm_response(response.text)
                result = self._sanitize_llm_result(result)
                return self._normalize_urgency(result)
            else:
                nlp_logger.warning("No API key — using clinical fallback engine")
                return self._fallback.diagnose(user_input, retrieved_docs or [])
        except Exception as e:
            nlp_logger.error(f"LLM generation failed: {e}")
            return self._fallback.diagnose(user_input, retrieved_docs or [])

    def _parse_llm_response(self, response_text: str) -> dict:
        """Parse the JSON response from the LLM."""
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

        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract outermost JSON object with regex
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Fix common LLM JSON issues (unescaped quotes in strings)
        if match:
            fixed = match.group(0)
            # Replace problematic unescaped quotes inside string values
            # by removing non-essential quote characters within arrays
            fixed = re.sub(
                r'(?<=\w)"(?=\s*\()',  # quote before parentheses like ..." (Guideline
                "'",
                fixed,
            )
            fixed = re.sub(
                r'(?<=\))"(?=\s*[,\]])',  # quote after parentheses like (Guideline 1)"
                "'",
                fixed,
            )
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        # Strategy 4: Extract key fields manually with regex
        try:
            primary = re.search(r'"primary_diagnosis"\s*:\s*"([^"]*)"', text)
            conf = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
            explanation = re.search(r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
            severity = re.search(r'"severity"\s*:\s*"([^"]*)"', text)
            urgency = re.search(r'"urgency"\s*:\s*"([^"]*)"', text)
            actions = re.findall(r'"recommended_actions"\s*:\s*\[(.*?)\]', text, re.DOTALL)

            recommended = []
            if actions:
                recommended = re.findall(r'"((?:[^"\\]|\\.)*)"', actions[0])

            # Also extract question_answer
            qa = re.search(r'"question_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)

            result = {
                "primary_diagnosis": primary.group(1) if primary else "Analysis completed",
                "confidence": float(conf.group(1)) if conf else 0.5,
                "explanation": explanation.group(1) if explanation else text[:500],
                "question_answer": qa.group(1) if qa else "",
                "differential_diagnoses": [],
                "supporting_evidence": [],
                "recommended_actions": recommended if recommended else ["Consult a healthcare professional"],
                "severity": severity.group(1) if severity else "moderate",
                "urgency": urgency.group(1) if urgency else "routine",
            }
            nlp_logger.info(f"Parsed LLM response with regex fallback: {result['primary_diagnosis']}")
            return result
        except Exception as e2:
            nlp_logger.error(f"All JSON parsing strategies failed: {e2}\nRaw response: {response_text}")

        return {
            "primary_diagnosis": "Analysis completed",
            "confidence": 0.5,
            "explanation": response_text[:500],
            "question_answer": "",
            "differential_diagnoses": [],
            "supporting_evidence": [],
            "recommended_actions": ["Consult a healthcare professional"],
            "severity": "moderate",
            "urgency": "routine",
        }

    def _sanitize_llm_result(self, result) -> dict:
        """Coerce LLM output to the types the rest of the pipeline expects.

        The response is JSON but not schema-checked: fields can come back
        null, as strings ("85%") or on a 0-100 scale, which would otherwise
        surface as a 500 further down the pipeline.
        """
        if not isinstance(result, dict):
            raise ValueError(
                f"LLM returned {type(result).__name__}, expected a JSON object"
            )

        result["primary_diagnosis"] = str(
            result.get("primary_diagnosis") or "Analysis completed"
        )
        result["confidence"] = self._coerce_score(result.get("confidence"))
        for key in ("explanation", "question_answer"):
            result[key] = str(result.get(key) or "")

        for key in ("supporting_evidence", "recommended_actions"):
            value = result.get(key)
            if isinstance(value, str):
                value = [value]
            result[key] = [str(v) for v in value] if isinstance(value, list) else []

        differentials = result.get("differential_diagnoses")
        result["differential_diagnoses"] = [
            {
                "condition": str(d.get("condition") or "Unknown"),
                "probability": self._coerce_score(d.get("probability")),
            }
            for d in (differentials if isinstance(differentials, list) else [])
            if isinstance(d, dict)
        ]
        return result

    @staticmethod
    def _coerce_score(value) -> float:
        """Parse a 0-1 score, tolerating "85%" and 85.

        Unparseable values become 0.0 so the case is treated as low confidence.
        """
        try:
            if isinstance(value, str):
                value = value.strip().rstrip("%")
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(score):
            return 0.0
        if 1.0 < score <= 100.0:
            score /= 100.0
        return min(max(score, 0.0), 1.0)

    # Severity -> minimum urgency mapping.
    # Keeps the two fields consistent in the safe direction: urgency is
    # raised to what the severity implies, never lowered below what the
    # LLM asked for.
    # E.g. severity="critical" + urgency="routine" -> raised to "emergency"
    SEVERITY_URGENCY_MAP = {
        "critical": "emergency",
        "high": "urgent",
        "moderate": "routine",
        "low": "routine",
        "unknown": "routine",
    }

    # Urgency rank for comparison
    _URGENCY_RANK = {"routine": 0, "urgent": 1, "emergency": 2}

    def _normalize_urgency(self, result: dict) -> dict:
        """Raise LLM urgency to the severity-appropriate minimum.

        Clinical guideline:
          critical -> at least emergency
          high     -> at least urgent
          moderate -> routine or above
          low      -> routine or above

        If the LLM returns an urgency below what its own severity implies,
        raise it. An urgency above that level is kept: de-escalating an
        "emergency" is the unsafe direction for a triage aid.

        Also sanitizes invalid values (e.g. 'N/A') from LLM output.
        """
        # Sanitize: coerce invalid severity/urgency to safe defaults
        VALID_SEVERITIES = {"low", "moderate", "high", "critical", "unknown"}
        VALID_URGENCIES = {"routine", "urgent", "emergency"}

        severity = result.get("severity", "moderate")
        severity = severity.lower().strip() if isinstance(severity, str) else "moderate"
        if severity not in VALID_SEVERITIES:
            nlp_logger.warning(
                f"Invalid severity '{result.get('severity')}' from LLM, defaulting to 'unknown'"
            )
            severity = "unknown"
        result["severity"] = severity

        urgency = result.get("urgency", "routine")
        urgency = urgency.lower().strip() if isinstance(urgency, str) else "routine"
        if urgency not in VALID_URGENCIES:
            nlp_logger.warning(
                f"Invalid urgency '{result.get('urgency')}' from LLM, defaulting to 'routine'"
            )
            urgency = "routine"
        result["urgency"] = urgency

        min_urgency = self.SEVERITY_URGENCY_MAP.get(severity, "routine")
        min_rank = self._URGENCY_RANK.get(min_urgency, 0)
        current_rank = self._URGENCY_RANK.get(urgency, 0)

        if current_rank < min_rank:
            nlp_logger.info(
                f"Urgency raised: severity={severity} requires at least "
                f"urgency={min_urgency}, LLM returned {urgency} -> {min_urgency}"
            )
            result["urgency"] = min_urgency
            result["_urgency_raised"] = True

        return result

    def _empty_diagnosis(self, reason: str) -> dict:
        return {
            "primary_diagnosis": "Unable to provide diagnosis",
            "confidence": 0.0,
            "explanation": reason,
            "question_answer": "",
            "differential_diagnoses": [],
            "supporting_evidence": [],
            "recommended_actions": ["Please provide symptoms or a clinical question"],
            "severity": "unknown",
            "urgency": "routine",
            "input_type": "text",
            "retrieved_documents": [],
        }

    def initialize_guidelines(self):
        """Sync ChromaDB with the guideline files (no-op when already current)."""
        count = self.vector_store.sync_guidelines(settings.data_dir / "guidelines")
        if count:
            nlp_logger.info(f"Guideline index rebuilt: {count} chunks")
