"""
Clinical NLP Text Preprocessor.
Normalizes medical abbreviations, performs bi-directional negation detection (NegEx),
synonym mapping, 2-tier entity extraction with longest non-overlapping interval selection,
and duration/severity metadata extraction using Enums.
"""

import re
from enum import Enum
from typing import Any
from app.utils.logger import nlp_logger


class SeverityLevel(str, Enum):
    """Clinical severity levels for extracted metadata."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


# Mapping from natural language qualifiers to canonical SeverityLevel Enum values
SEVERITY_MAPPING: dict[str, SeverityLevel] = {
    "slight": SeverityLevel.MILD,
    "mild": SeverityLevel.MILD,
    "low": SeverityLevel.MILD,
    "moderate": SeverityLevel.MODERATE,
    "medium": SeverityLevel.MODERATE,
    "intense": SeverityLevel.SEVERE,
    "severe": SeverityLevel.SEVERE,
    "high": SeverityLevel.SEVERE,
    "extreme": SeverityLevel.CRITICAL,
    "critical": SeverityLevel.CRITICAL,
    "life-threatening": SeverityLevel.CRITICAL,
}

# Common medical abbreviations and their expansions
MEDICAL_ABBREVIATIONS = {
    "bp": "blood pressure",
    "hr": "heart rate",
    "rr": "respiratory rate",
    "temp": "temperature",
    "hx": "history",
    "dx": "diagnosis",
    "tx": "treatment",
    "rx": "prescription",
    "sx": "symptoms",
    "pt": "patient",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "ha": "headache",
    "n/v": "nausea and vomiting",
    "abd": "abdominal",
    "bilat": "bilateral",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "cad": "coronary artery disease",
    "copd": "chronic obstructive pulmonary disease",
    "uti": "urinary tract infection",
    "uri": "upper respiratory infection",
    "chf": "congestive heart failure",
    "mi": "myocardial infarction",
    "cva": "cerebrovascular accident",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "gi": "gastrointestinal",
    "bmi": "body mass index",
    "ecg": "electrocardiogram",
    "ekg": "electrocardiogram",
    "ct": "computed tomography",
    "mri": "magnetic resonance imaging",
    "wbc": "white blood cell",
    "rbc": "red blood cell",
    "hgb": "hemoglobin",
    "plt": "platelet",
}

# Synonym mapping to canonical clinical terms
SYNONYM_MAP = {
    "breathlessness": "shortness of breath",
    "dyspnoea": "dyspnea",
    "difficulty breathing": "dyspnea",
    "trouble breathing": "dyspnea",
    "febrile": "fever",
    "high temperature": "fever",
    "pyrexia": "fever",
    "feverish": "fever",
    "cephalea": "headache",
    "head pain": "headache",
    "emesis": "vomiting",
    "throw up": "vomiting",
    "stomach ache": "abdominal pain",
    "belly pain": "abdominal pain",
    "abdominalache": "abdominal pain",
    "high blood pressure": "hypertension",
    "high blood sugar": "diabetes",
}

# Canonical symptom keywords for extraction
SYMPTOM_KEYWORDS = [
    "pain", "ache", "fever", "cough", "fatigue", "weakness", "nausea",
    "vomiting", "diarrhea", "constipation", "headache", "dizziness",
    "shortness of breath", "chest pain", "palpitation", "swelling",
    "rash", "itching", "bleeding", "numbness", "tingling", "seizure",
    "confusion", "anxiety", "depression", "insomnia", "weight loss",
    "weight gain", "appetite", "thirst", "urination", "blurred vision",
    "hearing loss", "tinnitus", "sore throat", "runny nose", "congestion",
    "wheezing", "joint pain", "muscle pain", "back pain", "abdominal pain",
    "difficulty swallowing", "heartburn", "bloating", "chills", "sweating",
    "tremor", "stiffness", "cramp", "bruising", "pallor", "jaundice",
    "sputum", "hemoptysis", "night sweats", "edema", "dyspnea", "orthopnea",
]

# Backward Negation triggers (preceding window up to 40 chars)
BACKWARD_NEGATION_TRIGGERS = [
    "no", "not", "denies", "denied", "without", "never", "negative for",
    "rules out", "ruled out", "free of", "no evidence of", "absent",
]

# Forward Negation triggers (following window tight 15 - 25 chars)
FORWARD_NEGATION_TRIGGERS = [
    "is ruled out", "ruled out", "is absent", "was ruled out",
    "is negative", "negative", "unlikely", "has resolved", "resolved",
]

# Termination terms that reset negation scope
NEGATION_TERMINATORS = ["but", "however", "except", "although", "yet", "still", "."]


class TextPreprocessor:
    """Production-grade Hybrid Clinical NLP preprocessor."""

    def __init__(self):
        self.abbreviations = MEDICAL_ABBREVIATIONS
        self.synonyms = SYNONYM_MAP
        self.symptom_keywords = set(SYMPTOM_KEYWORDS)
        self._spacy_nlp = None
        nlp_logger.info("TextPreprocessor initialized with bi-directional NegEx & non-overlapping matcher")

    def preprocess(self, text: str) -> dict[str, Any]:
        """
        Full clinical NLP preprocessing pipeline.

        Args:
            text: Raw user symptom statement

        Returns:
            Dictionary with cleaned, expanded text, active symptoms, negated symptoms,
            durations, severity qualifiers, and search query.
        """
        if not text:
            return {
                "original_text": "",
                "cleaned_text": "",
                "expanded_text": "",
                "extracted_symptoms": [],
                "negated_symptoms": [],
                "severity_qualifiers": [],
                "durations": [],
                "search_query": "",
                "symptom_count": 0,
            }

        nlp_logger.info(f"Preprocessing text: {text[:100]}...")

        # Step 1: Basic cleaning
        cleaned = self._clean_text(text)

        # Step 2: Expand medical abbreviations using regex word boundaries
        expanded = self._expand_abbreviations(cleaned)

        # Step 3: Extract severity Enum levels and duration metadata
        severity_qualifiers = self._extract_severity(expanded)
        durations = self._extract_durations(expanded)

        # Step 4: Tier 1 - Bi-directional NegEx with non-overlapping match selection
        active_symptoms, negated_symptoms = self._extract_symptoms_with_negation(expanded)

        # Step 5: Tier 2 - scispaCy / SpaCy fallback if Tier 1 returned no active symptoms
        if not active_symptoms:
            spacy_entities = self._extract_tier2_spacy(expanded)
            if spacy_entities:
                active_symptoms = list(dict.fromkeys(spacy_entities))

        # Step 6: Build optimized RAG search query
        search_query = self._build_search_query(expanded, active_symptoms)

        result = {
            "original_text": text,
            "cleaned_text": cleaned,
            "expanded_text": expanded,
            "extracted_symptoms": active_symptoms,
            "negated_symptoms": negated_symptoms,
            "severity_qualifiers": severity_qualifiers,
            "durations": durations,
            "search_query": search_query,
            "symptom_count": len(active_symptoms),
        }

        nlp_logger.info(
            f"Extraction complete: active={active_symptoms}, "
            f"negated={negated_symptoms}, severity={severity_qualifiers}, durations={durations}"
        )
        return result

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace, lowercase, and clean non-clinical symbols."""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s/\-.,;:()]", "", text)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Replace medical abbreviations using regex word boundaries."""
        for abbr, expansion in self.abbreviations.items():
            pattern = r"\b" + re.escape(abbr) + r"\b"
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text

    def _extract_severity(self, text: str) -> list[SeverityLevel]:
        """Extract and map severity qualifiers to SeverityLevel Enums."""
        found: list[SeverityLevel] = []
        for word, level in SEVERITY_MAPPING.items():
            if re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE):
                if level not in found:
                    found.append(level)
        return found

    def _extract_durations(self, text: str) -> list[str]:
        """Extract duration expressions (e.g., '3 days', 'for 2 weeks', 'since yesterday')."""
        pattern = r"\b(?:for\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:day|days|week|weeks|month|months|hour|hours|year|years)\b|\bsince\s+\w+\b"
        matches = re.findall(pattern, text, re.IGNORECASE)
        return list(dict.fromkeys(matches))

    def _extract_symptoms_with_negation(self, text: str) -> tuple[list[str], list[str]]:
        """
        Tier 1: Extract symptoms using non-overlapping interval match selection
        and bi-directional (backward & forward 15-25 char) negation detection.
        """
        text_lower = text.lower()

        # Step A: Apply synonym normalization to text
        normalized_text = text_lower
        for synonym, canonical in self.synonyms.items():
            pattern = r"\b" + re.escape(synonym) + r"\b"
            normalized_text = re.sub(pattern, canonical, normalized_text)

        # Step B: Find all raw candidate matches
        all_keywords = list(self.symptom_keywords | set(self.synonyms.values()))
        raw_candidates = []

        for kw in all_keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            for match in re.finditer(pattern, normalized_text):
                raw_candidates.append({
                    "symptom": kw,
                    "start": match.start(),
                    "end": match.end(),
                })

        # Step C: Non-overlapping Interval Matcher (Longest Match Dominance)
        # Sort by length descending O(N log N)
        raw_candidates.sort(key=lambda x: (x["end"] - x["start"], -x["start"]), reverse=True)

        selected_candidates = []
        for cand in raw_candidates:
            c_start, c_end = cand["start"], cand["end"]
            overlap = False
            for prev in selected_candidates:
                p_start, p_end = prev["start"], prev["end"]
                # Overlap condition: max(start1, start2) < min(end1, end2)
                if max(c_start, p_start) < min(c_end, p_end):
                    overlap = True
                    break
            if not overlap:
                selected_candidates.append(cand)

        # Sort selected candidates into original text order
        selected_candidates.sort(key=lambda x: x["start"])

        # Step D: Bi-directional Negation Detection
        active_symptoms = []
        negated_symptoms = []

        for cand in selected_candidates:
            symptom = cand["symptom"]
            start_pos, end_pos = cand["start"], cand["end"]
            is_negated = False

            # D1. Backward Negation Check (Preceding window up to 40 chars)
            back_start = max(0, start_pos - 40)
            preceding_text = normalized_text[back_start:start_pos].strip()

            for trigger in BACKWARD_NEGATION_TRIGGERS:
                trig_pattern = r"\b" + re.escape(trigger) + r"\b"
                trig_match = re.search(trig_pattern, preceding_text)
                if trig_match:
                    text_between = preceding_text[trig_match.end():]
                    if not any(term in text_between for term in NEGATION_TERMINATORS):
                        is_negated = True
                        break

            # D2. Forward Negation Check (Tight following window 15 - 25 chars)
            if not is_negated:
                fwd_end = min(len(normalized_text), end_pos + 25)
                following_text = normalized_text[end_pos:fwd_end].strip()

                for trigger in FORWARD_NEGATION_TRIGGERS:
                    trig_pattern = r"\b" + re.escape(trigger) + r"\b"
                    trig_match = re.search(trig_pattern, following_text)
                    if trig_match:
                        text_between = following_text[:trig_match.start()]
                        if not any(term in text_between for term in NEGATION_TERMINATORS):
                            is_negated = True
                            break

            if is_negated:
                negated_symptoms.append(symptom)
            else:
                active_symptoms.append(symptom)

        # Remove duplicates while preserving exact order
        ordered_active = list(dict.fromkeys(active_symptoms))
        ordered_negated = list(dict.fromkeys(negated_symptoms))

        return ordered_active, ordered_negated

    def _extract_tier2_spacy(self, text: str) -> list[str]:
        """Tier 2: scispaCy / spaCy Clinical NER entity extraction fallback."""
        try:
            import spacy
            if self._spacy_nlp is None:
                try:
                    self._spacy_nlp = spacy.load("en_ner_bc5cdr_md")
                except Exception:
                    self._spacy_nlp = spacy.load("en_core_web_sm")
            doc = self._spacy_nlp(text)
            return [ent.text.lower() for ent in doc.ents if ent.label_ in ("DISEASE", "CHEMICAL", "PROBLEM")]
        except Exception:
            return []

    def _build_search_query(self, text: str, active_symptoms: list[str]) -> str:
        """Build an optimized RAG vector store search query."""
        if active_symptoms:
            symptom_part = "active symptoms: " + ", ".join(active_symptoms)
            return f"{text}. {symptom_part}"
        return text
