"""
Text Preprocessor for clinical symptom text.
Normalizes, cleans, and extracts key symptoms from raw user input.
"""

import re
from app.utils.logger import nlp_logger

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

# Symptom keywords for extraction
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
]


class TextPreprocessor:
    """Preprocesses clinical symptom text for downstream NLP tasks."""

    def __init__(self):
        self.abbreviations = MEDICAL_ABBREVIATIONS
        self.symptom_keywords = SYMPTOM_KEYWORDS
        nlp_logger.info("TextPreprocessor initialized")

    def preprocess(self, text: str) -> dict:
        """
        Full preprocessing pipeline for symptom text.

        Args:
            text: Raw symptom text from user

        Returns:
            Dictionary with cleaned text, extracted symptoms, and metadata
        """
        nlp_logger.info(f"Preprocessing text: {text[:100]}...")

        # Step 1: Basic cleaning
        cleaned = self._clean_text(text)

        # Step 2: Expand abbreviations
        expanded = self._expand_abbreviations(cleaned)

        # Step 3: Extract symptoms
        symptoms = self._extract_symptoms(expanded)

        # Step 4: Build search query
        search_query = self._build_search_query(expanded, symptoms)

        result = {
            "original_text": text,
            "cleaned_text": cleaned,
            "expanded_text": expanded,
            "extracted_symptoms": symptoms,
            "search_query": search_query,
            "symptom_count": len(symptoms),
        }

        nlp_logger.info(f"Extracted {len(symptoms)} symptoms: {symptoms}")
        return result

    def _clean_text(self, text: str) -> str:
        """Remove noise, normalize whitespace, lowercase."""
        # Lowercase
        text = text.lower().strip()
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep medical punctuation
        text = re.sub(r"[^\w\s/\-.,;:()]", "", text)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Replace known medical abbreviations with full terms."""
        words = text.split()
        expanded_words = []
        for word in words:
            clean_word = word.strip(".,;:()")
            if clean_word in self.abbreviations:
                expanded_words.append(self.abbreviations[clean_word])
            else:
                expanded_words.append(word)
        return " ".join(expanded_words)

    def _extract_symptoms(self, text: str) -> list[str]:
        """Extract recognized symptom keywords from text."""
        found_symptoms = []
        text_lower = text.lower()
        for symptom in self.symptom_keywords:
            if symptom in text_lower:
                found_symptoms.append(symptom)
        return list(set(found_symptoms))

    def _build_search_query(self, text: str, symptoms: list[str]) -> str:
        """Build an optimized search query for vector store retrieval."""
        if symptoms:
            symptom_part = "symptoms: " + ", ".join(symptoms)
            return f"{text}. {symptom_part}"
        return text
