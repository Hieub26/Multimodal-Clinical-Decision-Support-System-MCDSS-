"""
Question Understanding module for clinical questions.
Parses clinical questions to extract intent, entities, and build
optimized queries for vector store retrieval.
"""

import re
from app.utils.logger import nlp_logger


# Intent classification patterns
INTENT_PATTERNS = {
    "diagnosis": [
        r"what (?:is|are|could be|might be)",
        r"diagnos[ei]",
        r"what (?:condition|disease|disorder)",
        r"differential",
        r"could (?:this|it|these) be",
        r"is (?:this|it) (?:a sign|an indication|related to)",
    ],
    "treatment": [
        r"how (?:to|do (?:you|we)) treat",
        r"treatment (?:for|of|option)",
        r"therap[iy]",
        r"medication (?:for|to)",
        r"prescri[bp]",
        r"manage(?:ment)?",
        r"what (?:drug|medicine)",
    ],
    "prognosis": [
        r"prognosis",
        r"outloo?k",
        r"surviv(?:al|e)",
        r"recover[iy]",
        r"how long",
        r"life expectancy",
    ],
    "risk_factors": [
        r"risk (?:factor|of)",
        r"cause[sd]?",
        r"etiology",
        r"predispos",
        r"susceptib",
        r"what (?:causes|leads to)",
    ],
    "drug_interaction": [
        r"interact(?:ion)?",
        r"contraindic",
        r"side effect",
        r"adverse",
        r"combin(?:e|ation|ing)",
        r"(?:can|should) (?:i|we|they) take .* (?:with|and)",
    ],
    "prevention": [
        r"preven(?:t|tion)",
        r"avoid",
        r"protect",
        r"screen(?:ing)?",
        r"vaccin",
        r"prophylax",
    ],
}

# Medical entity patterns
BODY_PARTS = [
    "head", "neck", "chest", "abdomen", "back", "arm", "leg", "foot",
    "hand", "knee", "hip", "shoulder", "elbow", "wrist", "ankle",
    "throat", "lung", "heart", "liver", "kidney", "stomach", "brain",
    "eye", "ear", "nose", "mouth", "skin", "bone", "muscle", "joint",
    "spine", "pelvis", "rib", "skull", "femur", "tibia",
]

# Disease / condition keywords for entity extraction
# Sorted longest-first so multi-word conditions match before sub-words
CONDITION_KEYWORDS = sorted([
    # Cardiovascular
    "heart attack", "myocardial infarction", "cardiac arrest", "heart failure",
    "coronary artery disease", "arrhythmia", "atrial fibrillation",
    "cardiomyopathy", "endocarditis", "hypertension",
    # Cerebrovascular / Neurological
    "stroke", "cerebrovascular accident", "transient ischemic attack",
    "meningitis", "encephalitis", "epilepsy", "aneurysm",
    # Pulmonary
    "pneumonia", "pulmonary embolism", "pneumothorax", "asthma",
    "chronic obstructive pulmonary disease", "tuberculosis", "lung cancer",
    "respiratory failure", "pulmonary edema",
    # Emergency / Critical
    "sepsis", "anaphylaxis", "hemorrhage", "shock",
    # GI
    "appendicitis", "pancreatitis", "cholecystitis", "cirrhosis",
    "gastrointestinal bleeding", "bowel obstruction", "peritonitis",
    # Renal
    "kidney failure", "renal failure", "kidney stone",
    # Metabolic
    "diabetes", "diabetic ketoacidosis", "thyroid storm", "hypoglycemia",
    # Oncological
    "cancer", "tumor", "malignancy", "lymphoma", "leukemia",
    # Infectious
    "influenza", "covid", "hiv", "hepatitis", "malaria", "dengue",
    # Thrombotic
    "deep vein thrombosis", "thrombosis", "embolism",
], key=len, reverse=True)


class QuestionUnderstanding:
    """Parses and understands clinical questions for better retrieval."""

    def __init__(self):
        self.intent_patterns = INTENT_PATTERNS
        self.body_parts = BODY_PARTS
        self.condition_keywords = CONDITION_KEYWORDS
        nlp_logger.info("QuestionUnderstanding module initialized")

    def analyze(self, question: str) -> dict:
        """
        Analyze a clinical question to extract intent, entities, and
        generate an optimized search query.

        Args:
            question: Raw clinical question from user

        Returns:
            Dictionary with intent, entities, and search query
        """
        nlp_logger.info(f"Analyzing question: {question[:100]}...")

        question_clean = question.lower().strip()

        # Extract intent
        intent = self._classify_intent(question_clean)

        # Extract medical entities
        entities = self._extract_entities(question_clean)

        # Build optimized search query
        search_query = self._build_search_query(question, intent, entities)

        result = {
            "original_question": question,
            "intent": intent,
            "entities": entities,
            "search_query": search_query,
        }

        nlp_logger.info(f"Question analysis: intent={intent}, entities={entities}")
        return result

    def _classify_intent(self, question: str) -> str:
        """Classify the clinical intent of the question."""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    return intent
        return "general"  # Default intent

    def _extract_entities(self, question: str) -> dict:
        """Extract medical entities from the question."""
        entities = {
            "body_parts": [],
            "conditions": [],
            "keywords": [],
        }

        # Extract body parts
        for part in self.body_parts:
            if re.search(r"\b" + re.escape(part) + r"\b", question):
                entities["body_parts"].append(part)

        # Extract disease / condition names (longest-first to avoid sub-matches)
        for condition in self.condition_keywords:
            if re.search(r"\b" + re.escape(condition) + r"\b", question):
                entities["conditions"].append(condition)

        # Extract important medical keywords
        medical_keywords = re.findall(
            r"\b(?:pain|fever|infection|inflammation|chronic|acute|severe|mild|"
            r"moderate|persistent|intermittent|bilateral|unilateral|progressive|"
            r"sudden|gradual|recurring|worsening|improving)\b",
            question,
        )
        entities["keywords"] = list(set(medical_keywords))

        return entities

    def _build_search_query(self, question: str, intent: str, entities: dict) -> str:
        """Build optimized search query for vector store."""
        query_parts = [question]

        if intent != "general":
            query_parts.append(f"clinical {intent}")

        if entities.get("conditions"):
            query_parts.append("conditions: " + ", ".join(entities["conditions"]))

        if entities.get("body_parts"):
            query_parts.append("affecting " + ", ".join(entities["body_parts"]))

        return " ".join(query_parts)
