"""
Question Understanding module for clinical questions.
Parses clinical questions to extract intent, entities, and build
optimized queries for vector store retrieval.
"""

import re
from app.utils.logger import nlp_logger


# Intent classification rules, checked in this order: the first intent with a
# matching pattern wins.
#
# Most clinical questions open with "what is/are", so that pattern says
# nothing about the intent and comes last: "What is the treatment for
# pneumonia?" is a treatment question. The specific intents are ordered so
# that a question naming two of them goes to the narrower one ("What are the
# side effects of the treatment for ...?" asks about side effects).
#
# Patterns are anchored on word boundaries: "cause" must not match
# "because", nor "treat" match "untreated".
INTENT_RULES = [
    # Asking which condition explains the complaint, even when the word
    # "cause" is used ("What could be causing my chest pain?")
    ("diagnosis", [
        r"\bwhat (?:condition|disease|disorder|illness)\b",
        r"\bwhat (?:is|could be|might be|may be) (?:causing|wrong)\b",
        r"\bcould (?:this|it|these|that) be\b",
        r"\bdo i have\b",
        r"\bdifferential\b",
    ]),
    ("drug_interaction", [
        r"\binteract",
        r"\bcontraindicat",
        r"\bside[- ]effects?\b",
        r"\badverse\b",
        r"\b(?:can|should|may) (?:i|we|they|you|he|she) (?:take|use|combine)\b.*\b(?:with|and)\b",
        r"\b(?:take|taken|use|used|combine|combined) together\b",
    ]),
    ("prognosis", [
        r"\bprognos[ie]s\b",
        r"\boutlook\b",
        r"\bsurviv(?:al|e)\b",
        r"\blife expectancy\b",
        r"\bhow long (?:does|will|do|until|before|can)\b",
        r"\brecovery\b",
        r"\bwill (?:i|he|she|it|they|this) (?:recover|get better|go away|heal|come back)\b",
        r"\bcomplications?\b",
    ]),
    ("prevention", [
        r"\bprevent",
        r"\bavoid",
        r"\bprotect",
        r"\bscreen(?:ing|ed)?\b",
        r"\bvaccin",
        r"\bprophyla",
    ]),
    ("risk_factors", [
        r"\brisk factors?\b",
        r"\b(?:at|increased?|higher|high) (?:the )?risk\b",
        r"\brisk of\b",
        r"\bcauses?\b",
        r"\bcaused by\b",
        r"\betiolog",
        r"\bpredispos",
        r"\bsusceptib",
        r"\bheredit",
        r"\bgenetic",
        r"\bwhy (?:do|does|did|am|is|are)\b",
    ]),
    ("treatment", [
        r"\btreat(?:s|ed|ing|ments?)?\b",
        r"\btherap(?:y|ies|eutic)",
        r"\bmedications?\b",
        r"\bmedicines?\b",
        r"\bdrugs?\b",
        r"\bantibiotics?\b",
        r"\bprescri",
        r"\bmanag(?:e|es|ed|ing|ement)\b",
        r"\bdos(?:e|es|age|ing)\b",
    ]),
    ("diagnosis", [
        r"\bdiagnos",
        r"\bis (?:this|it) (?:a sign|an indication|related to)\b",
        r"\b(?:signs?|symptoms?) of\b",
        r"\bwhat (?:is|are|could be|might be)\b",
    ]),
]

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
        self.intent_rules = INTENT_RULES
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
        # The question is patient text: its length is logged, never its content
        nlp_logger.info(f"Analyzing question ({len(question)} characters)")

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

        nlp_logger.info(
            f"Question analysis: intent={intent}, "
            f"entities={sum(len(found) for found in entities.values())}"
        )
        return result

    def _classify_intent(self, question: str) -> str:
        """Classify the clinical intent of the question."""
        for intent, patterns in self.intent_rules:
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
        entities["keywords"] = sorted(set(medical_keywords))

        return entities

    def _build_search_query(self, question: str, intent: str, entities: dict) -> str:
        """Build optimized search query for vector store."""
        query_parts = [question]

        if intent != "general":
            query_parts.append(f"clinical {intent.replace('_', ' ')}")

        if entities.get("conditions"):
            query_parts.append("conditions: " + ", ".join(entities["conditions"]))

        if entities.get("body_parts"):
            query_parts.append("affecting " + ", ".join(entities["body_parts"]))

        return " ".join(query_parts)
