"""
Clinical Fallback Engine — 4-Layer Resilient Diagnosis Pipeline.

When the LLM (Gemini) is unavailable (quota exhausted, network down, API error),
this engine replaces the naive first-match-and-break rule-based fallback with a
production-grade pipeline that mimics clinical cluster reasoning:

    Layer 1: Weighted Symptom Scoring  — each symptom contributes to multiple diseases
    Layer 2: Symptom Combination Rules — feature interactions (cough+fever ≠ cough alone)
    Layer 3: Retrieval Boost           — leverage retrieved guidelines even without LLM
    Layer 4: Disease Promotion         — replace generic labels with specific entities

The engine also produces a full explainability breakdown (fallback_reasoning) for
debugging and audit purposes.
"""

import math
import re
from collections import defaultdict
from app.core.nlp.text_preprocessor import TextPreprocessor
from app.utils.logger import nlp_logger


# ---------------------------------------------------------------------------
# DISEASE PROFILES — Metadata for every disease the engine can output
# ---------------------------------------------------------------------------

DISEASE_PROFILES = {
    "pneumonia": {
        "display": "Pneumonia",
        "severity": "high",
        "urgency": "urgent",
        "guideline_keywords": ["pneumonia", "phlegm", "infiltrate", "consolidation"],
    },
    "acs": {
        "display": "Acute Coronary Syndrome",
        "severity": "critical",
        "urgency": "emergency",
        "guideline_keywords": ["coronary", "angina", "stemi", "nstemi", "troponin"],
    },
    "copd": {
        "display": "COPD",
        "severity": "moderate",
        "urgency": "urgent",
        "guideline_keywords": ["copd", "chronic obstructive", "bronchitis", "emphysema"],
    },
    "heart_failure": {
        "display": "Heart Failure",
        "severity": "high",
        "urgency": "urgent",
        "guideline_keywords": ["heart failure", "cardiomegaly", "bnp", "ejection fraction"],
    },
    "tuberculosis": {
        "display": "Tuberculosis",
        "severity": "high",
        "urgency": "urgent",
        "guideline_keywords": ["tuberculosis", "mycobacterium", "hemoptysis", "tb"],
    },
    "asthma": {
        "display": "Asthma",
        "severity": "moderate",
        "urgency": "urgent",
        "guideline_keywords": ["asthma", "bronchospasm", "bronchodilator", "wheezing"],
    },
    "covid19": {
        "display": "COVID-19",
        "severity": "high",
        "urgency": "urgent",
        "guideline_keywords": ["sars-cov-2", "covid", "anosmia", "ground-glass"],
    },
    "hypertension": {
        "display": "Hypertension",
        "severity": "moderate",
        "urgency": "routine",
        "guideline_keywords": ["hypertension", "blood pressure", "hypertensive", "mmhg"],
    },
    "diabetes": {
        "display": "Diabetes Mellitus",
        "severity": "moderate",
        "urgency": "routine",
        "guideline_keywords": ["diabetes", "glucose", "hba1c", "insulin", "metformin"],
    },
    "pneumothorax": {
        "display": "Pneumothorax",
        "severity": "high",
        "urgency": "emergency",
        "guideline_keywords": ["pneumothorax", "collapsed lung", "pleural line"],
    },
    "pleural_effusion": {
        "display": "Pleural Effusion",
        "severity": "moderate",
        "urgency": "urgent",
        "guideline_keywords": ["pleural effusion", "costophrenic", "meniscus", "thoracentesis"],
    },
    "pulmonary_edema": {
        "display": "Pulmonary Edema",
        "severity": "high",
        "urgency": "emergency",
        "guideline_keywords": ["pulmonary edema", "kerley", "frothy sputum", "cephalization"],
    },
    "anemia": {
        "display": "Anemia",
        "severity": "moderate",
        "urgency": "routine",
        "guideline_keywords": ["anemia", "hemoglobin", "iron deficiency"],
    },
    "gerd": {
        "display": "GERD",
        "severity": "low",
        "urgency": "routine",
        "guideline_keywords": ["gerd", "heartburn", "reflux", "esophagus"],
    },
    "uti": {
        "display": "Urinary Tract Infection",
        "severity": "moderate",
        "urgency": "routine",
        "guideline_keywords": ["urinary tract infection", "dysuria", "uti", "pyelonephritis"],
    },
    "appendicitis": {
        "display": "Acute Appendicitis",
        "severity": "high",
        "urgency": "emergency",
        "guideline_keywords": ["appendicitis", "appendix", "mcburney"],
    },
    "stroke": {
        "display": "Stroke (CVA)",
        "severity": "critical",
        "urgency": "emergency",
        "guideline_keywords": ["stroke", "cva", "facial droop", "hemiparesis", "thrombolysis"],
    },
    "skin_condition": {
        "display": "Dermatological Condition",
        "severity": "low",
        "urgency": "routine",
        "guideline_keywords": ["eczema", "psoriasis", "dermatitis", "melanoma", "rash"],
    },
    "hypothyroidism": {
        "display": "Hypothyroidism",
        "severity": "moderate",
        "urgency": "routine",
        "guideline_keywords": ["hypothyroidism", "levothyroxine", "low t4", "high tsh"],
    },
    "hyperthyroidism": {
        "display": "Hyperthyroidism",
        "severity": "moderate",
        "urgency": "routine",
        "guideline_keywords": ["hyperthyroidism", "thyrotoxicosis", "low tsh", "methimazole"],
    },
    "atelectasis": {
        "display": "Atelectasis",
        "severity": "moderate",
        "urgency": "routine",
        "guideline_keywords": ["atelectasis", "collapse", "fissure", "incentive spirometry"],
    },
    "emphysema": {
        "display": "Emphysema",
        "severity": "moderate",
        "urgency": "urgent",
        "guideline_keywords": ["emphysema", "hyperinflation", "barrel chest"],
    },
}


# ---------------------------------------------------------------------------
# SYMPTOM WEIGHTS — Layer 1: symptom → disease contributions
# ---------------------------------------------------------------------------
# No generic "pain" entry — only specific pain terms to avoid false positives.

SYMPTOM_WEIGHTS = {
    "fever":               {"pneumonia": 2, "covid19": 2, "tuberculosis": 2, "uti": 1, "appendicitis": 1},
    "cough":               {"pneumonia": 2, "covid19": 2, "copd": 2, "asthma": 1, "tuberculosis": 2},
    "shortness of breath": {"pneumonia": 3, "asthma": 3, "copd": 3, "heart_failure": 2, "acs": 2,
                            "pulmonary_edema": 3, "covid19": 2, "anemia": 1},
    "chest pain":          {"acs": 3, "pneumonia": 1, "pneumothorax": 2, "gerd": 1, "pleural_effusion": 1},
    "sputum":              {"pneumonia": 3, "copd": 2, "tuberculosis": 2},
    "wheezing":            {"asthma": 3, "copd": 2, "pulmonary_edema": 1},
    "headache":            {"hypertension": 2, "stroke": 1, "anemia": 1},
    "nausea":              {"appendicitis": 2, "acs": 1, "gerd": 2},
    "vomiting":            {"appendicitis": 2, "gerd": 1},
    "dizziness":           {"anemia": 2, "hypertension": 1, "stroke": 2},
    "fatigue":             {"anemia": 3, "heart_failure": 2, "diabetes": 2, "hypothyroidism": 2, "copd": 1},
    "weight loss":         {"tuberculosis": 3, "diabetes": 2, "hyperthyroidism": 2},
    "rash":                {"skin_condition": 3},
    "joint pain":          {"skin_condition": 1},
    "abdominal pain":      {"appendicitis": 3, "gerd": 2, "uti": 1},
    "back pain":           {},
    "muscle pain":         {"covid19": 1},
    "palpitation":         {"hyperthyroidism": 3, "acs": 1, "anemia": 1},
    "swelling":            {"heart_failure": 3, "pleural_effusion": 1},
    "chills":              {"pneumonia": 2, "covid19": 2, "uti": 1, "tuberculosis": 1},
    "sweating":            {"acs": 2, "tuberculosis": 2, "hyperthyroidism": 1},
    "bleeding":            {"anemia": 2},
    "numbness":            {"diabetes": 2, "stroke": 2},
    "confusion":           {"stroke": 3, "pneumonia": 1},
    "heartburn":           {"gerd": 3},
    "bloating":            {"gerd": 1},
    "constipation":        {"hypothyroidism": 2, "gerd": 1},
    "itching":             {"skin_condition": 2},
    "congestion":          {"covid19": 2},
    "sore throat":         {"covid19": 2},
    "hemoptysis":          {"tuberculosis": 4, "pneumonia": 1},
    "night sweats":        {"tuberculosis": 3},
    "edema":               {"heart_failure": 3, "pleural_effusion": 1},
    "dyspnea":             {"pneumonia": 2, "heart_failure": 2, "copd": 2, "asthma": 2},
    "orthopnea":           {"heart_failure": 3, "pulmonary_edema": 2},
    "pallor":              {"anemia": 2},
    "jaundice":            {},
    # The entries below complete the table from the symptom lists of the
    # guideline library (data/guidelines): before them the hallmark symptoms
    # of several diseases the engine can output carried no weight at all.
    # UTI: dysuria, frequency, urgency, suprapubic pain; pyelonephritis adds
    # fever, chills, flank pain
    "dysuria":             {"uti": 4},
    "frequent urination":  {"uti": 3, "diabetes": 3},
    "hematuria":           {"uti": 2},
    "cloudy urine":        {"uti": 2},
    "pelvic pain":         {"uti": 2, "appendicitis": 1},
    "flank pain":          {"uti": 3},
    # Diabetes: polyuria, polydipsia, polyphagia, weight loss, fatigue,
    # blurred vision, slow-healing wounds
    "thirst":              {"diabetes": 3},
    "increased appetite":  {"diabetes": 2, "hyperthyroidism": 1},
    "blurred vision":      {"diabetes": 2, "hypertension": 1, "stroke": 1},
    "slow healing":        {"diabetes": 3},
    # Asthma: wheezing, shortness of breath, chest tightness, cough
    "chest tightness":     {"asthma": 3, "acs": 2, "copd": 1},
    # COVID-19: new loss of taste or smell; diarrhea (also listed for pneumonia)
    "loss of taste/smell": {"covid19": 4},
    "diarrhea":            {"covid19": 1, "pneumonia": 1},
    # Tuberculosis: loss of appetite
    "loss of appetite":    {"tuberculosis": 2, "appendicitis": 1},
    # Heart failure: sudden weight gain
    "weight gain":         {"heart_failure": 2, "hypothyroidism": 2},
    # Hypertension, severe: nosebleeds
    "nosebleed":           {"hypertension": 2},
    # Stroke: facial droop, speech difficulty, loss of balance
    "facial droop":        {"stroke": 4},
    "slurred speech":      {"stroke": 4},
    "loss of balance":     {"stroke": 2},
    # Skin: plaques, scales, sores
    "skin lesion":         {"skin_condition": 3},
    "difficulty swallowing": {"gerd": 2},
}

# Two names of one symptom score as one, so that the combination rules below
# see it whichever word the patient or clinician used.
SYMPTOM_ALIASES = {
    "dyspnea": "shortness of breath",
}


# ---------------------------------------------------------------------------
# COMBINATION RULES — Layer 2: symptom clusters → bonus scores
# ---------------------------------------------------------------------------

COMBINATION_RULES = [
    # Pneumonia cluster
    ({"cough", "fever"},                                    {"pneumonia": 3}),
    ({"cough", "fever", "sputum"},                          {"pneumonia": 5}),
    ({"cough", "fever", "shortness of breath"},             {"pneumonia": 4, "covid19": 3}),
    ({"fever", "chills", "sputum"},                         {"pneumonia": 4}),

    # Cardiac cluster
    ({"chest pain", "shortness of breath"},                 {"acs": 4, "pulmonary_edema": 2}),
    ({"chest pain", "sweating", "nausea"},                  {"acs": 5}),
    ({"chest pain", "shortness of breath", "sweating"},     {"acs": 6}),

    # Tuberculosis cluster
    ({"cough", "weight loss", "fever"},                     {"tuberculosis": 4}),
    ({"cough", "hemoptysis"},                               {"tuberculosis": 5}),
    ({"night sweats", "weight loss"},                       {"tuberculosis": 3}),

    # Respiratory cluster
    ({"wheezing", "shortness of breath"},                   {"asthma": 4, "copd": 3}),
    ({"wheezing", "cough"},                                 {"asthma": 3}),

    # Heart failure cluster
    ({"shortness of breath", "swelling", "fatigue"},        {"heart_failure": 5}),
    ({"shortness of breath", "edema"},                      {"heart_failure": 4}),

    # Stroke cluster
    ({"confusion", "numbness"},                             {"stroke": 5}),
    ({"headache", "dizziness", "confusion"},                {"stroke": 4}),

    # GI cluster
    ({"abdominal pain", "fever", "nausea"},                 {"appendicitis": 4}),
    ({"heartburn", "chest pain"},                           {"gerd": 4}),
    ({"heartburn", "difficulty swallowing"},                {"gerd": 3}),

    # Urinary cluster
    ({"dysuria", "frequent urination"},                     {"uti": 4}),
    ({"dysuria", "fever"},                                  {"uti": 2}),
    ({"fever", "flank pain"},                               {"uti": 4}),

    # Diabetes cluster (the classic triad and its pairs)
    ({"frequent urination", "thirst"},                      {"diabetes": 5}),
    ({"thirst", "increased appetite"},                      {"diabetes": 3}),
    ({"thirst", "weight loss"},                             {"diabetes": 3}),

    # Asthma cluster
    ({"wheezing", "chest tightness"},                       {"asthma": 4}),

    # Stroke cluster
    ({"facial droop", "slurred speech"},                    {"stroke": 5}),

    # Skin cluster
    ({"rash", "itching"},                                   {"skin_condition": 3}),
    ({"skin lesion", "itching"},                            {"skin_condition": 2}),
]


# ---------------------------------------------------------------------------
# RETRIEVAL BOOST — Layer 3 constants
# ---------------------------------------------------------------------------

RETRIEVAL_BOOST_FACTOR = 3.0
# Skip docs below this cosine similarity to prevent boost leak. Calibrated on
# the bundled guidelines with S-BioBert: symptom queries score ~0.48-0.59
# against relevant chunks, vague or off-topic text stays below ~0.33.
MIN_SIMILARITY_FOR_BOOST = 0.45
NEGATION_WINDOW_WORDS = 12

NEGATION_PATTERNS = [
    "no evidence of",
    "no evidence",
    "negative for",
    "unlikely",
    "without evidence of",
    "absence of",
    "not suggestive of",
    "rule out",
    "ruled out",
    "rules out",
    "excluded",
    "no sign of",
    "no signs of",
    "not consistent with",
]

# Single-word negation cues — catch expanded phrases like
# "no radiographic evidence of focal pneumonia" where multi-word
# patterns miss due to intervening adjectives/adverbs.
NEGATION_CUES = {"no", "not", "without", "negative", "absence", "unlikely"}


# ---------------------------------------------------------------------------
# DISEASE PROMOTION — Layer 4 constants
# ---------------------------------------------------------------------------

PROMOTION_MARGIN_ABSOLUTE = 2.0
PROMOTION_MARGIN_RELATIVE = 0.15

GENERIC_LABELS = {
    "Possible cardiac condition",
    "Respiratory condition",
    "Respiratory distress",
    "Possible infection",
    "Gastrointestinal condition",
    "General clinical assessment needed",
    "Gastrointestinal disturbance",
    "Musculoskeletal condition",
    "Dermatological condition",
    "Vestibular / neurological condition",
    "Tension headache / Migraine",
}

# ---------------------------------------------------------------------------
# CONFIDENCE — Fallback should never be as confident as LLM
# ---------------------------------------------------------------------------
# Confidence is read off how clearly the evidence singles out one disease:
# the margin of the leading score over the runner-up, as a share of the
# leading score. It used to grow with the leading score alone, which made a
# long list of symptoms shared by three diseases look more certain than two
# symptoms that fit only one.
#
# The curve is fitted to the development split of a public benchmark
# (evaluation/fallback_eval.py): with at least two supporting symptoms the
# leading diagnosis was right in 97% of cases at a margin of 0.7 or more, in
# about half between 0.6 and 0.7, in a quarter between 0.5 and 0.6 and in
# one case in eight below that. It stays below the measured figure at the
# top on purpose (see the cap): the benchmark covers 9 of the 22 diseases
# the engine can name.

FALLBACK_CONFIDENCE_CAP = 0.75
# (relative margin, confidence), linear in between. The NLP confidence
# threshold of 0.70 is met from a margin of 0.7.
CONFIDENCE_BY_MARGIN = [(0.0, 0.10), (0.5, 0.25), (0.7, 0.70), (0.9, 0.75), (1.0, 0.75)]
# One symptom is not a diagnosis, however exclusively it points to a disease
SINGLE_SYMPTOM_CONFIDENCE_CAP = 0.60
# No reported symptom counts towards the disease: only retrieved guideline
# text raised it
NO_SYMPTOM_CONFIDENCE_CAP = 0.20


# ---------------------------------------------------------------------------
# STARTUP VALIDATION — fail fast on config errors
# ---------------------------------------------------------------------------

def _validate_config():
    """Assert all disease references in weights and rules exist in DISEASE_PROFILES.

    Runs at module import time so misconfigurations surface immediately
    on server start rather than as runtime KeyErrors mid-request.
    """
    all_referenced = set()

    for symptom, mapping in SYMPTOM_WEIGHTS.items():
        for disease in mapping:
            all_referenced.add(disease)

    for required_symptoms, bonus_map in COMBINATION_RULES:
        for disease in bonus_map:
            all_referenced.add(disease)

    orphans = all_referenced - set(DISEASE_PROFILES.keys())
    if orphans:
        raise ValueError(
            f"ClinicalFallback config error: diseases {orphans} "
            f"referenced in SYMPTOM_WEIGHTS/COMBINATION_RULES but missing "
            f"from DISEASE_PROFILES. Add them to DISEASE_PROFILES or remove "
            f"the references."
        )


_validate_config()

# Order among diseases with the same score: the more severe first, so that a
# tie is reported with the more cautious severity and urgency, then the order
# of DISEASE_PROFILES.
_SEVERITY_ORDER = {"critical": 0, "high": 1, "moderate": 2, "low": 3}
_TIE_ORDER = {
    key: (_SEVERITY_ORDER[profile["severity"]], position)
    for position, (key, profile) in enumerate(DISEASE_PROFILES.items())
}


# ---------------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------------

class ClinicalFallbackEngine:
    """4-layer resilient clinical fallback when LLM is unavailable."""

    def __init__(self, preprocessor: TextPreprocessor | None = None):
        self._preprocessor = preprocessor or TextPreprocessor()

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def diagnose(self, user_input: str, retrieved_docs: list[dict]) -> dict:
        """Run the full 4-layer fallback pipeline.

        Args:
            user_input: Preprocessed/expanded patient text.
            retrieved_docs: Documents retrieved from ChromaDB (may be empty).

        Returns:
            Diagnosis dict compatible with RAGEngine output format.
        """
        nlp_logger.info("Clinical fallback engine activated (LLM unavailable)")

        # Extract symptoms from the input
        preprocessed = self._preprocessor.preprocess(user_input)
        symptoms = {
            SYMPTOM_ALIASES.get(symptom, symptom)
            for symptom in preprocessed["extracted_symptoms"]
        }

        nlp_logger.info(f"Fallback symptoms: {len(symptoms)}")

        # Initialize scores for all known diseases
        scores = defaultdict(float)

        # --- Layer 1: Weighted Symptom Scoring ---
        scores_after_L1 = self._apply_weighted_scoring(scores, symptoms)

        # --- Layer 2: Symptom Combination Rules ---
        scores_after_L2, combination_deltas = self._apply_combination_rules(
            scores, symptoms
        )

        # --- Layer 3: Retrieval Boost (negation-aware) ---
        scores_after_L3, retrieval_deltas = self._apply_retrieval_boost(
            scores, retrieved_docs
        )

        # Snapshot final scores (only diseases with score > 0)
        final_scores = {k: v for k, v in scores.items() if v > 0}

        if not final_scores:
            nlp_logger.info("No disease scored — returning generic assessment")
            return self._empty_fallback(symptoms, preprocessed)

        # Rank diseases. Equal scores are common (one symptom often gives
        # the same weight to several diseases), so the order among them is
        # fixed: without it the leading disease followed the iteration order
        # of a set and changed from one server start to the next.
        ranked = sorted(
            final_scores.items(), key=lambda item: (-item[1], _TIE_ORDER[item[0]])
        )
        top_key, top_score = ranked[0]
        second_key = ranked[1][0] if len(ranked) > 1 else None
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # --- Layer 4: Disease Promotion (margin-guarded) ---
        promoted, was_promoted = self._promote_diagnosis(
            top_key, top_score, second_score
        )

        # The reported symptoms that count towards the leading disease
        supporting_symptoms = sorted(
            symptom for symptom in symptoms
            if SYMPTOM_WEIGHTS.get(symptom, {}).get(top_key, 0) > 0
        )

        confidence = self._confidence(
            top_score, second_score, len(supporting_symptoms)
        )

        profile = DISEASE_PROFILES[top_key]

        # Build explainability breakdown
        fallback_reasoning = {
            "method": "clinical_fallback_4layer",
            "weighted_scores": dict(scores_after_L1),
            "combination_bonus": combination_deltas,
            "retrieval_boost": retrieval_deltas,
            "final_scores": final_scores,
            "top_disease": top_key,
            "top_score": round(top_score, 2),
            "supporting_symptoms": supporting_symptoms,
            "runner_up_disease": second_key,
            "runner_up_score": round(second_score, 2),
            "promotion_applied": was_promoted,
            "promotion_margin_absolute": round(top_score - second_score, 2),
            "promotion_margin_relative": (
                round((top_score - second_score) / max(top_score, 1e-6), 3)
            ),
        }

        nlp_logger.info(
            f"Fallback diagnosis: {promoted} "
            f"(score={top_score:.1f}, runner_up={second_key}:{second_score:.1f}, "
            f"promoted={was_promoted})"
        )

        # Build differential diagnoses from ranked scores
        differentials = []
        for disease_key, score in ranked[1:4]:  # top 3 runners-up
            prob = round(score / max(top_score, 1e-6) * confidence, 2)
            differentials.append({
                "condition": DISEASE_PROFILES[disease_key]["display"],
                "probability": min(prob, confidence - 0.05),
            })

        return {
            "primary_diagnosis": promoted,
            "confidence": confidence,
            "differential_diagnoses": differentials,
            "explanation": (
                f"Based on {len(symptoms)} reported symptoms "
                f"({', '.join(sorted(symptoms)) if symptoms else 'general complaint'}), "
                f"the clinical fallback engine identified {promoted} as the most likely "
                f"condition (score: {top_score:.1f}). "
                f"This assessment used weighted symptom scoring, symptom cluster analysis, "
                f"and retrieval-based guideline matching. "
                f"{'Disease was promoted from a generic label. ' if was_promoted else ''}"
                f"Note: this is a fallback analysis — LLM-based reasoning was unavailable."
            ),
            "question_answer": "",
            "supporting_evidence": [
                f"Symptom: {s}" for s in sorted(symptoms)[:5]
            ],
            "recommended_actions": [
                "Consult a healthcare professional for proper evaluation",
                "Provide detailed medical history",
                "Consider relevant diagnostic tests",
            ],
            "severity": profile["severity"],
            "urgency": profile["urgency"],
            "fallback_reasoning": fallback_reasoning,
        }

    @staticmethod
    def _confidence(top_score: float, second_score: float, supporting: int) -> float:
        """Confidence in the leading disease (see CONFIDENCE_BY_MARGIN)."""
        margin = (top_score - second_score) / max(top_score, 1e-6)
        margin = min(max(margin, 0.0), 1.0)

        confidence = CONFIDENCE_BY_MARGIN[-1][1]
        for (low, at_low), (high, at_high) in zip(CONFIDENCE_BY_MARGIN, CONFIDENCE_BY_MARGIN[1:]):
            if margin <= high:
                confidence = at_low + (at_high - at_low) * (margin - low) / (high - low)
                break

        cap = FALLBACK_CONFIDENCE_CAP
        if supporting == 1:
            cap = SINGLE_SYMPTOM_CONFIDENCE_CAP
        elif supporting == 0:
            cap = NO_SYMPTOM_CONFIDENCE_CAP
        # Rounded down: a margin just short of a knot must not be lifted
        # onto it, and over the threshold it marks
        return math.floor(min(confidence, cap) * 100 + 1e-9) / 100

    # ---------------------------------------------------------------
    # Layer 1: Weighted Symptom Scoring
    # ---------------------------------------------------------------

    def _apply_weighted_scoring(
        self, scores: defaultdict, symptoms: set
    ) -> dict:
        """Score each disease based on individual symptom weights."""
        for symptom in symptoms:
            if symptom in SYMPTOM_WEIGHTS:
                for disease, weight in SYMPTOM_WEIGHTS[symptom].items():
                    scores[disease] += weight

        return {k: v for k, v in scores.items() if v > 0}

    # ---------------------------------------------------------------
    # Layer 2: Symptom Combination Rules
    # ---------------------------------------------------------------

    def _apply_combination_rules(
        self, scores: defaultdict, symptoms: set
    ) -> tuple[dict, dict]:
        """Apply bonus scores for recognized symptom clusters."""
        combination_deltas = defaultdict(float)

        for required_symptoms, bonus_map in COMBINATION_RULES:
            if required_symptoms.issubset(symptoms):
                for disease, bonus in bonus_map.items():
                    scores[disease] += bonus
                    combination_deltas[disease] += bonus

        return (
            {k: v for k, v in scores.items() if v > 0},
            dict(combination_deltas),
        )

    # ---------------------------------------------------------------
    # Layer 3: Retrieval Boost (negation-aware, word-based window)
    # ---------------------------------------------------------------

    def _apply_retrieval_boost(
        self, scores: defaultdict, retrieved_docs: list[dict]
    ) -> tuple[dict, dict]:
        """Boost disease scores using retrieved guideline documents.

        Uses a word-based negation window (NEGATION_WINDOW_WORDS) instead of a
        character-based one to handle varied sentence lengths more robustly.
        """
        retrieval_deltas = defaultdict(float)

        for doc in retrieved_docs:
            text = doc.get("text", "").lower()
            # Vector store uses cosine distance, so 1 - distance is the
            # cosine similarity (clamped: distance can exceed 1).
            similarity = min(max(1.0 - doc.get("distance", 1.0), 0.0), 1.0)

            # Gate: skip low-similarity docs to prevent boost leak
            if similarity < MIN_SIMILARITY_FOR_BOOST:
                continue

            for disease_key, profile in DISEASE_PROFILES.items():
                for keyword in profile["guideline_keywords"]:
                    if not self._has_affirmed_mention(text, keyword):
                        continue

                    # Use similarity² decay — low similarity decays quadratically
                    boost = (similarity ** 2) * RETRIEVAL_BOOST_FACTOR
                    scores[disease_key] += boost
                    retrieval_deltas[disease_key] += round(boost, 2)
                    break  # one boost per disease per document

        return (
            {k: v for k, v in scores.items() if v > 0},
            dict(retrieval_deltas),
        )

    @classmethod
    def _has_affirmed_mention(cls, text: str, keyword: str) -> bool:
        """True if the keyword occurs as a whole word at least once un-negated.

        Whole-word matching matters: as substrings "tb" hits "heartburn" and
        "uti" hits "routine".
        """
        pattern = r"\b" + re.escape(keyword) + r"\b"
        return any(
            not cls._is_negated(text, match.start())
            for match in re.finditer(pattern, text)
        )

    @staticmethod
    def _is_negated(text: str, kw_pos: int) -> bool:
        """Check if the keyword at kw_pos is negated using a two-tier word-based approach.

        Tier 1: Multi-word patterns (e.g., "no evidence of", "ruled out")
                — high precision, catches exact clinical negation phrases.
        Tier 2: Single-word cues (e.g., "no", "not", "without")
                — higher recall, catches expanded phrases like
                "no radiographic evidence of focal pneumonia" where
                multi-word patterns miss due to intervening words.

        Both tiers use a NEGATION_WINDOW_WORDS word window before the keyword.
        """
        # Get the text before the keyword, split into words,
        # then take the last NEGATION_WINDOW_WORDS words
        prefix_text = text[:kw_pos]
        prefix_words = prefix_text.split()
        window_words = prefix_words[-NEGATION_WINDOW_WORDS:]
        window_text = " ".join(window_words)

        # Tier 1: Multi-word negation patterns
        if any(neg in window_text for neg in NEGATION_PATTERNS):
            return True

        # Tier 2: Single-word negation cues
        if any(word.strip(".,;:()") in NEGATION_CUES for word in window_words):
            return True

        return False

    # ---------------------------------------------------------------
    # Layer 4: Disease Promotion (margin-guarded)
    # ---------------------------------------------------------------

    def _promote_diagnosis(
        self, top_key: str, top_score: float, second_score: float
    ) -> tuple[str, bool]:
        """Promote generic presentation labels to specific disease entities.

        Promotion is guarded by both absolute AND relative margins:
        - Absolute: top - second >= PROMOTION_MARGIN_ABSOLUTE
        - Relative: (top - second) / top >= PROMOTION_MARGIN_RELATIVE

        If either condition is met, the disease is promoted with confidence.
        Otherwise, it is still promoted but flagged as requiring further evaluation.
        """
        display = DISEASE_PROFILES[top_key]["display"]
        gap_absolute = top_score - second_score
        gap_relative = gap_absolute / max(top_score, 1e-6)

        is_confident = (
            gap_absolute >= PROMOTION_MARGIN_ABSOLUTE
            or gap_relative >= PROMOTION_MARGIN_RELATIVE
        )

        if is_confident:
            return display, True
        else:
            return f"{display} (requires further evaluation)", False

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def _empty_fallback(self, symptoms: set, preprocessed: dict) -> dict:
        """Return a minimal diagnosis when no disease scored."""
        return {
            "primary_diagnosis": "General clinical assessment needed",
            # The low end of CONFIDENCE_BY_MARGIN: nothing points anywhere
            "confidence": CONFIDENCE_BY_MARGIN[0][1],
            "differential_diagnoses": [
                {"condition": "Requires further evaluation", "probability": 0.05}
            ],
            "explanation": (
                f"Based on reported symptoms "
                f"({', '.join(sorted(symptoms)) if symptoms else 'general complaint'}), "
                f"the clinical fallback engine could not identify a specific condition. "
                f"Further clinical evaluation is recommended."
            ),
            "question_answer": "",
            "supporting_evidence": [f"Symptom: {s}" for s in sorted(symptoms)[:5]],
            "recommended_actions": [
                "Consult a healthcare professional for proper evaluation",
                "Provide detailed medical history",
                "Consider relevant diagnostic tests",
            ],
            "severity": "moderate",
            "urgency": "routine",
            "fallback_reasoning": {
                "method": "clinical_fallback_4layer",
                "weighted_scores": {},
                "combination_bonus": {},
                "retrieval_boost": {},
                "final_scores": {},
                "top_disease": None,
                "top_score": 0,
                "supporting_symptoms": [],
                "runner_up_disease": None,
                "runner_up_score": 0,
                "promotion_applied": False,
                "promotion_margin_absolute": 0,
                "promotion_margin_relative": 0,
            },
        }
