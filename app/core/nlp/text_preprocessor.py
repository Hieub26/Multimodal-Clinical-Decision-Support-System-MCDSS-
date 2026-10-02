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

# Disease / condition keywords for extraction (includes red-flag conditions)
# These complement SYMPTOM_KEYWORDS so the pipeline can recognize disease
# names mentioned in user input, not just symptoms.
DISEASE_KEYWORDS = [
    # Cardiovascular
    "heart attack", "myocardial infarction", "cardiac arrest", "heart failure",
    "congestive heart failure", "coronary artery disease", "arrhythmia",
    "atrial fibrillation", "cardiomyopathy", "endocarditis", "hypertension",
    # Cerebrovascular / Neurological
    "stroke", "cerebrovascular accident", "transient ischemic attack",
    "meningitis", "encephalitis", "epilepsy", "aneurysm",
    # Pulmonary
    "pneumonia", "pulmonary embolism", "pneumothorax", "asthma",
    "chronic obstructive pulmonary disease", "tuberculosis", "lung cancer",
    "respiratory failure", "pulmonary edema",
    # Emergency / Critical
    "sepsis", "anaphylaxis", "hemorrhage", "shock",
    "disseminated intravascular coagulation",
    # Gastrointestinal
    "appendicitis", "pancreatitis", "cholecystitis", "cirrhosis",
    "gastrointestinal bleeding", "bowel obstruction", "peritonitis",
    # Renal
    "kidney failure", "renal failure", "nephrolithiasis", "kidney stone",
    # Endocrine / Metabolic
    "diabetes", "diabetic ketoacidosis", "thyroid storm", "hypoglycemia",
    # Oncological
    "cancer", "tumor", "malignancy", "lymphoma", "leukemia",
    # Infectious
    "influenza", "covid", "hiv", "hepatitis", "malaria", "dengue",
    # Musculoskeletal
    "fracture", "osteoporosis", "rheumatoid arthritis",
    # Thrombotic
    "deep vein thrombosis", "thrombosis", "embolism",
]

# Long-term, non-communicable conditions. In the present tense they are a
# person's background ("my mother has diabetes"), not an event in progress.
# Infections are left out on purpose: a relative's tuberculosis is an
# exposure that matters for the patient's own diagnosis.
CHRONIC_CONDITIONS = {
    "hypertension", "diabetes", "asthma",
    "chronic obstructive pulmonary disease", "coronary artery disease",
    "heart failure", "congestive heart failure", "cardiomyopathy",
    "arrhythmia", "atrial fibrillation", "epilepsy",
    "cancer", "lung cancer", "tumor", "malignancy", "lymphoma", "leukemia",
    "cirrhosis", "osteoporosis", "rheumatoid arthritis",
}

# Backward Negation triggers (preceding window up to 40 chars)
BACKWARD_NEGATION_TRIGGERS = [
    "no", "not", "denies", "denied", "without", "never", "negative for",
    "rules out", "ruled out", "free of", "no evidence of", "absent", "nothing",
    "no history of", "no signs of", "no symptoms of",
]
BACKWARD_NEGATION_WINDOW = 40

# Forward Negation triggers (following window tight 15 - 25 chars)
FORWARD_NEGATION_TRIGGERS = [
    "is ruled out", "ruled out", "is absent", "was ruled out",
    "is negative", "negative", "unlikely", "has resolved", "resolved",
]
FORWARD_NEGATION_WINDOW = 25

# Termination terms that reset negation scope
NEGATION_TERMINATORS = ["but", "however", "except", "although", "yet", "still"]

# Sentence/clause end (a period inside a number like 38.5 does not count)
_SENTENCE_END = re.compile(r"[.;](?!\d)")

# Hard end of a negation's scope between the trigger and the mention:
# a terminator word (whole word: "but", not "attributed") or a sentence end.
# Commas are handled separately, see _scope_crosses_commas.
_NEGATION_SCOPE_BREAK = re.compile(
    r"\b(?:" + "|".join(NEGATION_TERMINATORS) + r")\b|" + _SENTENCE_END.pattern
)

# Triggers that govern a whole list: "denies fever, cough, chest pain".
# The remaining triggers ("no", "not", "without", ...) attach to the next
# term only. "no fever, chest pain" most likely reports chest pain, so after a
# comma they stay in force solely in an explicit or-list:
# "no fever, cough, or chest pain".
LIST_NEGATION_TRIGGERS = {
    "denies", "denied", "negative for", "rules out", "ruled out",
    "free of", "no evidence of", "no history of", "no signs of", "no symptoms of",
}

_OR_WORD = re.compile(r"\b(?:or|nor)\b")
_CONJUNCTION = re.compile(r"\b(?:and|or|nor)\b")

# Words showing that a comma-separated chunk is a clause of its own (a new
# statement about the patient) rather than one more item of a list.
_CLAUSE_CUE = re.compile(
    r"\b(?:but|however|except|although|yet|still|"
    r"i|im|me|my|he|she|they|we|patient|"
    r"has|have|had|having|is|are|was|were|am|"
    r"reports?|reported|reporting|complains?|complained|complaining|"
    r"presents?|presented|presenting|experienc\w+|develop\w+|"
    r"feels?|feeling|felt|started|starting|began|noticed|"
    r"now|also|just|currently|today|yesterday|since|for|with|"
    r"radiat\w+|worse|worsening|persist\w*|"
    r"severe|mild|moderate|intense)\b|\d"
)

# --- Family history -------------------------------------------------------
# A relative can be history ("mother had a stroke last year") or the person
# the text is about ("my father is having a stroke"). The wording alone does
# not always tell which, so family_context() reports how strong the evidence
# for "history" is and each caller applies its own policy: symptom extraction
# accepts a plain past tense, the red-flag gate only a stated or dated history.
#
# Children and spouses are left out of the relatives: they are usually the
# patient being described by a caregiver.
_FAMILY_MEMBER = (
    r"(?:mother|father|mom|mum|dad|brother|sister|"
    r"grandmother|grandfather|grandma|grandpa|uncle|aunt|cousin|"
    r"parents?|siblings?|grandparents?|relatives?)"
)
_FAMILY_MEMBER_WORD = re.compile(r"\b" + _FAMILY_MEMBER + r"(?:['’]?s)?\b")
# Relative as the subject of a verb: "mother ...", "sisters both ...",
# "mother, who ..."
_MEMBER_SUBJECT = (
    r"\b" + _FAMILY_MEMBER + r"s?(?:,?\s+who)?\s+(?:(?:also|both|once)\s+)?"
)
# ...or a pronoun standing in for one named earlier in the sentence:
# "my mother's stroke was last year and she has diabetes"
_PRONOUN_SUBJECT = r"\b(?:he|she|they)\s+(?:(?:also|both|once)\s+)?"
_PAST_VERBS = (
    r"(?:had|died|passed away|suffered|(?:was|were) diagnosed(?: with)?|"
    r"(?:has|have) (?:a )?(?:history|hx) of)\b"
)
_PRESENT_VERBS = r"(?:has|have|suffers|lives)\b"

# (kind, pattern, needs a relative named earlier in the sentence).
# Most specific first: on the same subject the earlier trigger wins.
_FAMILY_TRIGGERS = (
    # "family history of diabetes, hypertension and stroke": governs a list
    ("explicit", re.compile(
        r"\b(?:family (?:medical )?(?:history|hx)|fhx?)\b"
        r"(?:\s+(?:is\s+)?(?:significant|positive|notable|remarkable)\s+for)?"
    ), False),
    # "mother had a stroke", "father has a history of ..."
    ("past", re.compile(_MEMBER_SUBJECT + _PAST_VERBS), False),
    ("past", re.compile(_PRONOUN_SUBJECT + _PAST_VERBS), True),
    # "mother has diabetes"
    ("present", re.compile(_MEMBER_SUBJECT + _PRESENT_VERBS), False),
    ("present", re.compile(_PRONOUN_SUBJECT + _PRESENT_VERBS), True),
    # No verb, so no tense: "mother's stroke" (up to two words before the
    # mention; the apostrophe is optional because text cleaning strips it)
    # and the note style "mother: diabetes"
    ("nominal", re.compile(
        r"\b" + _FAMILY_MEMBER + r"(?:['’]s|s['’]?)\s+(?:[\w-]+\s+){0,2}$"
    ), False),
    ("nominal", re.compile(r"\b" + _FAMILY_MEMBER + r"s?\s*:"), False),
)
# After the mention: "diabetes runs in my family"
_FAMILY_AFTER_MENTION = re.compile(
    r"\s*(?:also\s+)?runs?\s+in\s+(?:the|my|our|his|her)\s+family\b"
)

# Between a family trigger and the mention: the text has turned to the patient
_PATIENT_SUBJECT = re.compile(
    r"\b(?:i|im|me|my|myself|patient|pt|but|however|although)\b"
)
# ...or carries on about the relative in a new statement: "... and she has ..."
_RELATIVE_PRONOUN = re.compile(r"\b(?:he|she|they|who)\b")
# After a past-tense trigger a present-tense verb does the same:
# "mother had a stroke last year and has chest pain"
_PRESENT_VERB = re.compile(
    r"\b(?:has|have|is|are|feels?|complains?|reports?|can|cannot)\b"
)
# Start of the clause a relative's trigger sits in (searched backwards)
_CLAUSE_START = re.compile(r",|" + _SENTENCE_END.pattern + r"|\b(?:and|but)\b")
# Where that clause stops after the mention: punctuation, a conjunction, or
# the text turning to someone ("... and now I have ...")
_FAMILY_CLAUSE_END = re.compile(
    r",|" + _SENTENCE_END.pattern
    + r"|\b(?:and|but|i|im|me|my|patient|pt|he|she|they)\b"
)
# The relative's problem is happening now or still going on: not history
_CURRENT_CUE = re.compile(
    r"\b(?:now|currently|today|tonight|just|suddenly|earlier|yesterday|still|"
    r"again|ongoing|since|started|starting|began|having|experiencing|getting|"
    r"worsening|worse|this (?:morning|afternoon|evening|week)|last night|"
    r"for the (?:last|past)|(?:minutes?|hours?|days?|weeks?) ago)\b"
)
# The text itself places the event in the past
_HISTORICAL_CUE = re.compile(
    r"\b(?:(?:months?|years?|decades?) ago|last (?:month|year|decade)|"
    r"in (?:19|20)\d{2}|at (?:the )?age(?: of)? \d+|aged \d+|"
    r"when (?:i|he|she|they) (?:was|were)|long ago|in the past|previously|"
    r"(?:history|hx) of|died|passed away)\b"
)
FAMILY_HISTORY_WINDOW = 80

# Contexts (see family_context) where the text itself states or dates the
# history. Only these are strong enough for the red-flag gate: the tense alone
# is not ("my mother had a stroke" may be a call for help).
STATED_FAMILY_HISTORY_CONTEXTS = {"explicit", "historical"}

# Negative contractions, with or without the apostrophe ("doesn't", "doesnt").
_CONTRACTIONS = [
    (re.compile(r"\bcan['’]t\b"), "can not"),
    (re.compile(r"\bwon['’]t\b"), "will not"),
    (re.compile(
        r"\b(do|does|did|is|are|was|were|has|have|had|would|should|could)n['’]?t\b"
    ), r"\1 not"),
]


def normalize_contractions(text: str) -> str:
    """Spell out negative contractions in lowercased text.

    Without this, stripping punctuation turns "don't" into "dont" and the
    negation is lost ("I don't have chest pain" -> active chest pain).
    """
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    return text


def _is_list_segment(segment: str) -> bool:
    """A comma-separated chunk that reads as further list items, not a new clause.

    Symptom names run to about three words ("shortness of breath"); anything
    longer is a description unless a conjunction joins several items.
    """
    if _CLAUSE_CUE.search(segment):
        return False
    return bool(_CONJUNCTION.search(segment)) or len(segment.split()) <= 3


def _scope_crosses_commas(
    text: str, scope_start: int, mention_end: int, needs_or: bool
) -> bool:
    """Whether a scope opened at scope_start still covers a mention that sits
    after one or more commas.

    Every chunk between the trigger and the mention (the mention's own chunk
    included, up to the next comma) must read as a list item. With needs_or
    the list must also be closed by "or"/"nor" at or after the mention.

    When the text is ambiguous this answers False: reading a denied symptom
    as present costs a cautious result, the reverse hides a real symptom.
    """
    sentence_end = _SENTENCE_END.search(text, mention_end)
    clause = text[scope_start:sentence_end.start() if sentence_end else len(text)]
    segments = clause.split(",")
    mention_index = text[scope_start:mention_end].count(",")

    if not needs_or:
        return all(_is_list_segment(s) for s in segments[1:mention_index + 1])

    for i in range(mention_index, len(segments)):
        if _OR_WORD.search(segments[i]):
            return all(_is_list_segment(s) for s in segments[1:i + 1])
    return False


def is_negated(text: str, start: int, end: int) -> bool:
    """Bi-directional NegEx check for the mention at text[start:end].

    Expects lowercased text with contractions already normalized.
    """
    # Backward: nearest occurrence of each trigger in the preceding window
    back_start = max(0, start - BACKWARD_NEGATION_WINDOW)
    preceding_text = text[back_start:start]
    if back_start > 0 and text[back_start - 1].isalnum() and text[back_start].isalnum():
        # Window opens mid-word ("pia|no fever"): drop the partial token
        preceding_text = preceding_text.partition(" ")[2]
    window_offset = start - len(preceding_text)

    for trigger in BACKWARD_NEGATION_TRIGGERS:
        trig_matches = list(
            re.finditer(r"\b" + re.escape(trigger) + r"\b", preceding_text)
        )
        if not trig_matches:
            continue
        trigger_end = trig_matches[-1].end()
        text_between = preceding_text[trigger_end:]
        if _NEGATION_SCOPE_BREAK.search(text_between):
            continue
        if "," in text_between and not _scope_crosses_commas(
            text, window_offset + trigger_end, end,
            needs_or=trigger not in LIST_NEGATION_TRIGGERS,
        ):
            continue
        return True

    # Forward: tight following window
    following_text = text[end:end + FORWARD_NEGATION_WINDOW]
    for trigger in FORWARD_NEGATION_TRIGGERS:
        trig_match = re.search(r"\b" + re.escape(trigger) + r"\b", following_text)
        if trig_match:
            text_between = following_text[:trig_match.start()]
            if not _NEGATION_SCOPE_BREAK.search(text_between):
                return True

    return False


def family_context(text: str, start: int, end: int) -> str | None:
    """How the mention at text[start:end] relates to a relative of the patient.

    Returns:
        None          not governed by a relative: the patient's own
        "current"     a relative's problem that is happening now
        "present"     a relative's condition in the present tense
                      ("mother has diabetes", "mother's diabetes")
        "past"        a relative's event in the past tense, undated
                      ("mother had a stroke")
        "historical"  a relative's event the text dates in the past
                      ("mother had a stroke last year", "father died of ...")
        "explicit"    stated as family history
                      ("family history of ...", "... runs in my family")

    Expects lowercased text.
    """
    if _FAMILY_AFTER_MENTION.match(text, end):
        return "explicit"

    # Look back within the same sentence for the nearest family trigger
    window_start = max(0, start - FAMILY_HISTORY_WINDOW)
    preceding_text = text[window_start:start]
    sentence_ends = list(_SENTENCE_END.finditer(preceding_text))
    if sentence_ends:
        window_start += sentence_ends[-1].end()
        preceding_text = text[window_start:start]

    nearest = None  # (trigger start, -priority, kind, trigger end)
    for priority, (kind, pattern, needs_relative) in enumerate(_FAMILY_TRIGGERS):
        matches = [
            m for m in pattern.finditer(preceding_text)
            if not needs_relative
            or _FAMILY_MEMBER_WORD.search(preceding_text, 0, m.start())
        ]
        if matches:
            candidate = (matches[-1].start(), -priority, kind, matches[-1].end())
            nearest = max(nearest, candidate) if nearest else candidate
    if nearest is None:
        return None

    trigger_start, _, kind, trigger_end = nearest
    text_between = preceding_text[trigger_end:]
    if _PATIENT_SUBJECT.search(text_between):
        return None

    if kind == "explicit":
        # "family history of a, b and c": carries across list items, but not
        # into a new statement ("family history of stroke and now has ...")
        if _CLAUSE_CUE.search(text_between.partition(",")[0]):
            return None
        if "," in text_between and not _scope_crosses_commas(
            text, window_start + trigger_end, end, needs_or=False
        ):
            return None
        return "explicit"

    # "<relative> had/has ...": governs its own clause only
    if "," in text_between:
        return None
    # A further statement about the same relative is about now, whatever came
    # before it: "mother had a stroke last year and she is having a seizure"
    if _RELATIVE_PRONOUN.search(text_between) or (
        kind == "past" and _PRESENT_VERB.search(text_between)
    ):
        return "current"

    clause_starts = list(_CLAUSE_START.finditer(preceding_text, 0, trigger_start))
    clause_start = window_start + (clause_starts[-1].end() if clause_starts else 0)
    clause_end = _FAMILY_CLAUSE_END.search(text, end)
    clause = text[clause_start:clause_end.start() if clause_end else len(text)]

    if _CURRENT_CUE.search(clause):
        return "current"
    # A date settles it for a past-tense or verbless mention. A present tense
    # with a date ("has chest pain since last month") is an ongoing problem,
    # never history.
    if kind != "present" and _HISTORICAL_CUE.search(clause):
        return "historical"
    return "past" if kind == "past" else "present"


class TextPreprocessor:
    """Production-grade Hybrid Clinical NLP preprocessor."""

    def __init__(self):
        self.abbreviations = MEDICAL_ABBREVIATIONS
        self.synonyms = SYNONYM_MAP
        self.symptom_keywords = set(SYMPTOM_KEYWORDS)
        self.disease_keywords = set(DISEASE_KEYWORDS)
        # Combined vocabulary for extraction pipeline
        self.all_clinical_keywords = self.symptom_keywords | self.disease_keywords
        self._spacy_nlp = None
        nlp_logger.info("TextPreprocessor initialized")

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
                "family_history": [],
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
        active_symptoms, negated_symptoms, family_history = (
            self._extract_symptoms_with_negation(expanded)
        )

        # Step 5: Tier 2 - scispaCy / SpaCy fallback if Tier 1 returned no active symptoms
        if not active_symptoms:
            # Tier 2 has no context detection of its own: do not let it bring
            # back what Tier 1 already ruled out.
            excluded = set(negated_symptoms) | set(family_history)
            spacy_entities = [
                e for e in self._extract_tier2_spacy(expanded) if e not in excluded
            ]
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
            "family_history": family_history,
            "severity_qualifiers": severity_qualifiers,
            "durations": durations,
            "search_query": search_query,
            "symptom_count": len(active_symptoms),
        }

        nlp_logger.info(
            f"Extraction complete: active={active_symptoms}, "
            f"negated={negated_symptoms}, family_history={family_history}, "
            f"severity={severity_qualifiers}, durations={durations}"
        )
        return result

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace, lowercase, and clean non-clinical symbols."""
        text = text.lower().strip()
        text = normalize_contractions(text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s/\-.,;:()]", "", text)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Replace medical abbreviations using regex word boundaries."""
        # After a number "hr" is a duration ("for 3 hr"), not heart rate
        text = re.sub(r"(\d)\s*hrs?\b", r"\1 hours", text)
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
        """Extract duration expressions"""
        pattern = r"\b(?:for\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:day|days|week|weeks|month|months|hour|hours|year|years)\b|\bsince\s+\w+\b"
        matches = re.findall(pattern, text, re.IGNORECASE)
        return list(dict.fromkeys(matches))

    def _extract_symptoms_with_negation(
        self, text: str
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Tier 1: Extract symptoms using non-overlapping interval match selection
        and bi-directional (backward & forward 15-25 char) negation detection.

        Returns:
            (active, negated, family history) — each mention lands in exactly
            one list; only the active ones describe the patient's current state.
        """
        text_lower = text.lower()

        # Step A: Apply synonym normalization to text
        normalized_text = text_lower
        for synonym, canonical in self.synonyms.items():
            pattern = r"\b" + re.escape(synonym) + r"\b"
            normalized_text = re.sub(pattern, canonical, normalized_text)

        # Step B: Find all raw candidate matches
        all_keywords = list(self.all_clinical_keywords | set(self.synonyms.values()))
        raw_candidates = []

        for kw in all_keywords:
            # Plural allowed: "headaches", "seizures", "kidney stones"
            pattern = r"\b" + re.escape(kw) + r"s?\b"
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

        # Step D: Bi-directional Negation Detection + family-history context
        active_symptoms = []
        negated_symptoms = []
        family_history = []

        contexts = [
            "negated"
            if is_negated(normalized_text, cand["start"], cand["end"])
            else family_context(normalized_text, cand["start"], cand["end"])
            for cand in selected_candidates
        ]
        # Does the text report anything about the patient themselves? If not,
        # a relative in the present tense is the person being described.
        patient_reports_own = None in contexts

        for cand, context in zip(selected_candidates, contexts):
            symptom = cand["symptom"]
            if context == "negated":
                negated_symptoms.append(symptom)
            elif self._is_family_history(symptom, context, patient_reports_own):
                family_history.append(symptom)
            else:
                active_symptoms.append(symptom)

        # Remove duplicates while preserving exact order
        ordered_active = list(dict.fromkeys(active_symptoms))
        ordered_negated = list(dict.fromkeys(negated_symptoms))
        ordered_family = list(dict.fromkeys(family_history))

        return ordered_active, ordered_negated, ordered_family

    def _is_family_history(
        self, symptom: str, context: str | None, patient_reports_own: bool
    ) -> bool:
        """Decide from its family_context whether a mention is family history."""
        if context in STATED_FAMILY_HISTORY_CONTEXTS:
            return True
        if context == "past":
            # Undated past tense: a relative's disease is history, but a bare
            # symptom ("my mother had chest pain") is more likely a caregiver
            # describing the patient.
            return symptom in self.disease_keywords
        if context in ("present", "current"):
            # "my mother has diabetes, I have a headache": her chronic
            # condition is background to the patient's own complaint.
            return symptom in CHRONIC_CONDITIONS and patient_reports_own
        return False

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
