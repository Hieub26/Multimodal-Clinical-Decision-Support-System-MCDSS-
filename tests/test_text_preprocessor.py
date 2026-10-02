import pytest

from app.core.nlp.text_preprocessor import TextPreprocessor


@pytest.fixture(scope="module")
def preprocessor():
    return TextPreprocessor()


@pytest.mark.parametrize("text, active, negated", [
    # Contractions keep their negation once punctuation is stripped
    ("I don't have chest pain", [], ["chest pain"]),
    ("Patient doesnt have fever", [], ["fever"]),
    # The nearest trigger decides, not the first one in the window
    ("no fever but has no chest pain", [], ["fever", "chest pain"]),
    ("no fever but has cough", ["cough"], ["fever"]),
    # A decimal point is not a sentence end
    ("no fever above 38.5 or cough", [], ["fever", "cough"]),
])
def test_negation(preprocessor, text, active, negated):
    result = preprocessor.preprocess(text)
    assert result["extracted_symptoms"] == active
    assert result["negated_symptoms"] == negated


@pytest.mark.parametrize("text, active, negated", [
    # "no" attaches to the next term; after a comma a new symptom is reported
    ("no fever, chest pain radiating to left arm", ["chest pain"], ["fever"]),
    ("no fever, has severe chest pain radiating to left arm", ["chest pain"], ["fever"]),
    ("no fever, cough and chest pain", ["cough", "chest pain"], ["fever"]),
    ("cough without fever, chest pain", ["cough", "chest pain"], ["fever"]),
    # ...unless an "or" marks the whole run as one negated list
    ("no fever, cough, or chest pain", [], ["fever", "cough", "chest pain"]),
    ("no fever, cough or chest pain", [], ["fever", "cough", "chest pain"]),
    ("no fever or chills, chest pain since morning", ["chest pain"], ["fever", "chills"]),
    # "denies" governs the list it introduces
    ("denies fever, cough, chest pain", [], ["fever", "cough", "chest pain"]),
    ("denies fever, chills, nausea, and vomiting", [], ["fever", "chills", "nausea", "vomiting"]),
    ("denies fever, shortness of breath or wheezing", [], ["fever", "shortness of breath", "wheezing"]),
    # ...but not a clause that follows it
    ("denies fever, chest pain radiating to left arm", ["chest pain"], ["fever"]),
    ("denies fever, has cough", ["cough"], ["fever"]),
    ("denies fever, cough. has chest pain", ["chest pain"], ["fever", "cough"]),
])
def test_negation_scope_across_commas(preprocessor, text, active, negated):
    result = preprocessor.preprocess(text)
    assert result["extracted_symptoms"] == active
    assert result["negated_symptoms"] == negated


@pytest.mark.parametrize("text, active, family", [
    ("mother had a stroke last year, I have a mild headache", ["headache"], ["stroke"]),
    ("father died of a heart attack and now I have chest pain", ["chest pain"], ["heart attack"]),
    ("my mother had diabetes and my father had hypertension", [], ["diabetes", "hypertension"]),
    ("family history of diabetes, hypertension and stroke. now has chest pain",
     ["chest pain"], ["diabetes", "hypertension", "stroke"]),
    ("family history of heart failure, I have a cough", ["cough"], ["heart failure"]),
    ("diabetes runs in my family, I have a headache", ["headache"], ["diabetes"]),
    ("brother and sister both had asthma, I have wheezing", ["wheezing"], ["asthma"]),
    # A relative's clause ends at the comma: what follows is read as the patient's
    ("mother had a stroke, headache and nausea", ["headache", "nausea"], ["stroke"]),
    ("last year my mother had a stroke, I have a headache", ["headache"], ["stroke"]),
    ("my mother has a history of stroke, I have a headache", ["headache"], ["stroke"]),
    # Possessive, dated: "mother's stroke was last year"
    ("my mother's stroke was last year, I have a headache", ["headache"], ["stroke"]),
    # A relative's chronic condition is background once the patient reports
    # symptoms of their own
    ("my mother has diabetes, I have a headache", ["headache"], ["diabetes"]),
    ("my mother's diabetes is well controlled, I have blurred vision",
     ["blurred vision"], ["diabetes"]),
    ("My mom has high blood pressure and my dad has diabetes. I have had headaches for a week",
     ["headache"], ["hypertension", "diabetes"]),
    ("my mother's stroke was last year and she has diabetes. I have a cough",
     ["cough"], ["stroke", "diabetes"]),
    # Note style
    ("mother: diabetes, father: hypertension. I have chest pain",
     ["chest pain"], ["diabetes", "hypertension"]),
    ("FHx: MI in father at 55. Pt reports chest pain",
     ["chest pain"], ["myocardial infarction"]),
    # History and a current complaint in the same sentence
    ("my mother had a stroke last year and has chest pain", ["chest pain"], ["stroke"]),
    ("my mother had a stroke last year and she is having a seizure", ["seizure"], ["stroke"]),
])
def test_family_history_is_not_a_patient_symptom(preprocessor, text, active, family):
    result = preprocessor.preprocess(text)
    assert result["extracted_symptoms"] == active
    assert result["family_history"] == family


@pytest.mark.parametrize("text, active", [
    # A relative merely being mentioned does not take the symptom away
    ("my wife says my chest pain is getting worse", ["chest pain"]),
    # The relative is the person being described
    ("my father is having chest pain and sweating", ["chest pain", "sweating"]),
    ("my mother has chest pain", ["chest pain"]),
    ("my mother had chest pain and sweating", ["chest pain", "sweating"]),
    ("my mother has diabetes and chest pain", ["diabetes", "chest pain"]),
    ("my father has diabetes and he is confused and sweating", ["diabetes", "sweating"]),
    ("my mother's chest pain is getting worse", ["chest pain"]),
    ("my father has chest pain since last month", ["chest pain"]),
    ("my son had a seizure", ["seizure"]),
    ("my husband had a stroke", ["stroke"]),
    # ...or the event is happening now
    ("my mother had a stroke this morning", ["stroke"]),
    ("my father just had a heart attack", ["heart attack"]),
    ("my mother had a seizure 2 hours ago", ["seizure"]),
    # A relative's infection is an exposure, not background
    ("my mother has tuberculosis, I have a cough", ["tuberculosis", "cough"]),
])
def test_relative_as_the_patient_is_not_family_history(preprocessor, text, active):
    result = preprocessor.preprocess(text)
    assert result["extracted_symptoms"] == active
    assert result["family_history"] == []


def test_negated_family_history_counts_as_negated(preprocessor):
    result = preprocessor.preprocess("no family history of stroke, has headache")
    assert result["negated_symptoms"] == ["stroke"]
    assert result["family_history"] == []
    assert result["extracted_symptoms"] == ["headache"]


def test_plural_symptoms_are_recognized(preprocessor):
    result = preprocessor.preprocess("I have had headaches and two seizures, no fevers")
    assert result["extracted_symptoms"] == ["headache", "seizure"]
    assert result["negated_symptoms"] == ["fever"]


def test_hr_after_number_is_a_duration(preprocessor):
    result = preprocessor.preprocess("chest pain for 3 hr, hr 110")
    assert "3 hours" in result["expanded_text"]
    assert "heart rate 110" in result["expanded_text"]
    assert result["durations"] == ["for 3 hours"]
